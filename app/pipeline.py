import os
import shutil
import asyncio
import logging
from pathlib import Path
import httpx
from typing import Any, Dict, List, Optional
import math

from app import config
from app import llm
from app import qdrant_store
from app import db

logger = logging.getLogger(__name__)

# --- Extraction ---

async def extract_text(file_path: str) -> str:
    """Route file extraction based on extension."""
    ext = Path(file_path).suffix.lower()

    # Simple text-based files
    if ext in ['.txt', '.md', '.py', '.json', '.yml', '.yaml', '.csv', '.html']:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to read native text file {file_path}: {e}")
            raise

    # Binary / Office files go to Docling
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            with open(file_path, 'rb') as f:
                files = {'file': (os.path.basename(file_path), f)}
                # docling-serve usually has a /convert endpoint
                response = await client.post(f"{config.DOCLING_URL}/convert", files=files)
                response.raise_for_status()
                # Assuming docling returns JSON with the extracted text or markdown
                data = response.json()
                return data.get("text") or data.get("markdown", "")
    except Exception as e:
        logger.error(f"Docling extraction failed for {file_path}: {e}")
        raise

# --- Chunking ---

def chunk_text(text: str) -> List[str]:
    """Basic character-based chunking with overlap."""
    if not text:
        return []

    size = config.CHUNK_SIZE
    overlap = config.CHUNK_OVERLAP

    if size <= overlap:
        size = overlap + 500 # Sanity check fallback

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap

    return chunks

# --- Cleaning ---

async def clean_text(full_text: str) -> str:
    """Cleans a full text in blocks using the LLM with length checks to prevent hallucinations."""
    if not config.CLEAN_TEXT or not full_text.strip():
        return full_text

    prompt_template = db.get_prompt("clean_text")
    if not prompt_template:
        return full_text

    block_size = config.CLEAN_BLOCK_CHARS
    blocks = [full_text[i:i+block_size] for i in range(0, len(full_text), block_size)]

    cleaned_blocks = []

    for block in blocks:
        if not block.strip():
            cleaned_blocks.append(block)
            continue

        prompt = prompt_template.format(text=block)
        try:
            cleaned = await llm.chat([{"role": "user", "content": prompt}])

            orig_len = len(block.strip())
            new_len = len(cleaned.strip())

            if orig_len == 0:
                cleaned_blocks.append(cleaned)
                continue

            ratio = new_len / orig_len

            # Anti-Hallucination / Anti-Loss protections
            if ratio < config.CLEAN_MIN_RATIO:
                logger.warning(f"Cleaning removed too much text (ratio: {ratio:.2f}). Reverting block.")
                cleaned_blocks.append(block)
            elif ratio > config.CLEAN_MAX_RATIO:
                logger.warning(f"Cleaning generated too much text (ratio: {ratio:.2f}). Reverting block.")
                cleaned_blocks.append(block)
            else:
                cleaned_blocks.append(cleaned)
        except Exception as e:
            logger.error(f"Error during block cleaning: {e}")
            cleaned_blocks.append(block)

    return "".join(cleaned_blocks)

# --- Gating ---

async def gatekeeper(text: str, source: str, force_category: Optional[str] = None) -> Dict[str, Any]:
    """LLM evaluates if the text should be kept and categorizes it."""
    prompt_template = db.get_prompt("gatekeeper")
    categories_str = ", ".join(config.KNOWLEDGE_CATEGORIES)
    prompt = prompt_template.format(text=text[:10000], categories=categories_str) # Limit to avoid huge context just for gating

    fallback = {
        "keep": True,
        "summary": "Auto-accepted (fallback)",
        "tags": [],
        "category": force_category if force_category and force_category in config.KNOWLEDGE_CATEGORIES else "general"
    }

    try:
        result = await llm.chat([
            {"role": "system", "content": "You must output strictly valid JSON."},
            {"role": "user", "content": prompt}
        ], json_mode=True)

        if result and isinstance(result, dict) and "keep" in result:
            # Enforce valid category
            cat = result.get("category", "general")
            if force_category and force_category in config.KNOWLEDGE_CATEGORIES:
                cat = force_category
            elif cat not in config.KNOWLEDGE_CATEGORIES:
                cat = "general"

            result["category"] = cat
            return result
        else:
            logger.warning("Gatekeeper returned invalid JSON structure. Using fallback.")
            return fallback
    except Exception as e:
        logger.error(f"Gatekeeper failed: {e}. Using fallback.")
        return fallback

# --- Enrichment ---

async def enrich(text: str, source: str, force_category: Optional[str] = None) -> str:
    """
    Check if knowledge already exists.
    Returns: 'NEW' (added), 'DUPLICATE' (ignored), or 'REVIEW' (added to queue).
    """
    if not text.strip():
        return "DUPLICATE"

    # Evaluate the document (get category, tags)
    gate_result = await gatekeeper(text, source, force_category)
    if not gate_result.get("keep", False):
        return "DUPLICATE" # Not worth keeping

    category = gate_result.get("category", "general")
    tags = gate_result.get("tags", [])

    # Semantically search existing knowledge
    # Use a snippet for searching
    search_query = text[:2000]

    highest_score = 0.0

    if config.CONFLICT_CHECK_ALL_CATEGORIES:
        for cat in config.KNOWLEDGE_CATEGORIES:
            similar_docs = await qdrant_store.search(search_query, category=cat, top_k=1)
            if similar_docs and similar_docs[0]["score"] > highest_score:
                highest_score = similar_docs[0]["score"]
    else:
        similar_docs = await qdrant_store.search(search_query, category=category, top_k=1)
        if similar_docs:
            highest_score = similar_docs[0]["score"]

    if highest_score >= config.CONFLICT_THRESHOLD:
        # Potential conflict / update -> Send to review queue
        db.add_to_review(
            text=text,
            source=source,
            category=category,
            reason=f"High similarity ({highest_score:.2f}) to existing document."
        )
        return "REVIEW"

    # NEW -> Process, clean, and upsert

    # 1. Clean the full text in efficient blocks
    cleaned_text = await clean_text(text)

    # 2. Chunk the cleaned text for embedding
    chunks = chunk_text(cleaned_text)
    processed_chunks = []

    for c in chunks:
        processed_chunks.append({
            "text": c,
            "source": source,
            "category": category,
            "tags": tags
        })

    # 3. Upsert into Vector DB
    await qdrant_store.upsert(processed_chunks)
    return "NEW"

# --- Reindex Thread Trigger ---

_reindex_task: Optional[asyncio.Task] = None
_reindex_status = {"status": "idle", "processed": 0, "total": 0, "progress": 0.0, "error": None, "collections": {}}

async def _reindex_worker(clean: bool):
    """Background worker to realistically re-embed all documents from Qdrant across categories."""
    global _reindex_status
    _reindex_status = {
        "status": "running",
        "processed": 0,
        "total": 0,
        "progress": 0.0,
        "error": None,
        "collections": {}
    }

    try:
        logger.info("Re-indexing started.")

        # 1. Count totals and prep state
        total_points = 0
        all_points_by_cat = {}
        for cat in config.KNOWLEDGE_CATEGORIES:
            col_name = qdrant_store._get_collection_name(cat)
            count = await qdrant_store.count_points(col_name)
            if count > 0:
                _reindex_status["collections"][col_name] = {"total": count, "processed": 0}
                total_points += count
            else:
                 _reindex_status["collections"][col_name] = {"total": 0, "processed": 0}

        _reindex_status["total"] = total_points

        if total_points == 0:
            logger.info("No points to re-index.")
            _reindex_status["status"] = "completed"
            _reindex_status["progress"] = 100.0
            return

        # 2. Process each collection
        for cat in config.KNOWLEDGE_CATEGORIES:
            col_name = qdrant_store._get_collection_name(cat)

            if _reindex_status["collections"][col_name]["total"] == 0:
                continue

            # Scroll all payloads for this collection
            logger.info(f"Scrolling points for {col_name}...")
            payloads = await qdrant_store.scroll_all(col_name)

            # Recreate the collection (essential if embedding dim changes)
            await qdrant_store.recreate_collection(col_name)

            # Batch process points
            batch_size = 64
            for i in range(0, len(payloads), batch_size):
                batch = payloads[i:i+batch_size]

                processed_batch = []
                for p in batch:
                    text = p.get("text", "")
                    if clean:
                        text = await clean_text(text)

                    processed_batch.append({
                        "text": text,
                        "source": p.get("source", ""),
                        "category": p.get("category", cat),
                        "tags": p.get("tags", [])
                    })

                # Upsert the new embeddings
                if processed_batch:
                    try:
                        await qdrant_store.upsert(processed_batch)
                    except Exception as e:
                        logger.error(f"Failed to upsert batch during reindex of {col_name}: {e}")

                _reindex_status["processed"] += len(batch)
                _reindex_status["collections"][col_name]["processed"] += len(batch)
                _reindex_status["progress"] = round((_reindex_status["processed"] / _reindex_status["total"]) * 100.0, 2)

        logger.info("Re-indexing completed.")
        _reindex_status["progress"] = 100.0
        _reindex_status["status"] = "completed"

    except Exception as e:
        logger.error(f"Reindex failed: {e}")
        _reindex_status["status"] = "failed"
        _reindex_status["error"] = str(e)

def start_reindex(clean: bool = False) -> Dict[str, Any]:
    global _reindex_task
    if _reindex_task and not _reindex_task.done():
        return {"status": "already_running"}

    _reindex_task = asyncio.create_task(_reindex_worker(clean))
    return {"status": "started"}

def get_reindex_status() -> Dict[str, Any]:
    return _reindex_status
