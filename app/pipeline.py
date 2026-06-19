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

async def clean_chunk(chunk: str) -> str:
    """Cleans a text chunk using the LLM with length checks to prevent hallucinations."""
    if not config.CLEAN_TEXT:
        return chunk

    prompt_template = db.get_prompt("clean_text")
    if not prompt_template:
        return chunk

    prompt = prompt_template.format(text=chunk)

    try:
        cleaned = await llm.chat([{"role": "user", "content": prompt}])

        orig_len = len(chunk.strip())
        new_len = len(cleaned.strip())

        if orig_len == 0:
            return cleaned

        ratio = new_len / orig_len

        # Anti-Hallucination / Anti-Loss protections
        if ratio < config.CLEAN_MIN_RATIO:
            logger.warning(f"Cleaning removed too much text (ratio: {ratio:.2f}). Reverting to original.")
            return chunk
        if ratio > config.CLEAN_MAX_RATIO:
            logger.warning(f"Cleaning generated too much text (ratio: {ratio:.2f}). Reverting to original.")
            return chunk

        return cleaned
    except Exception as e:
        logger.error(f"Error during chunk cleaning: {e}")
        return chunk

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
    similar_docs = await qdrant_store.search(search_query, category=category, top_k=1)

    if similar_docs and similar_docs[0]["score"] >= config.CONFLICT_THRESHOLD:
        # Potential conflict / update -> Send to review queue
        db.add_to_review(
            text=text,
            source=source,
            category=category,
            reason=f"High similarity ({similar_docs[0]['score']:.2f}) to existing document."
        )
        return "REVIEW"

    # NEW -> Process, clean, and upsert
    chunks = chunk_text(text)
    processed_chunks = []

    for i, c in enumerate(chunks):
        cleaned = await clean_chunk(c)
        processed_chunks.append({
            "text": cleaned,
            "source": source,
            "category": category,
            "tags": tags,
            "chunk_index": i
        })

    await qdrant_store.upsert(processed_chunks)
    return "NEW"

# --- Update Document ---

async def update_source_document(source: str, change_description: str) -> dict:
    """Updates an existing document in the knowledge base by applying a specific change."""
    # 1. Fetch chunks
    chunks = await qdrant_store.scroll_by_source(source)
    if not chunks:
        return {"status": "not_found", "message": f"Source '{source}' not found in any category."}

    # 2. Sort by chunk_index to reconstruct original text
    chunks.sort(key=lambda x: x.get("chunk_index", 0))

    # 3. Reconstruct full text, accounting for overlap
    # We append the chunk entirely if it's the first one. For subsequent ones,
    # we assume chunking was done with config.CHUNK_OVERLAP and skip the overlap part.
    original_text = chunks[0]["text"]
    for c in chunks[1:]:
        if len(c["text"]) > config.CHUNK_OVERLAP:
            original_text += c["text"][config.CHUNK_OVERLAP:]
        else:
            original_text += c["text"]

    # 4. Apply change using the LLM and strict update prompt
    prompt_template = db.get_prompt("update_document")
    if not prompt_template:
        return {"status": "error", "message": "update_document prompt not found in database."}

    # Replace manually because text might contain curly braces
    prompt_filled = prompt_template.replace("{change}", change_description).replace("{document}", original_text)

    try:
        new_text = await llm.chat([{"role": "user", "content": prompt_filled}])
    except Exception as e:
        logger.error(f"Error calling LLM for document update: {e}")
        return {"status": "error", "message": f"Model error: {e}"}

    # 5. Safety Net
    if not new_text or not new_text.strip():
        return {"status": "error", "message": "Sicherheitsabbruch: LLM lieferte leeren Text."}

    if len(new_text) < len(original_text) * 0.70:
        return {"status": "error", "message": "Sicherheitsabbruch: zu viel Text verloren (<70% Originallänge)."}

    # 6. Extract metadata from the first chunk
    category = chunks[0]["category"]
    tags = chunks[0]["tags"]

    # 7. Replace
    await qdrant_store.delete_by_source(source, category)

    new_chunks = chunk_text(new_text)
    processed_new_chunks = []

    for i, c in enumerate(new_chunks):
        processed_new_chunks.append({
            "text": c,
            "source": source,
            "category": category,
            "tags": tags,
            "chunk_index": i
        })

    await qdrant_store.upsert(processed_new_chunks)

    return {
        "status": "updated",
        "source": source,
        "old_chunks": len(chunks),
        "new_chunks": len(new_chunks)
    }

# --- Reindex Thread Trigger ---

_reindex_task: Optional[asyncio.Task] = None
_reindex_status = {"status": "idle", "progress": 0.0, "error": None}

async def _reindex_worker(clean: bool):
    """Background worker to re-embed all documents (simulated logic for now, actual extraction needs original docs or chunk reconstruction)"""
    global _reindex_status
    _reindex_status = {"status": "running", "progress": 0.0, "error": None}

    try:
        # In a real Qdrant scroll logic:
        # 1. Fetch all points
        # 2. Re-embed
        # 3. Upsert
        # Since this requires full text reconstruction from chunks which is complex,
        # a basic implementation just logs the intent for this specific MVP scope.
        logger.info("Re-indexing started (Mock process for MVP).")
        await asyncio.sleep(5) # Simulate work
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
