import logging
import uuid
from typing import Any, Dict, List, Optional
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, ScoredPoint
from app import config
from app.llm import embed

logger = logging.getLogger(__name__)

# Initialize Qdrant Client
client = AsyncQdrantClient(url=config.QDRANT_URL, timeout=30.0)

def _get_collection_name(category: str) -> str:
    """Returns the Qdrant collection name for a given category."""
    # Fallback to 'general' if category not in allowed list
    safe_cat = category if category in config.KNOWLEDGE_CATEGORIES else "general"
    return f"kb_{safe_cat}"

async def init_collections():
    """Ensures all category collections exist in Qdrant."""
    existing = await client.get_collections()
    existing_names = [c.name for c in existing.collections]

    for cat in config.KNOWLEDGE_CATEGORIES:
        col_name = _get_collection_name(cat)
        if col_name not in existing_names:
            logger.info(f"Creating Qdrant collection: {col_name} (dim: {config.EMBEDDING_DIM})")
            await client.create_collection(
                collection_name=col_name,
                vectors_config=VectorParams(
                    size=config.EMBEDDING_DIM,
                    distance=Distance.COSINE
                )
            )

async def search(query: str, category: Optional[str] = None, top_k: int = 5) -> List[Dict[str, Any]]:
    """Semantically search within a category (or 'general' if none specified)."""
    # Embed the query
    query_vector = (await embed([query]))[0]

    col_name = _get_collection_name(category or "general")

    try:
        search_result: List[ScoredPoint] = await client.search(
            collection_name=col_name,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True
        )

        results = []
        for point in search_result:
            results.append({
                "id": point.id,
                "score": point.score,
                "text": point.payload.get("text", ""),
                "source": point.payload.get("source", ""),
                "category": point.payload.get("category", ""),
                "tags": point.payload.get("tags", [])
            })
        return results
    except Exception as e:
        logger.error(f"Error searching Qdrant collection {col_name}: {e}")
        return []

async def upsert(chunks: List[Dict[str, Any]]):
    """
    Upsert documents/chunks into Qdrant.
    Expected chunk format: {"text": "...", "source": "...", "category": "...", "tags": [...]}
    """
    if not chunks:
        return

    texts = [c["text"] for c in chunks]
    embeddings = await embed(texts)

    points_by_collection: Dict[str, List[PointStruct]] = {}

    for i, chunk in enumerate(chunks):
        col_name = _get_collection_name(chunk.get("category", "general"))

        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=embeddings[i],
            payload={
                "text": chunk["text"],
                "source": chunk.get("source", ""),
                "category": chunk.get("category", "general"),
                "tags": chunk.get("tags", [])
            }
        )

        if col_name not in points_by_collection:
            points_by_collection[col_name] = []
        points_by_collection[col_name].append(point)

    # Upsert points by collection
    for col_name, points in points_by_collection.items():
        try:
            await client.upsert(
                collection_name=col_name,
                points=points
            )
            logger.info(f"Upserted {len(points)} points into {col_name}")
        except Exception as e:
            logger.error(f"Error upserting into {col_name}: {e}")

async def get_stats() -> Dict[str, Any]:
    """Retrieve points count for all knowledge base collections."""
    stats = {}
    for cat in config.KNOWLEDGE_CATEGORIES:
        col_name = _get_collection_name(cat)
        try:
            info = await client.get_collection(col_name)
            stats[col_name] = {"points_count": info.points_count}
        except Exception:
            stats[col_name] = {"points_count": 0, "error": "not found or error"}
    return stats
