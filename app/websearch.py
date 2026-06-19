import logging
import asyncio
import httpx
from typing import Any, Dict, List, Optional
from app import config
from app import llm

logger = logging.getLogger(__name__)

async def search_searxng(query: str, top_k: int) -> List[Dict[str, str]]:
    """Query SearXNG for top URLs."""
    url = f"{config.SEARXNG_URL.rstrip('/')}/search"
    params = {
        "q": query,
        "format": "json"
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            results = []
            for item in data.get("results", [])[:top_k]:
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "published_date": item.get("publishedDate", "")
                })
            return results
    except Exception as e:
        logger.error(f"SearXNG search failed for query '{query}': {e}")
        return []

async def crawl_url(url: str) -> str:
    """Extract Markdown from URL using Crawl4AI."""
    c_url = f"{config.CRAWL4AI_URL.rstrip('/')}/md"
    headers = {"Content-Type": "application/json"}
    if config.CRAWL4AI_TOKEN:
        headers["Authorization"] = f"Bearer {config.CRAWL4AI_TOKEN}"

    payload = {"url": url}

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(c_url, headers=headers, json=payload)
            response.raise_for_status()
            # Handle potential variation in Crawl4AI 0.8.9 response
            try:
                data = response.json()
                md = data.get("markdown", data.get("content", ""))
            except Exception:
                md = response.text

            return md[:config.WEBSEARCH_EVAL_CHARS] # Cap length for evaluation
    except Exception as e:
        logger.error(f"Crawl4AI failed for url '{url}': {e}")
        return ""

async def evaluate_content(query: str, text: str) -> Dict[str, Any]:
    """Ask LLM to evaluate relevance and extract key points."""
    if not text.strip():
        return {"relevant": False, "key_points": []}

    prompt = (
        f"Du bist ein Recherche-Assistent. Bewerte den folgenden Text auf Relevanz bezüglich der Suchanfrage: '{query}'.\n"
        "Antworte AUSSCHLIESSLICH im JSON-Format:\n"
        "{\"relevant\": true/false, \"key_points\": [\"punkt 1\", \"punkt 2\"]}\n\n"
        f"TEXT (Auszug):\n{text}"
    )

    try:
        result = await llm.chat([
            {"role": "user", "content": prompt}
        ], json_mode=True)
        return {
            "relevant": result.get("relevant", False),
            "key_points": result.get("key_points", [])
        }
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return {"relevant": True, "key_points": ["Evaluation failed, keeping content."]}

async def websearch(query: str, top_k: int = config.WEBSEARCH_TOP_K, evaluate: bool = config.WEBSEARCH_EVALUATE) -> List[Dict[str, Any]]:
    """Full websearch pipeline: SearXNG -> Crawl4AI -> Evaluate (Optional)."""
    # 1. Search
    search_results = await search_searxng(query, top_k)
    if not search_results:
        return []

    # 2. Crawl in parallel
    crawl_tasks = [crawl_url(r["url"]) for r in search_results]
    crawled_texts = await asyncio.gather(*crawl_tasks)

    final_results = []

    # 3. Process & Evaluate
    for i, res in enumerate(search_results):
        raw_text = crawled_texts[i]
        if not raw_text:
            continue

        content = raw_text[:config.WEBSEARCH_MAX_CHARS]
        item = {
            "title": res["title"],
            "url": res["url"],
            "published_date": res["published_date"],
            "content": content,
            "raw_content": raw_text, # Keep slightly longer raw version if needed
            "score": 1.0 # Default score
        }

        if evaluate:
            eval_res = await evaluate_content(query, raw_text)
            if not eval_res.get("relevant", False):
                continue # Skip irrelevant
            item["key_points"] = eval_res.get("key_points", [])
            item["score"] = 0.9 # High score if evaluated relevant

        final_results.append(item)

    return final_results
