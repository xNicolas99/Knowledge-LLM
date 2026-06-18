import time
import asyncio
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app import websearch

# --- Schemas based on Tavily API ---

class TavilySearchRequest(BaseModel):
    query: str
    search_depth: str = "basic" # "basic" or "advanced"
    max_results: int = 5
    include_answer: bool = False
    include_raw_content: bool = False
    include_domains: Optional[List[str]] = None
    exclude_domains: Optional[List[str]] = None
    api_key: Optional[str] = None # Support passing key in body

class TavilyExtractRequest(BaseModel):
    urls: List[str]
    api_key: Optional[str] = None

# --- Adapters ---

async def tavily_search(req: TavilySearchRequest) -> Dict[str, Any]:
    """Adapter for Tavily /search endpoint."""
    start_time = time.time()

    # advanced depth triggers LLM evaluation
    evaluate = req.search_depth == "advanced"

    # Run the websearch pipeline
    results = await websearch.websearch(
        query=req.query,
        top_k=req.max_results,
        evaluate=evaluate
    )

    formatted_results = []
    for r in results:
        item = {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": r.get("content", ""),
            "score": r.get("score", 1.0),
            "published_date": r.get("published_date", "")
        }
        if req.include_raw_content:
            item["raw_content"] = r.get("raw_content", "")
        formatted_results.append(item)

    answer = None
    if req.include_answer and formatted_results:
        # In a full implementation, we'd synthesize an answer from the content using the LLM.
        # For simplicity/speed in MVP, we just take key points if available, or a generic response.
        answer_parts = []
        for r in results:
            if "key_points" in r and r["key_points"]:
                answer_parts.extend(r["key_points"])
        if answer_parts:
            answer = " ".join(answer_parts[:3])
        else:
            answer = "Answer synthesized from search results is not available without LLM synthesis."

    response_time = round(time.time() - start_time, 2)

    return {
        "query": req.query,
        "results": formatted_results,
        "answer": answer,
        "response_time": response_time
    }

async def tavily_extract(req: TavilyExtractRequest) -> Dict[str, Any]:
    """Adapter for Tavily /extract endpoint."""
    start_time = time.time()

    results = []
    failed_results = []

    crawl_tasks = [websearch.crawl_url(url) for url in req.urls]
    crawled_texts = await asyncio.gather(*crawl_tasks)

    for url, text in zip(req.urls, crawled_texts):
        if text:
            results.append({"url": url, "raw_content": text})
        else:
            failed_results.append({"url": url, "error": "Extraction failed"})

    response_time = round(time.time() - start_time, 2)

    return {
        "results": results,
        "failed_results": failed_results,
        "response_time": response_time
    }
