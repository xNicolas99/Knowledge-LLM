import json
import logging
from typing import Any, Dict, List, Optional
import httpx
import re

from app import config

logger = logging.getLogger(__name__)

def _extract_first_json(text: str) -> Optional[Dict[str, Any]]:
    """Extracts the first JSON object from a string, handling markdown blocks."""
    # Try to find JSON block enclosed in markdown
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        # Fallback to finding the first { and last }
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and start < end:
            json_str = text[start:end+1]
        else:
            return None

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        logger.error(f"Failed to decode extracted JSON: {json_str[:100]}...")
        return None

async def chat(messages: List[Dict[str, str]], json_mode: bool = False, temperature: float = 0.7) -> Any:
    """Send a chat completion request to the OpenAI-compatible LLM endpoint."""
    url = f"{config.LLM_BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {config.LLM_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": config.LLM_MODEL,
        "messages": messages,
        "temperature": temperature
    }

    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            content = result["choices"][0]["message"]["content"]

            if json_mode:
                parsed_json = _extract_first_json(content)
                if parsed_json is not None:
                    return parsed_json
                logger.warning("LLM was asked for JSON but failed to produce a valid structure. Raw content returned.")

            return content

        except httpx.HTTPError as e:
            logger.error(f"HTTP error communicating with LLM: {e}")
            raise
        except (KeyError, IndexError) as e:
            logger.error(f"Unexpected response format from LLM: {e}")
            raise

async def embed(texts: List[str]) -> List[List[float]]:
    """Get embeddings for a list of texts from the OpenAI-compatible embedding endpoint."""
    if not texts:
        return []

    url = f"{config.EMBEDDING_BASE_URL.rstrip('/')}/embeddings"
    headers = {
        "Authorization": f"Bearer {config.EMBEDDING_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": config.EMBEDDING_MODEL,
        "input": texts
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()

            # OpenAI API returns data ordered by index
            embeddings = []
            for item in sorted(result["data"], key=lambda x: x["index"]):
                embeddings.append(item["embedding"])

            return embeddings

        except httpx.HTTPError as e:
            logger.error(f"HTTP error communicating with Embedding endpoint: {e}")
            raise
        except (KeyError, IndexError) as e:
            logger.error(f"Unexpected response format from Embedding endpoint: {e}")
            raise
