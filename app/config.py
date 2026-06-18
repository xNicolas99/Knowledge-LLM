import os
import sys

def get_env_or_fail(key: str) -> str:
    val = os.getenv(key)
    if not val:
        print(f"ERROR: Missing required environment variable: {key}", file=sys.stderr)
        sys.exit(1)
    return val

def get_env(key: str, default: str) -> str:
    val = os.getenv(key)
    if val is None or val.strip() == "":
        return default
    return val

def get_env_bool(key: str, default: bool) -> bool:
    val = os.getenv(key)
    if val is None or val.strip() == "":
        return default
    return val.lower() in ("true", "1", "yes", "y", "t")

def get_env_int(key: str, default: int) -> int:
    val = os.getenv(key)
    if val is None or val.strip() == "":
        return default
    try:
        return int(val)
    except ValueError:
        print(f"WARNING: Environment variable {key} could not be parsed as int. Using default {default}.", file=sys.stderr)
        return default

def get_env_float(key: str, default: float) -> float:
    val = os.getenv(key)
    if val is None or val.strip() == "":
        return default
    try:
        return float(val)
    except ValueError:
        print(f"WARNING: Environment variable {key} could not be parsed as float. Using default {default}.", file=sys.stderr)
        return default

# LLM Configuration
LLM_BASE_URL = get_env_or_fail("LLM_BASE_URL")
LLM_API_KEY = get_env_or_fail("LLM_API_KEY")
LLM_MODEL = get_env_or_fail("LLM_MODEL")

# Embedding Configuration
EMBEDDING_BASE_URL = get_env_or_fail("EMBEDDING_BASE_URL")
EMBEDDING_API_KEY = get_env_or_fail("EMBEDDING_API_KEY")
EMBEDDING_MODEL = get_env_or_fail("EMBEDDING_MODEL")
EMBEDDING_DIM = get_env_int("EMBEDDING_DIM", 768)

# Knowledge Base Categories
_categories_str = get_env("KNOWLEDGE_CATEGORIES", "it,science,biology,business,general")
KNOWLEDGE_CATEGORIES = [c.strip() for c in _categories_str.split(",") if c.strip()]
if not KNOWLEDGE_CATEGORIES:
    KNOWLEDGE_CATEGORIES = ["general"]

# Chunking
CHUNK_SIZE = get_env_int("CHUNK_SIZE", 1000)
CHUNK_OVERLAP = get_env_int("CHUNK_OVERLAP", 100)

# Enrichment
CONFLICT_THRESHOLD = get_env_float("CONFLICT_THRESHOLD", 0.75)
CONFLICT_CHECK_ALL_CATEGORIES = get_env_bool("CONFLICT_CHECK_ALL_CATEGORIES", False)

# Text Cleaning
CLEAN_TEXT = get_env_bool("CLEAN_TEXT", True)
CLEAN_MIN_RATIO = get_env_float("CLEAN_MIN_RATIO", 0.5)
CLEAN_MAX_RATIO = get_env_float("CLEAN_MAX_RATIO", 1.15)
CLEAN_BLOCK_CHARS = get_env_int("CLEAN_BLOCK_CHARS", 6000)

# Web Search
WEBSEARCH_TOP_K = get_env_int("WEBSEARCH_TOP_K", 5)
WEBSEARCH_EVALUATE = get_env_bool("WEBSEARCH_EVALUATE", True)
WEBSEARCH_MAX_CHARS = get_env_int("WEBSEARCH_MAX_CHARS", 4000)
WEBSEARCH_EVAL_CHARS = get_env_int("WEBSEARCH_EVAL_CHARS", 6000)
CRAWL4AI_TOKEN = get_env("CRAWL4AI_TOKEN", "")
CRAWL_CONCURRENCY = get_env_int("CRAWL_CONCURRENCY", 3)

# Security / Access
API_KEY = get_env_or_fail("API_KEY")
REQUIRE_AUTH_FOR_READ = get_env_bool("REQUIRE_AUTH_FOR_READ", True)

# Internal URLs (within Compose network)
QDRANT_URL = get_env("QDRANT_URL", "http://qdrant:6333")
SEARXNG_URL = get_env("SEARXNG_URL", "http://searxng:8080")
CRAWL4AI_URL = get_env("CRAWL4AI_URL", "http://crawl4ai:11225")
DOCLING_URL = get_env("DOCLING_URL", "http://docling:5000")

# Database Path
DB_PATH = get_env("DB_PATH", "/app/data/raggate.db")
WATCH_DIR = get_env("WATCH_DIR", "/watch")
