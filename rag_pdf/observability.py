import os
import time
import json
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone

logger = logging.getLogger("rag_observability")
logger.setLevel(logging.INFO)

if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [RAG] %(message)s"))
    logger.addHandler(ch)

def setup_langsmith():
    """
    Ensures LangSmith tracing is active if environment variables are set.
    """
    if os.getenv("LANGCHAIN_API_KEY"):
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        if not os.getenv("LANGCHAIN_PROJECT"):
            os.environ["LANGCHAIN_PROJECT"] = "semantic-document-rag"
        logger.info(f"LangSmith Tracing active for project: {os.environ.get('LANGCHAIN_PROJECT')}")
        return True
    return False

class QueryTrace:
    """Records performance, latency, tokens, and retrieval metrics for a query."""
    def __init__(self, query: str, model_name: str):
        self.query = query
        self.model_name = model_name
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.retrieval_latency: float = 0.0
        self.generation_latency: float = 0.0
        self.total_latency: float = 0.0
        self.retrieved_chunks_count: int = 0
        self.source_pages: List[Any] = []
        self.cached: bool = False
        self.guardrail_blocked: bool = False
        self.guardrail_reason: Optional[str] = None
        self.error: Optional[str] = None

    def record_retrieval(self, latency: float, count: int, pages: List[Any]):
        self.retrieval_latency = latency
        self.retrieved_chunks_count = count
        self.source_pages = pages

    def complete(self, cached: bool = False, error: Optional[str] = None):
        self.end_time = time.time()
        self.total_latency = self.end_time - self.start_time
        self.cached = cached
        self.error = error

        log_payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": self.query,
            "model": self.model_name,
            "cached": self.cached,
            "chunks_retrieved": self.retrieved_chunks_count,
            "source_pages": self.source_pages,
            "total_latency_sec": round(self.total_latency, 3),
            "retrieval_latency_sec": round(self.retrieval_latency, 3),
            "error": self.error,
        }
        logger.info(json.dumps(log_payload))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model_name,
            "total_latency": round(self.total_latency, 2),
            "retrieval_latency": round(self.retrieval_latency, 2),
            "chunks_count": self.retrieved_chunks_count,
            "pages": self.source_pages,
            "cached": self.cached,
            "error": self.error,
        }

