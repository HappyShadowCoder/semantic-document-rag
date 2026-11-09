import numpy as np
from typing import Optional, Tuple, Dict, Any
from .embedder import get_embedder

class SemanticCache:
    """
    In-memory semantic cache using query embeddings and cosine similarity.
    Safely degrades if no embedding model is configured yet.
    """
    def __init__(self, similarity_threshold: float = 0.92):
        self.similarity_threshold = similarity_threshold
        self.cache: Dict[str, Dict[str, Any]] = {}

    def _get_embedder(self, openai_key: str = None, gemini_key: str = None):
        try:
            return get_embedder(openai_key=openai_key, gemini_key=gemini_key)
        except Exception:
            return None

    def _cosine_similarity(self, vec1: list, vec2: list) -> float:
        v1 = np.array(vec1, dtype=float)
        v2 = np.array(vec2, dtype=float)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))

    def get(self, query: str, doc_name: str = "", openai_key: str = None, gemini_key: str = None) -> Optional[Tuple[str, float]]:
        if not self.cache:
            return None

        embedder = self._get_embedder(openai_key=openai_key, gemini_key=gemini_key)
        if embedder is None:
            return None

        try:
            query_embedding = embedder.embed_query(query)
        except Exception:
            return None

        best_score = -1.0
        best_match = None

        for item in self.cache.values():
            if doc_name and item.get("doc_name") != doc_name:
                continue

            score = self._cosine_similarity(query_embedding, item["embedding"])
            if score > best_score:
                best_score = score
                best_match = item

        if best_match and best_score >= self.similarity_threshold:
            return best_match["response"], best_score

        return None

    def put(self, query: str, response: str, doc_name: str = "", openai_key: str = None, gemini_key: str = None):
        embedder = self._get_embedder(openai_key=openai_key, gemini_key=gemini_key)
        if embedder is None:
            return

        try:
            query_embedding = embedder.embed_query(query)
            key = f"{doc_name}:{query}"
            self.cache[key] = {
                "query": query,
                "doc_name": doc_name,
                "embedding": query_embedding,
                "response": response
            }
        except Exception:
            pass

    def clear(self):
        self.cache.clear()
