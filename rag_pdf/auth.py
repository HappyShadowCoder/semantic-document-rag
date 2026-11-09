import os
import time
from typing import Dict, Tuple

class AccessController:
    """
    Handles authentication, session-level isolation, and request rate limiting.
    """
    def __init__(self, requests_per_minute: int = 30):
        self.requests_per_minute = requests_per_minute
        self.request_history: Dict[str, list] = {}

    def is_auth_enabled(self) -> bool:
        return bool(os.getenv("APP_ACCESS_KEY"))

    def verify_key(self, provided_key: str) -> bool:
        expected_key = os.getenv("APP_ACCESS_KEY")
        if not expected_key:
            return True
        return provided_key.strip() == expected_key.strip()

    def check_rate_limit(self, session_id: str) -> Tuple[bool, str]:
        """
        Token-bucket / sliding window rate limiter per session_id.
        """
        now = time.time()
        window_start = now - 60.0

        if session_id not in self.request_history:
            self.request_history[session_id] = []

        # Filter out timestamps older than 60 seconds
        self.request_history[session_id] = [
            t for t in self.request_history[session_id] if t > window_start
        ]

        if len(self.request_history[session_id]) >= self.requests_per_minute:
            return False, f"Rate limit exceeded ({self.requests_per_minute} requests/min). Please wait a moment."

        self.request_history[session_id].append(now)
        return True, ""

    @staticmethod
    def get_user_collection_name(session_id: str, base_collection: str = "pdf_chunks") -> str:
        """Returns isolated collection name per user session."""
        import hashlib
        safe_hash = hashlib.md5(session_id.encode()).hexdigest()[:8]
        return f"{base_collection}_{safe_hash}"

