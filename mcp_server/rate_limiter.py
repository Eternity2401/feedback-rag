"""
rate_limiter.py - In-memory rate limiting, input bounds, and caching for zero-cost API protection.
"""

import os
import sys
import time
import logging
from collections import defaultdict, deque
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("mcp_server.rate_limiter")

# Configuration via environment variables with safe defaults
MAX_INPUT_LENGTH = int(os.getenv("MCP_MAX_INPUT_LENGTH", "4000"))
MAX_TOP_K = int(os.getenv("MCP_MAX_TOP_K", "10"))
PER_IP_RATE_LIMIT = int(os.getenv("MCP_PER_IP_REQUEST_LIMIT", "15"))  # requests per minute
RATE_LIMIT_WINDOW = 60.0  # seconds
CACHE_TTL = 300.0  # 5 minutes in-memory cache TTL
CACHE_MAX_ENTRIES = 256


class InMemoryRateLimiter:
    """Sliding-window in-memory rate limiter per IP address."""

    def __init__(self, limit: int = PER_IP_RATE_LIMIT, window_seconds: float = RATE_LIMIT_WINDOW):
        self.limit = limit
        self.window = window_seconds
        self.requests: Dict[str, deque] = defaultdict(deque)

    def is_allowed(self, client_ip: str = "default") -> Tuple[bool, Optional[str]]:
        now = time.time()
        client_history = self.requests[client_ip]

        # Purge timestamps older than sliding window
        while client_history and client_history[0] <= now - self.window:
            client_history.popleft()

        if len(client_history) >= self.limit:
            return False, f"Rate limit reached ({self.limit} req/min). Please try again shortly."

        client_history.append(now)
        return True, None


class SimpleLRUCache:
    """Lightweight in-memory LRU cache with TTL expiration."""

    def __init__(self, max_entries: int = CACHE_MAX_ENTRIES, ttl_seconds: float = CACHE_TTL):
        self.max_entries = max_entries
        self.ttl = ttl_seconds
        self.cache: Dict[str, Tuple[float, Any]] = {}
        self.order: deque = deque()

    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            timestamp, value = self.cache[key]
            if time.time() - timestamp <= self.ttl:
                return value
            # Expired entry
            del self.cache[key]
        return None

    def set(self, key: str, value: Any) -> None:
        now = time.time()
        if key not in self.cache and len(self.cache) >= self.max_entries:
            # Evict oldest
            while self.order:
                old_key = self.order.popleft()
                if old_key in self.cache:
                    del self.cache[old_key]
                    break
        self.cache[key] = (now, value)
        self.order.append(key)


# Global instances for server lifecycle
rate_limiter = InMemoryRateLimiter()
response_cache = SimpleLRUCache()


def validate_text_input(text: str, field_name: str = "input") -> Optional[str]:
    """Validates that input text is non-empty and does not exceed maximum character length."""
    if not text or not text.strip():
        return f"The '{field_name}' parameter cannot be empty."
    if len(text) > MAX_INPUT_LENGTH:
        return f"The '{field_name}' parameter exceeds maximum allowed length of {MAX_INPUT_LENGTH} characters."
    return None


def sanitize_k(k: int, default: int = 3, max_k: int = MAX_TOP_K) -> int:
    """Clamps requested k parameter to safe bounds [1, max_k]."""
    try:
        val = int(k)
        return max(1, min(val, max_k))
    except (ValueError, TypeError):
        return default
