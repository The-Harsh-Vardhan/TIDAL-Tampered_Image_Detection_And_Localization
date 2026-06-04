"""Thread-safe bounded LRU cache for TIDAL inference results."""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any


class InferenceCache:
    """Bounded LRU cache keyed by SHA-256 of raw image bytes.

    Thread-safe via a single lock.  Oldest entries are evicted when
    ``max_size`` is reached.
    """

    def __init__(self, max_size: int = 128) -> None:
        self._max_size = max(1, max_size)
        self._store: OrderedDict[str, Any] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._lock = threading.Lock()

    # ── Public API ─────────────────────────────────────────────────────────

    def get(self, key: str) -> Any | None:
        """Return cached value or ``None`` if not present."""
        with self._lock:
            if key not in self._store:
                self._misses += 1
                return None
            # Move to end (most recently used)
            self._store.move_to_end(key)
            self._hits += 1
            return self._store[key]

    def set(self, key: str, value: Any) -> None:
        """Store a value; evict the LRU entry when at capacity."""
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                self._store[key] = value
                return
            self._store[key] = value
            if len(self._store) > self._max_size:
                self._store.popitem(last=False)  # evict oldest

    def stats(self) -> dict:
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._store),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / total, 4) if total else 0.0,
            }

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._hits = 0
            self._misses = 0

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._store
