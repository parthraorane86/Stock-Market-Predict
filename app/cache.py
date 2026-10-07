"""A tiny thread-safe in-memory cache with per-entry expiry."""
from __future__ import annotations

import threading
import time
from typing import Any, Callable, Hashable


class TTLCache:
    """Cache values for `ttl` seconds. Failures (None) are never cached."""

    def __init__(self, ttl: float) -> None:
        self.ttl = ttl
        self._store: dict[Hashable, tuple[float, Any]] = {}
        self._key_locks: dict[Hashable, threading.Lock] = {}
        self._lock = threading.Lock()

    def _fresh(self, key: Hashable) -> tuple[bool, Any]:
        entry = self._store.get(key)
        if entry and entry[0] > time.monotonic():
            return True, entry[1]
        return False, None

    def get_or_set(self, key: Hashable, factory: Callable[[], Any]) -> Any:
        with self._lock:
            hit, value = self._fresh(key)
            if hit:
                return value
            key_lock = self._key_locks.setdefault(key, threading.Lock())

        # One thread computes a given key; others wait and reuse the result.
        with key_lock:
            with self._lock:
                hit, value = self._fresh(key)
            if hit:
                return value
            value = factory()
            if value is not None:
                with self._lock:
                    self._store[key] = (time.monotonic() + self.ttl, value)
            return value

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
