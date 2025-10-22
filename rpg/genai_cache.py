"""Utilities for managing Gemini cached content for repeated prompts."""

# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from typing import Iterable, List, Optional

try:  # pragma: no cover - optional dependency at runtime
    from google import genai as google_genai
    from google.genai import types as google_genai_types
except (ModuleNotFoundError, ImportError):  # pragma: no cover
    google_genai = None  # type: ignore[assignment]
    google_genai_types = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)

_DEFAULT_TTL_SECONDS = 3600
_TTL_ENV_KEY = "GEMINI_CACHE_TTL_SECONDS"


def _resolve_api_key() -> Optional[str]:
    """Return the API key to use for the Gemini cache client."""

    value = os.environ.get("GEMINI_API_KEY")
    if value:
        return value
    return None


def _ttl_from_env() -> int:
    """Return the configured TTL in seconds, falling back to defaults."""

    value = os.environ.get(_TTL_ENV_KEY)
    if value is None:
        return _DEFAULT_TTL_SECONDS
    try:
        ttl = int(value.strip())
    except ValueError:
        logger.warning(
            "Invalid %s value %r; defaulting to %d seconds",
            _TTL_ENV_KEY,
            value,
            _DEFAULT_TTL_SECONDS,
        )
        return _DEFAULT_TTL_SECONDS
    if ttl < 1:
        logger.warning(
            "%s must be at least 1; using default %d",
            _TTL_ENV_KEY,
            _DEFAULT_TTL_SECONDS,
        )
        return _DEFAULT_TTL_SECONDS
    return ttl


@dataclass(frozen=True)
class CachedConfig:
    """Wrapper storing cached content metadata for callers."""

    config: object
    name: str


class GeminiCacheManager:
    """Create and reuse Gemini cached content for static prompt segments."""

    def __init__(
        self,
        *,
        client: object | None = None,
        api_key: str | None = None,
        ttl_seconds: int | None = None,
    ) -> None:
        if google_genai is None or google_genai_types is None:
            raise ModuleNotFoundError("google-genai package not available")
        if client is not None:
            self._client = client
        else:
            if not api_key:
                api_key = _resolve_api_key()
            if not api_key:
                raise ValueError("API key required to initialise Gemini cache manager")
            self._client = google_genai.Client(api_key=api_key)
        self._ttl_seconds = ttl_seconds if ttl_seconds is not None else _ttl_from_env()
        self._lock = threading.Lock()
        self._cache_names: dict[str, str] = {}

    @staticmethod
    def _format_ttl(seconds: int) -> str:
        return f"{max(1, seconds)}s"

    def _text_to_content(self, text: str):
        """Return a ``types.Content`` instance for ``text``."""

        return google_genai_types.Content(
            parts=[google_genai_types.Part.from_text(text=text)]
        )

    def _find_existing_cache(self, display_name: str):
        try:
            for cache in self._client.caches.list():
                if getattr(cache, "display_name", None) == display_name:
                    return cache
        except Exception as exc:  # pragma: no cover - network failure
            logger.warning("Failed to list Gemini caches: %s", exc)
        return None

    def get_cached_config(
        self,
        *,
        display_name: str,
        model: str,
        texts: Iterable[str],
        system_instruction: str | None = None,
    ) -> Optional[object]:
        """Return a ``GenerateContentConfig`` referencing cached ``texts``.

        If the cached content does not already exist, it is created. When it does
        exist the original TTL is left untouched so the cache naturally expires.
        """

        if google_genai_types is None:
            return None
        filtered: List[str] = [part.strip() for part in texts if str(part).strip()]
        if not filtered:
            return None
        contents = [self._text_to_content(text) for text in filtered]
        with self._lock:
            cache_name = self._cache_names.get(display_name)
            if not cache_name:
                existing = self._find_existing_cache(display_name)
                if existing is not None:
                    cache_name = existing.name
                else:
                    kwargs = dict(
                        display_name=display_name,
                        contents=contents,
                        ttl=self._format_ttl(self._ttl_seconds),
                    )
                    if system_instruction:
                        kwargs["system_instruction"] = system_instruction
                    cache = self._client.caches.create(
                        model=model,
                        config=google_genai_types.CreateCachedContentConfig(**kwargs),
                    )
                    cache_name = cache.name
                self._cache_names[display_name] = cache_name
        return google_genai_types.GenerateContentConfig(cached_content=cache_name)


_manager_lock = threading.Lock()
_cached_manager: GeminiCacheManager | None = None


def get_cache_manager() -> GeminiCacheManager | None:
    """Return a process-wide :class:`GeminiCacheManager` instance if available."""

    global _cached_manager
    if _cached_manager is not None:
        return _cached_manager
    if google_genai is None or google_genai_types is None:
        return None
    with _manager_lock:
        if _cached_manager is not None:
            return _cached_manager
        api_key = _resolve_api_key()
        if not api_key:
            logger.debug("Gemini cache disabled: no API key present")
            return None
        try:
            _cached_manager = GeminiCacheManager(api_key=api_key)
        except Exception as exc:  # pragma: no cover - runtime configuration issues
            logger.warning("Failed to initialise Gemini cache manager: %s", exc)
            return None
        return _cached_manager


__all__ = ["GeminiCacheManager", "get_cache_manager"]
