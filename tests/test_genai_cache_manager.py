# SPDX-License-Identifier: GPL-3.0-or-later

import os
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import MagicMock, call, patch

from rpg.genai_cache import GeminiCacheManager


class GeminiCacheManagerTest(TestCase):
    @patch("rpg.genai_cache.google_genai")
    @patch("rpg.genai_cache.google_genai_types")
    def test_manager_creates_and_reuses_cached_content(
        self, mock_types, mock_google
    ):
        mock_client = MagicMock()
        mock_google.Client.return_value = mock_client
        mock_client.caches.list.return_value = []
        mock_client.caches.create.return_value = SimpleNamespace(name="cached/1")

        mock_types.Part.from_text = MagicMock(side_effect=lambda text: {"text": text})
        mock_types.Content = MagicMock(
            side_effect=lambda **kwargs: {"content": kwargs}
        )
        mock_types.CreateCachedContentConfig = MagicMock(
            side_effect=lambda **kwargs: {"create": kwargs}
        )
        mock_types.GenerateContentConfig = MagicMock(
            side_effect=lambda **kwargs: {"generate": kwargs}
        )

        manager = GeminiCacheManager(api_key="token", ttl_seconds=120)
        config = manager.get_cached_config(
            display_name="demo",
            model="models/gemini-2.0",
            texts=["alpha", "beta"],
        )
        self.assertEqual(config, {"generate": {"cached_content": "cached/1"}})
        self.assertEqual(
            [call.kwargs for call in mock_types.Content.call_args_list],
            [
                {"role": "user", "parts": [{"text": "alpha"}]},
                {"role": "user", "parts": [{"text": "beta"}]},
            ],
        )
        mock_client.caches.create.assert_called_once()
        mock_client.caches.update.assert_not_called()

        # Reuse cached entry without creating a new one
        config_again = manager.get_cached_config(
            display_name="demo",
            model="models/gemini-2.0",
            texts=["alpha"],
        )
        self.assertEqual(config_again, {"generate": {"cached_content": "cached/1"}})
        self.assertEqual(mock_client.caches.create.call_count, 1)
        mock_client.caches.update.assert_not_called()

    @patch("rpg.genai_cache.google_genai")
    @patch("rpg.genai_cache.google_genai_types")
    def test_manager_marks_failed_cache_and_does_not_retry(
        self, mock_types, mock_google
    ):
        mock_client = MagicMock()
        mock_google.Client.return_value = mock_client
        mock_client.caches.list.return_value = []
        mock_client.caches.create.side_effect = RuntimeError("too small")

        mock_types.Part.from_text = MagicMock(side_effect=lambda text: {"text": text})
        mock_types.Content = MagicMock(
            side_effect=lambda **kwargs: {"content": kwargs}
        )
        mock_types.CreateCachedContentConfig = MagicMock(
            side_effect=lambda **kwargs: {"create": kwargs}
        )
        mock_types.GenerateContentConfig = MagicMock(
            side_effect=lambda **kwargs: {"generate": kwargs}
        )

        manager = GeminiCacheManager(api_key="token", ttl_seconds=120)

        self.assertIsNone(
            manager.get_cached_config(
                display_name="demo",
                model="models/gemini-2.0",
                texts=["alpha"],
            )
        )
        # Subsequent attempts should not invoke the API again once marked failed
        self.assertIsNone(
            manager.get_cached_config(
                display_name="demo",
                model="models/gemini-2.0",
                texts=["alpha"],
            )
        )
        mock_client.caches.create.assert_called_once()
        mock_client.caches.list.assert_called_once()

    @patch("rpg.genai_cache.google_genai")
    @patch("rpg.genai_cache.google_genai_types")
    def test_manager_does_not_retry_after_multiple_requests(
        self, mock_types, mock_google
    ):
        mock_client = MagicMock()
        mock_google.Client.return_value = mock_client
        mock_client.caches.list.return_value = []
        mock_client.caches.create.side_effect = RuntimeError("too small")

        mock_types.Part.from_text = MagicMock(side_effect=lambda text: {"text": text})
        mock_types.Content = MagicMock(
            side_effect=lambda **kwargs: {"content": kwargs}
        )
        mock_types.CreateCachedContentConfig = MagicMock(
            side_effect=lambda **kwargs: {"create": kwargs}
        )
        mock_types.GenerateContentConfig = MagicMock(
            side_effect=lambda **kwargs: {"generate": kwargs}
        )

        manager = GeminiCacheManager(api_key="token", ttl_seconds=120)

        for _ in range(3):
            self.assertIsNone(
                manager.get_cached_config(
                    display_name="demo",
                    model="models/gemini-2.0",
                    texts=["alpha"],
                )
            )

        mock_client.caches.create.assert_called_once()
        mock_client.caches.list.assert_called_once()

    @patch("rpg.genai_cache.google_genai")
    @patch("rpg.genai_cache.google_genai_types")
    def test_get_cache_manager_without_api_key_returns_none(
        self, mock_types, mock_google
    ):
        mock_google.Client.return_value = MagicMock()
        mock_types.Part.from_text = MagicMock(side_effect=lambda text: {"text": text})
        mock_types.Content = MagicMock(
            side_effect=lambda **kwargs: {"content": kwargs}
        )

        with patch.dict(os.environ, {}, clear=True):
            import rpg.genai_cache as genai_cache

            genai_cache._cached_manager = None
            self.assertIsNone(genai_cache.get_cache_manager())

