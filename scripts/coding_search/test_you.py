"""Offline contracts for the two published You configurations."""
import os
import unittest
from unittest.mock import patch
from coding_search import search


class YouContracts(unittest.TestCase):
    def test_search_and_separate_contents_fetch(self):
        payload = {"results": {"web": [{"url": "https://example.com", "snippets": ["fallback"],
                    "contents": {"highlights": ["Relevant evidence"]}}]}}
        for arm in ("you_highlights", "you_highlights_core"):
            with self.subTest(arm=arm), patch.dict(os.environ, {"YDC_API_KEY": "test-key"}):
                backend = search.get_backend(arm)
                self.assertIn(arm, search.SEARCH_ONLY_BACKENDS)
                self.assertIn(arm, search.SEARCH_FETCH_BACKENDS)
                self.assertIn(arm, search.DUAL_SPLIT_BACKENDS)
                body = {"query": "Unchanged question?", "count": 8,
                        "extraction": {"extraction_mode": "highlights"}}
                if arm.endswith("_core"):
                    body["knowledge"] = "core"
                with patch.object(search, "vendor_call", return_value=(payload, {})) as call:
                    self.assertEqual(backend.search("Unchanged question?", max_results=8)[0]["snippet"], "Relevant evidence")
                    self.assertEqual(call.call_count, 1)
                    self.assertEqual(call.call_args.args[2], search.YOU_SEARCH_URL)
                    self.assertEqual(call.call_args.kwargs["json_body"], body)
                with patch.object(search, "vendor_call", return_value=([{"url": "https://example.com", "markdown": "Page"}], {})) as call:
                    self.assertEqual(backend.fetch("https://example.com")["content"], "Page")
                    self.assertEqual(call.call_count, 1)
                    self.assertEqual(call.call_args.args[2], search.YOU_CONTENTS_URL)
                    self.assertEqual(call.call_args.kwargs["json_body"], {
                        "urls": ["https://example.com"], "formats": ["markdown"], "crawl_timeout": 20})

    def test_retired_plain_search_not_in_roster(self):
        with self.assertRaises(KeyError):
            search.get_backend("you")


if __name__ == "__main__":
    unittest.main()
