import os
import unittest
from unittest.mock import patch
import nimble
from coding_search import search as coding
from coding_search.eval import parse_arms

class NimbleTests(unittest.TestCase):
    def test_search_and_fetch_contracts(self):
        for depth in ("lite", "standard"):
            key = f"nimble_{depth}"
            payload = {"results": [{"url": "https://a.test", "description": "x" * 1500, "content": "FULL PAGE"}, {"url": "https://a.test"}, {}]}
            with patch.dict(os.environ, {"NIMBLE_API_KEY": "test"}), patch.object(coding, "vendor_call", return_value=(payload, {})) as call:
                backend = coding.get_backend(key)
                hits = backend.search("query", max_results=1)
                self.assertEqual(call.call_args.kwargs["json_body"], {"query": "query", "search_depth": depth, "focus": "general", "full_content": False, "max_results": 1})
                self.assertEqual(hits, [{"url": "https://a.test", "title": "", "snippet": "x" * 1200}])
                self.assertEqual(parse_arms(key, split="search-fetch"), (key,))
                self.assertIn(key, coding.SEARCH_ONLY_BACKENDS)
                call.return_value = ({"status": "success", "data": {"markdown": "x" * 20000}}, {})
                page = backend.fetch("https://a.test")
                self.assertEqual(call.call_args.args[2], nimble.EXTRACT_URL)
                self.assertEqual(call.call_args.kwargs["json_body"], {"url": "https://a.test", "formats": ["markdown"]})
                self.assertEqual(len(page["content"]), coding.DEFAULT_MAX_FETCH_CHARS)
                self.assertTrue(page["_meta"]["truncated"])
    def test_failed_extract_is_not_evidence(self):
        for payload in ({}, {"status": "failed", "data": {"markdown": "error"}}, {"status_code": 403, "data": {"markdown": "blocked"}}):
            with self.assertRaises(RuntimeError): nimble.extract_page(payload, "https://a.test", 100)
