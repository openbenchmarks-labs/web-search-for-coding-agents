"""Offline contracts for the Keenable arm (no network)."""
import os
import unittest
from unittest.mock import MagicMock, patch

import requests

from coding_search import search
from coding_search.eval import parse_arms

ARM = "keenable"


def _http_error(status: int) -> requests.HTTPError:
    response = MagicMock(status_code=status)
    error = requests.HTTPError(f"{status} for {search.KEENABLE_FETCH_URL}", response=response)
    setattr(error, "vendor_meta", {"http_status": status, "request": {}, "response": {}})
    return error


def _fake_vendor_call(*outcomes):
    """vendor_call stand-in that mimics last_meta handling on failure."""
    queue = list(outcomes)

    def call(backend, *args, **kwargs):
        outcome = queue.pop(0)
        if isinstance(outcome, Exception):
            backend.last_meta = dict(getattr(outcome, "vendor_meta", None) or {})
            raise outcome
        return outcome, {"http_status": 200}

    return MagicMock(side_effect=call)


class KeenableContracts(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {"KEENABLE_API_KEY": "test-key"})
        env.start()
        self.addCleanup(env.stop)

    def test_roster_membership(self):
        self.assertIn(ARM, search.SEARCH_ONLY_BACKENDS)
        self.assertIn(ARM, search.SEARCH_FETCH_BACKENDS)
        self.assertIn(ARM, search.DUAL_SPLIT_BACKENDS)
        self.assertEqual(search.get_backend(ARM).name, ARM)
        self.assertEqual(parse_arms(ARM, split="search-only"), (ARM,))
        self.assertEqual(parse_arms(ARM, split="search-fetch"), (ARM,))
        keenable_ids = [b for b in search.BACKENDS if "keenable" in b]
        self.assertEqual(keenable_ids, [ARM])

    def test_search_request_contract(self):
        payload = {"results": [{"url": "https://a.test", "title": "A", "snippet": "S"}]}
        with patch.object(search, "vendor_call", return_value=(payload, {})) as call:
            hits = search.get_backend(ARM).search("Unchanged question?", max_results=8)
        self.assertEqual(hits, [{"url": "https://a.test", "title": "A", "snippet": "S"}])
        self.assertEqual(call.call_count, 1)
        self.assertEqual(call.call_args.args[1:3], ("POST", search.KEENABLE_SEARCH_URL))
        self.assertEqual(call.call_args.kwargs["headers"]["X-API-Key"], "test-key")
        self.assertEqual(call.call_args.kwargs["json_body"],
                         {"query": "Unchanged question?", "mode": "pro", "max_results": 8})

    def test_missing_key_raises(self):
        with patch.dict(os.environ, {"KEENABLE_API_KEY": ""}):
            with self.assertRaises(RuntimeError):
                search.get_backend(ARM).search("q")

    def test_parser(self):
        payload = {"results": [
            {"url": "https://a.test", "title": "A", "description": "desc", "snippet": "x" * 2000},
            {"url": "https://b.test", "description": "only description", "snippet": ""},
            {"title": "no url"},
            {"url": "https://a.test", "snippet": "dupe"},
            {"url": "https://c.test", "published_at": 1761955200},
            None, "not-a-dict", ["list"], {"url": None}, {"url": "   "},
        ]}
        hits = search.parse_keenable_hits(payload, max_results=10)
        self.assertEqual([h["url"] for h in hits], ["https://a.test", "https://b.test", "https://c.test"])
        self.assertEqual(hits[0]["snippet"], "x" * 1200)
        self.assertEqual(hits[1]["snippet"], "only description")
        self.assertEqual(hits[2]["snippet"], "")
        self.assertEqual(len(search.parse_keenable_hits(payload, max_results=1)), 1)
        for empty in (None, {}, {"results": None}, []):
            self.assertEqual(search.parse_keenable_hits(empty), [])

    def test_cost(self):
        self.assertAlmostEqual(search.keenable_cost_usd({"usage": {"credits": 2}}), 0.008)
        self.assertAlmostEqual(search.keenable_cost_usd({"usage": {"credits": 0}}), 0.0)
        self.assertAlmostEqual(search.keenable_cost_usd({"results": []}), 0.004)
        self.assertAlmostEqual(search.keenable_cost_usd({"usage": {"credits": "1"}}), 0.004)
        self.assertAlmostEqual(search.keenable_cost_usd(None), 0.004)
        self.assertAlmostEqual(search.keenable_cost_usd({"usage": {"credits": True}}), 0.004)
        self.assertAlmostEqual(search.keenable_cost_usd({"usage": None}), 0.004)
        with patch.object(search, "vendor_call", return_value=({"results": [], "usage": {"credits": 3}}, {})):
            backend = search.get_backend(ARM)
            backend.search("q")
            self.assertAlmostEqual(backend.last_meta["cost_usd"], 0.012)

    def test_fetch_indexed_request_contract(self):
        page = {"url": "https://a.test/final", "title": "T", "content": "x" * 20000,
                "published_at": 1761955200, "usage": {"credits": 1}}
        with patch.object(search, "vendor_call", _fake_vendor_call(page)) as call:
            backend = search.get_backend(ARM)
            out = backend.fetch("https://a.test")
        self.assertEqual(call.call_count, 1)
        self.assertEqual(call.call_args.args[1:3], ("GET", search.KEENABLE_FETCH_URL))
        self.assertEqual(call.call_args.kwargs["headers"]["X-API-Key"], "test-key")
        self.assertEqual(call.call_args.kwargs["params"],
                         {"url": "https://a.test", "max_chars": search.DEFAULT_MAX_FETCH_CHARS, "live": "false"})
        self.assertEqual(out["url"], "https://a.test/final")
        self.assertEqual(len(out["content"]), search.DEFAULT_MAX_FETCH_CHARS)
        self.assertTrue(out["_meta"]["truncated"])
        self.assertNotIn("attempts", out["_meta"])
        self.assertAlmostEqual(out["_meta"]["cost_usd"], 0.004)

    def test_fetch_http_error_is_failed_fetch_no_retry(self):
        for status in (404, 422, 500):
            with self.subTest(status=status), patch.object(
                search, "vendor_call", _fake_vendor_call(_http_error(status))
            ) as call:
                with self.assertRaises(requests.HTTPError):
                    search.get_backend(ARM).fetch("https://a.test")
                self.assertEqual(call.call_count, 1)

    def test_fetch_empty_content_is_failed_fetch_no_retry(self):
        for payload in ({"content": ""}, {"content": None}, {}, None):
            with self.subTest(payload=payload), patch.object(
                search, "vendor_call", _fake_vendor_call(payload)
            ) as call:
                backend = search.get_backend(ARM)
                with self.assertRaises(RuntimeError):
                    backend.fetch("https://a.test")
                self.assertEqual(call.call_count, 1)
                self.assertAlmostEqual(backend.last_meta["cost_usd"], 0.004)

    def test_api_key_never_in_request_audit(self):
        response = MagicMock(ok=True, status_code=200, content=b"{}", headers={}, text="{}")
        response.json.return_value = {"results": []}
        with patch.object(search.requests, "request", return_value=response) as sent:
            backend = search.get_backend(ARM)
            backend.search("q")
        self.assertEqual(sent.call_args.kwargs["headers"]["X-API-Key"], "test-key")
        self.assertNotIn("test-key", repr(backend.last_meta))
        self.assertEqual(backend.last_meta["request"]["headers"]["X-API-Key"], "***REDACTED***")

    def test_api_key_header_redacted(self):
        for header in ("X-API-Key", "x-api-key", "X-Api-Key"):
            self.assertEqual(search.redact_headers({header: "secret"}), {header: "***REDACTED***"})


if __name__ == "__main__":
    unittest.main()
