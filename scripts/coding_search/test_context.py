"""Context adapter contracts, including failures that are unsafe to induce live."""

import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import requests

from coding_search import search
from coding_search.eval import parse_arms


class ContextContracts(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"CONTEXT_API_KEY": "test-context-key"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.backend = search.get_backend("context")

    def response(self, payload, status=200):
        response = requests.Response()
        response.status_code = status
        response._content = json.dumps(payload).encode()
        return response

    def test_both_boards(self):
        for split in ("search-only", "search-fetch"):
            self.assertEqual(parse_arms("context", split=split), ("context",))
            self.assertIn("context", parse_arms("all", split=split))
        self.assertNotIn("context", search.FETCH_FORBIDDEN)
        self.assertNotIn("context", search.FETCH_REQUIRED)

    def test_search_limits_highlights_deduplication_and_redaction(self):
        rows = [
            {"url": "https://example.com", "title": "Example", "description": "x" * 1500,
             "highlights": {"code": "SUCCESS", "highlights": ["h" * 21000]},
             "markdown": {"markdown": "Do not send full pages in search"}},
            {"url": "https://example.com"}, {}, None,
            {"url": "https://example.org", "description": "description"},
        ]
        for limit, requested in ((1, 10), (8, 10), (20, 20), (101, 100)):
            with self.subTest(limit=limit), patch.object(
                search.requests, "request", return_value=self.response({"results": rows})
            ) as call:
                hits = self.backend.search("site:example.com unchanged query", max_results=limit)
                self.assertEqual(call.call_args.args, ("POST", search.CONTEXT_SEARCH_URL))
                self.assertEqual(call.call_args.kwargs["json"], {
                    "query": "site:example.com unchanged query", "numResults": requested,
                    "highlightsOptions": {"enabled": True, "maxCharacters": 20000},
                })
                self.assertEqual(hits[0], {
                    "url": "https://example.com", "title": "Example", "snippet": "h" * 20000,
                })
                self.assertEqual(len(hits), min(limit, 2))
                self.assertEqual(self.backend.last_meta["n_hits"], len(hits))
                self.assertNotIn("test-context-key", json.dumps(self.backend.last_meta))

    def test_highlights_and_description_fallback(self):
        for highlights, expected in (
            ({"code": "SUCCESS", "highlights": ["Relevant", "evidence", None]}, "Relevant\nevidence"),
            ({"code": "TIMEOUT", "highlights": ["not evidence"]}, "snippet"),
            ({"code": "SUCCESS", "highlights": []}, "snippet"),
            ({"code": "SUCCESS", "highlights": ["  "]}, "snippet"),
            ({"code": "NOT_REQUESTED", "highlights": None}, "snippet"),
        ):
            with self.subTest(highlights=highlights), patch.object(
                search.requests, "request", return_value=self.response({"results": [
                    {"url": "https://example.com", "description": "snippet", "highlights": highlights}
                ]})
            ):
                self.assertEqual(self.backend.search("query")[0]["snippet"], expected)

    def test_empty_search(self):
        with patch.object(search.requests, "request", return_value=self.response({"results": []})):
            self.assertEqual(self.backend.search("query"), [])
            self.assertTrue(self.backend.last_meta["empty"])

    def test_fetch_redirect_truncation_and_redaction(self):
        payload = {"success": True, "url": "https://example.com", "markdown": "x" * 20000,
                   "cache_metadata": {"status": "hit", "age_ms": 1000},
                   "metadata": {"finalUrl": "https://example.com/docs", "title": "Docs"}}
        with patch.object(search.requests, "request", return_value=self.response(payload)) as call:
            page = self.backend.fetch("https://example.com", objective="find docs")
            self.assertEqual(call.call_args.args, ("GET", search.CONTEXT_SCRAPE_URL))
            self.assertEqual(call.call_args.kwargs["params"], {
                "url": "https://example.com", "useMainContentOnly": "true",
            })
            self.assertEqual(page["url"], "https://example.com/docs")
            self.assertEqual(page["title"], "Docs")
            self.assertEqual(page["content"], "x" * search.DEFAULT_MAX_FETCH_CHARS)
            self.assertTrue(page["_meta"]["truncated"])
            self.assertEqual(page["_meta"]["cache_metadata"], payload["cache_metadata"])
            self.assertNotIn("test-context-key", json.dumps(page))

    def test_failed_and_malformed_responses_preserve_current_audit(self):
        cases = (
            ("search", "query", {}),
            ("search", "query", {"results": "invalid"}),
            ("fetch", "https://example.com", {"success": False, "markdown": "error page"}),
            ("fetch", "https://example.com", {"success": True, "markdown": ""}),
            ("fetch", "https://example.com", {"success": True, "markdown": {"error": "bad"}}),
        )
        for method, arg, payload in cases:
            with self.subTest(payload=payload), patch.object(
                search.requests, "request", return_value=self.response(payload)
            ):
                self.backend.last_meta = {"stale": True}
                with self.assertRaises(RuntimeError):
                    getattr(self.backend, method)(arg)
                self.assertEqual(self.backend.last_meta["response"]["body"], payload)
                self.assertNotIn("stale", self.backend.last_meta)

    def test_http_errors_and_missing_key(self):
        for method, arg in (("search", "query"), ("fetch", "https://example.com")):
            with self.subTest(method=method), patch.object(
                search.requests, "request", return_value=self.response({"message": "unauthorized"}, 401)
            ):
                with self.assertRaises(requests.HTTPError):
                    getattr(self.backend, method)(arg)
                self.assertEqual(self.backend.last_meta["http_status"], 401)
            with patch.dict(os.environ, {"CONTEXT_API_KEY": ""}), patch.object(search.requests, "request") as call:
                with self.assertRaisesRegex(RuntimeError, "CONTEXT_API_KEY"):
                    getattr(self.backend, method)(arg)
                call.assert_not_called()


@unittest.skipUnless(os.environ.get("CONTEXT_LIVE") == "1", "set CONTEXT_LIVE=1 for paid API calls")
class ContextLive(unittest.TestCase):
    def test_search_then_fetch(self):
        from coding_search.env import load_environment

        load_environment()
        backend = search.get_backend("context")
        query = "site:docs.python.org asyncio TaskGroup create_task"
        hits = backend.search(query)
        self.assertTrue(hits)
        self.assertLessEqual(len(hits), search.DEFAULT_MAX_RESULTS)
        self.assertTrue(any("TaskGroup" in hit["snippet"] for hit in hits))
        search_meta = dict(backend.last_meta)
        self.assertEqual(search_meta["request"]["json"]["highlightsOptions"], {"enabled": True, "maxCharacters": 20000})
        page = backend.fetch(hits[0]["url"])
        self.assertTrue(page["content"].strip())
        self.assertLessEqual(len(page["content"]), search.DEFAULT_MAX_FETCH_CHARS)
        report = {
            "query": query,
            "hits": hits,
            "search": search.public_meta(search_meta),
            "search_request": search_meta["request"],
            "page": {key: value for key, value in page.items() if key != "_meta"},
            "fetch": search.public_meta(backend.last_meta),
            "fetch_request": backend.last_meta["request"],
        }
        artifact = Path("runs/context-smoke.json")
        encoded = json.dumps(report, indent=2)
        self.assertNotIn(os.environ["CONTEXT_API_KEY"], encoded)
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(encoded + "\n")
        print(f"Context live search + fetch artifact: {artifact}")


if __name__ == "__main__":
    unittest.main()
