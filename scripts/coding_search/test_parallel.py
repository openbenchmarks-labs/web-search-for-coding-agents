"""Offline regressions for Parallel's site: query translation."""

import json
import os
import unittest
from unittest.mock import patch

import requests

from coding_search import search


class ParallelSitePolicyTests(unittest.TestCase):
    def test_embedded_site_text_is_not_an_operator(self):
        for query in (
            "website:docs.example.com authentication",
            "https://example.com/site:reference authentication",
            "my_site:docs.example.com authentication",
            "-site:old.example.com authentication",
        ):
            with self.subTest(query=query):
                self.assertEqual(search.parallel_site_policy(query), (query, []))

    def test_positive_operators_keep_host_normalization_and_deduplication(self):
        query = (
            "site:https://www.Example.com/docs authentication\t"
            "SITE:'docs.example.org/api'\nsite:example.com/ -site:old.example.com"
        )
        self.assertEqual(search.parallel_site_policy(query), (
            "authentication -site:old.example.com",
            ["example.com", "docs.example.org"],
        ))

    def test_literal_text_and_real_operators_can_coexist(self):
        self.assertEqual(search.parallel_site_policy(
            "website:literal.example site:docs.example.com authentication"
        ), ("website:literal.example authentication", ["docs.example.com"]))

    def test_registered_modes_send_the_translated_query_and_return_hits(self):
        hit = {"url": "https://docs.example.com/auth", "title": "Authentication",
               "snippet": "Use the current authentication API."}
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps({"results": [{
            "url": hit["url"], "title": hit["title"], "excerpts": [hit["snippet"]],
        }]}).encode()
        literal = "website:literal.example https://example.com/site:reference -site:old.example.com"
        cases = (
            (literal, literal, []),
            (literal + " site:docs.example.com auth", literal + " auth", ["docs.example.com"]),
            ("site:docs.example.com", "docs.example.com", ["docs.example.com"]),
        )
        for mode, timeout in (("basic", 60), ("advanced", 90), ("fast", 45), ("turbo", 45)):
            for query, sent, domains in cases:
                with self.subTest(mode=mode, query=query), patch.dict(
                    os.environ, {"PARALLEL_API_KEY": "test-key"}, clear=True
                ), patch.object(search.requests, "request", return_value=response) as request:
                    backend = search.get_backend("parallel_" + mode)
                    self.assertEqual(backend.search(query, max_results=2), [hit])
                    settings = {"max_results": 2}
                    if domains:
                        settings["source_policy"] = {"include_domains": domains}
                    request.assert_called_once_with(
                        "POST", search.PARALLEL_SEARCH_URL,
                        headers={"x-api-key": "test-key", "Content-Type": "application/json"},
                        json={"objective": sent, "search_queries": [sent], "mode": mode,
                              "advanced_settings": settings},
                        params=None, timeout=timeout,
                    )
                    self.assertEqual(backend.last_meta["hit_urls"], [hit["url"]])
                    self.assertEqual(backend.last_meta["n_hits"], 1)
                    if domains:
                        self.assertEqual(backend.last_meta["include_domains"], domains)
                        self.assertEqual(backend.last_meta["query_stripped"], sent)
                    else:
                        self.assertNotIn("include_domains", backend.last_meta)
                        self.assertNotIn("query_stripped", backend.last_meta)


if __name__ == "__main__":
    unittest.main()
