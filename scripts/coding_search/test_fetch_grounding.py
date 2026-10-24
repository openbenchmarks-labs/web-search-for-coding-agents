"""A failed or empty fetch is not evidence that its URL was retrieved."""
import unittest

from coding_search.judge import hop_retrieved_text, hop_urls, score_needles
from coding_search.tasks import CodingTask


class FetchGrounding(unittest.TestCase):
    def test_failed_fetch_does_not_ground_a_citation(self):
        url = "https://docs.example.com/api/current"
        task = CodingTask(
            id="audit", repo="", outfile="answer.py", prompt="Find the field",
            required=["current_api_field"],
        )
        run = {
            "search_hops": [{"hits": []}],
            "fetch_hops": [{"url": url, "error": "HTTP 404", "content": ""}],
        }
        source = "\n".join([f"# source: {url}", 'value = "current_api_field"', ""])
        self.assertEqual(hop_retrieved_text(run), "")
        self.assertEqual(hop_urls(run), [])
        score = score_needles(source, task, observed_urls=hop_urls(run), retrieved_text=hop_retrieved_text(run))
        self.assertFalse(score.passed)
        self.assertTrue(any(note.startswith("ungrounded:") for note in score.notes))

    def test_failed_fetch_body_is_not_retrieved_text(self):
        run = {"fetch_hops": [{"url": "https://docs.example.com/api", "error": "timeout", "content": "current_api_field"}]}
        self.assertEqual(hop_retrieved_text(run), "")
        self.assertEqual(hop_urls(run), [])

    def test_successful_fetch_still_grounds(self):
        url = "https://docs.example.com/api/current"
        run = {"fetch_hops": [{"url": url, "error": None, "content": "current_api_field", "title": "API"}]}
        self.assertEqual(hop_urls(run), [url])
        self.assertIn("current_api_field", hop_retrieved_text(run))

    def test_empty_success_does_not_ground(self):
        run = {"fetch_hops": [{"url": "https://docs.example.com/api", "content": "   "}]}
        self.assertEqual(hop_urls(run), [])


if __name__ == "__main__":
    unittest.main()
