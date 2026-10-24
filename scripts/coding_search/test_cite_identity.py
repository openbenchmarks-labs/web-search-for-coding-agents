"""Citation URLs match the observed document, not a string prefix."""
import unittest

from coding_search.judge import url_was_observed


class CiteIdentity(unittest.TestCase):
    def test_rejects_a_different_document(self):
        pairs = [
            ("https://docs.example.com.evil.test/api", "https://docs.example.com"),
            ("https://docs.example.com/api/v10", "https://docs.example.com/api/v1"),
            ("https://docs.example.com/article?id=wrong", "https://docs.example.com/article?id=right"),
            ("https://docs.example.com/API", "https://docs.example.com/api"),
            ("https://docs.example.com/api/v1/widgets", "https://docs.example.com/api/v1"),
        ]
        for cited, observed in pairs:
            with self.subTest(cited=cited):
                self.assertFalse(url_was_observed(cited, [observed]))

    def test_accepts_the_same_document(self):
        observed = ["https://www.docs.example.com/api/v1/#section"]
        for cited in (
            "https://docs.example.com/api/v1",
            "http://docs.example.com/api/v1/",
        ):
            with self.subTest(cited=cited):
                self.assertTrue(url_was_observed(cited, observed))
        self.assertTrue(url_was_observed(
            "https://docs.example.com/api/v1?a=1&b=2",
            ["https://docs.example.com/api/v1?b=2&a=1"],
        ))


if __name__ == "__main__":
    unittest.main()
