"""Offline contracts for judge-evidence persistence and `replay` (issue #16)."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from coding_search.agent import AgentRun, FetchCall, SearchCall
from coding_search.eval import main
from coding_search.judge import hop_retrieved_text, hop_urls, score_needles
from coding_search.replay import judge_evidence, judge_task, replay_paths, replay_row
from coding_search.tasks import CodingTask

NEEDLE = "current_api_field"
SOURCE = "\n".join(["# source: https://citation.example.com/reference", 'value = "current_api_field"', ""])


def make_task() -> CodingTask:
    return CodingTask(id="audit", repo="", outfile="answer.py", prompt="Find the field", required=[NEEDLE])


def make_run(*, fetch: bool = True) -> AgentRun:
    """Needle sits past the clip: char 1000 of a fetch body (preview keeps 800),
    or char 1300 of a search snippet (summary keeps 1200)."""
    run = AgentRun(question="", answer=SOURCE, model="offline", backend="audit", used_search=True)
    if fetch:
        run.fetches = [FetchCall(url="https://docs.example.com/page", backend="audit", content="x" * 1000 + NEEDLE)]
    else:
        run.searches = [SearchCall(query="field", backend="audit",
                                   hits=[{"url": "https://docs.example.com/s", "title": "t", "snippet": "y" * 1300 + NEEDLE}])]
    return run


def make_row(run: AgentRun, task: CodingTask, *, with_evidence: bool = True) -> dict:
    """Same shape run_arm writes."""
    score = score_needles(SOURCE, task, observed_urls=hop_urls(run), retrieved_text=hop_retrieved_text(run))
    row = {
        "id": task.id,
        "backend": "audit+fetch",
        "search_hops": [call.hop_dict() for call in run.searches],
        "fetch_hops": [call.hop_dict() for call in run.fetches],
        "passed": score.passed,
        "score": score.to_dict(),
        "source": SOURCE,
        "error": None,
    }
    if with_evidence:
        row["judge_task"] = judge_task(task)
        row["judge_evidence"] = judge_evidence(run)
    return row


class ReplayContracts(unittest.TestCase):
    def test_issue_16_summaries_alone_do_not_reproduce(self):
        # The reproduction from the issue: live passes, clipped summaries fail.
        task = make_task()
        for fetch in (True, False):
            with self.subTest(fetch=fetch):
                row = make_row(make_run(fetch=fetch), task, with_evidence=False)
                self.assertTrue(row["passed"])
                report = replay_row(row, {task.id: task})
                self.assertEqual(report["evidence_source"], "hop_summaries")
                self.assertEqual(report["status"], "mismatch")
                self.assertEqual(report["replay_notes"], [f"ungrounded:{NEEDLE}"])

    def test_judge_evidence_reproduces_live_verdict(self):
        task = make_task()
        for fetch in (True, False):
            with self.subTest(fetch=fetch):
                row = make_row(make_run(fetch=fetch), task)
                report = replay_row(row)  # no dataset access needed
                self.assertEqual(report["evidence_source"], "judge_evidence")
                self.assertEqual(report["status"], "match")
                self.assertTrue(report["replay_passed"])

    def test_evidence_is_exactly_what_the_judge_saw(self):
        run = make_run()
        evidence = judge_evidence(run)
        self.assertEqual(evidence["retrieved_text"], hop_retrieved_text(run))
        self.assertEqual(evidence["observed_urls"], hop_urls(run))
        self.assertIn(NEEDLE, evidence["retrieved_text"])
        self.assertEqual(len(evidence["sha256"]), 64)

    def test_failing_runs_replay_as_failing(self):
        task = make_task()
        run = make_run()
        run.fetches[0].content = "no needle here"
        row = make_row(run, task)
        self.assertFalse(row["passed"])
        report = replay_row(row)
        self.assertEqual(report["status"], "match")
        self.assertFalse(report["replay_passed"])

    def test_tampered_evidence_is_flagged(self):
        row = make_row(make_run(), make_task())
        row["judge_evidence"]["retrieved_text"] += " edited"
        self.assertEqual(replay_row(row)["status"], "corrupt")

    def test_rows_that_cannot_be_replayed_are_skipped(self):
        row = make_row(make_run(), make_task())
        no_source = {k: v for k, v in row.items() if k != "source"}
        self.assertEqual(replay_row(no_source)["status"], "skipped")
        legacy = make_row(make_run(), make_task(), with_evidence=False)
        self.assertEqual(replay_row(legacy)["status"], "skipped")  # no judge_task, no dataset
        errored = dict(row, error="RuntimeError: boom")
        self.assertEqual(replay_row(errored)["status"], "skipped")

    def test_run_arm_rows_carry_replayable_evidence(self):
        from unittest.mock import patch

        from coding_search import eval as eval_mod

        task = make_task()
        with patch.object(eval_mod, "run_agent", return_value=make_run()):
            row = eval_mod.run_arm(
                task, client=None, backend="audit", allow_fetch=True, model="offline",
                max_turns=1, max_searches=1, max_fetches=1,
            )
        self.assertTrue(row["passed"])
        self.assertEqual(row["judge_task"]["required"], [NEEDLE])
        row = json.loads(json.dumps(row))  # survives the --out round trip
        self.assertEqual(replay_row(row)["status"], "match")

    def test_replay_cli_reads_eval_summary(self):
        task = make_task()
        good = make_row(make_run(), task)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "eval.json"
            path.write_text(json.dumps({"n": 1, "results": [good]}))
            self.assertEqual(len(replay_paths([path])), 1)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(main(["replay", str(path)]), 0)
            self.assertIn("match=1 mismatch=0", out.getvalue())

            bad = dict(good, judge_evidence=dict(good["judge_evidence"], sha256="0" * 64))
            path.write_text(json.dumps({"n": 1, "results": [bad]}))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["replay", str(path)]), 1)


if __name__ == "__main__":
    unittest.main()
