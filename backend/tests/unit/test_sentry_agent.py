"""Self-heal publication, retry, and remote-base regressions; no external services."""

import argparse
import importlib.util
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "sentry-agent.py"
spec = importlib.util.spec_from_file_location("sentry_agent", SCRIPT)
agent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent)

ISSUE = {
    "shortId": "HTN-BACKEND-7",
    "title": "Failure for private-person@example.test at 192.0.2.42",
    "culprit": "private-host.internal",
    "permalink": "https://sentry.io/issues/7/",
}
VERDICT = {
    "action": "fix",
    "title": "Handle missing task data",
    "root_cause": "Missing data was not handled",
    "summary": "Validate the task shape",
    "tests": "Regression test passed",
    "confidence": "high",
}


def args(**overrides):
    return argparse.Namespace(
        **{
            "base": "main",
            "dry_run": False,
            "budget": 3,
            "issue": None,
            "max": 20,
            "watch": None,
            **overrides,
        }
    )


class HealTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.state_path = self.root / "state.json"
        self.worktree = self.root / "worktrees" / "htn-backend-7"
        self.worktree.joinpath("docs").mkdir(parents=True)
        self.state = {}

        def run(command, **_kwargs):
            return SimpleNamespace(
                stdout="[]" if command[1:3] == ["pr", "list"] else "https://github.test/pull/1\n"
            )

        for name, value in (
            ("STATE", self.state_path),
            ("git", lambda *a, **kw: "abc123"),
            ("investigate", lambda *a: VERDICT.copy()),
            ("tool", lambda name: name),
            ("run", run),
        ):
            mocked = patch.object(agent, name, value)
            mocked.start()
            self.addCleanup(mocked.stop)

    def test_public_incident_uses_only_sanitized_verdict(self):
        agent.heal_issue(ISSUE, self.state, args())
        content = self.worktree.joinpath(agent.INCIDENTS).read_text()
        incident = json.loads(content)
        self.assertEqual(incident["title"], VERDICT["title"])
        self.assertNotIn("culprit", incident)
        for private_value in (ISSUE["title"], ISSUE["culprit"], "192.0.2.42"):
            self.assertNotIn(private_value, content)
        self.assertEqual(self.state[ISSUE["shortId"]]["status"], "published")
        self.assertTrue(agent.should_skip(self.state[ISSUE["shortId"]]))

    def test_dry_run_is_unpublished_and_live_poll_retries_it(self):
        agent.heal_issue(ISSUE, self.state, args(dry_run=True))
        self.assertEqual(self.state[ISSUE["shortId"]]["status"], "dry_run")
        self.assertFalse(agent.should_skip(self.state[ISSUE["shortId"]]))
        with (
            patch.object(agent, "sentry", return_value=[ISSUE]),
            patch.object(agent, "PROJECTS", ("htn-backend",)),
            patch.object(agent, "heal_issue") as heal_issue,
        ):
            agent.heal(args())
        heal_issue.assert_called_once()

    def test_agent_error_is_persisted_but_retryable(self):
        with patch.object(
            agent,
            "investigate",
            return_value={
                "action": "error",
                "summary": "temporary agent failure",
            },
        ):
            agent.heal_issue(ISSUE, self.state, args())
        self.assertEqual(self.state[ISSUE["shortId"]]["status"], "error")
        self.assertFalse(agent.should_skip(self.state[ISSUE["shortId"]]))

    def test_repeated_dry_run_preserves_review_worktree(self):
        agent.heal_issue(ISSUE, self.state, args(dry_run=True))
        record = self.worktree.joinpath(agent.INCIDENTS).read_text()
        with (
            patch.object(agent, "sentry", return_value=[ISSUE]),
            patch.object(agent, "PROJECTS", ("htn-backend",)),
            patch.object(agent, "investigate") as investigate,
            patch.object(agent, "git") as git,
        ):
            agent.heal(args(dry_run=True))
        investigate.assert_not_called()
        git.assert_not_called()
        self.assertEqual(self.worktree.joinpath(agent.INCIDENTS).read_text(), record)
        self.assertTrue(agent.should_skip({"status": "fix"}, dry_run=True))

    def test_repeated_agent_errors_stop_after_three_attempts(self):
        with (
            patch.object(agent, "sentry", return_value=[ISSUE]),
            patch.object(agent, "PROJECTS", ("htn-backend",)),
            patch.object(
                agent,
                "investigate",
                return_value={
                    "action": "error",
                    "summary": "budget exhausted",
                },
            ) as investigate,
        ):
            for _ in range(4):
                agent.heal(args())
            self.assertEqual(investigate.call_count, 3)
            entry = agent.load_state()[ISSUE["shortId"]]
            self.assertEqual(entry["attempts"], 3)
            self.assertTrue(agent.should_skip(entry))
            # An explicitly selected issue can still be retried by its operator.
            with patch.object(agent, "sentry", return_value=ISSUE):
                agent.heal(args(issue=ISSUE["shortId"]))
            self.assertEqual(investigate.call_count, 4)

    def test_failed_pr_creation_resumes_same_branch_without_another_agent(self):
        def failed_create(command, **_kwargs):
            if command[1:3] == ["pr", "list"]:
                return SimpleNamespace(stdout="[]")
            raise RuntimeError("GitHub temporarily unavailable")

        with (
            patch.object(agent, "run", side_effect=failed_create),
            patch.object(agent, "git", return_value="abc123") as git,
        ):
            with self.assertRaisesRegex(RuntimeError, "temporarily unavailable"):
                agent.heal_issue(ISSUE, self.state, args())
        first_push = next(c.args for c in git.call_args_list if c.args[0] == "push")
        saved = agent.load_state()
        self.assertEqual(saved[ISSUE["shortId"]]["status"], "pending_publish")
        with (
            patch.object(agent, "investigate") as investigate,
            patch.object(agent, "git") as git,
        ):
            agent.heal_issue(ISSUE, saved, args())
        investigate.assert_not_called()
        self.assertEqual([c.args for c in git.call_args_list if c.args[0] == "push"], [first_push])
        self.assertFalse(any(c.args[:2] == ("worktree", "add") for c in git.call_args_list))
        self.assertEqual(agent.load_state()[ISSUE["shortId"]]["status"], "published")

    def test_lost_pr_response_reuses_existing_pr_without_pushing(self):
        self.state[ISSUE["shortId"]] = {
            "status": "pending_publish",
            "branch": "self-heal/htn-backend-7",
            "attempts": 1,
            "publication": {"base": "main"},
        }
        with (
            patch.object(
                agent,
                "run",
                return_value=SimpleNamespace(
                    stdout='[{"url": "https://github.test/pull/1", "headRefOid": "abc123"}]',
                ),
            ) as run,
            patch.object(agent, "git", return_value="abc123") as git,
            patch.object(agent, "investigate") as investigate,
        ):
            agent.heal_issue(ISSUE, self.state, args())
        investigate.assert_not_called()
        self.assertEqual(run.call_count, 1)
        self.assertFalse(any(c.args[0] == "push" for c in git.call_args_list))
        self.assertTrue(agent.should_skip(agent.load_state()[ISSUE["shortId"]]))

    def test_old_pr_for_same_issue_does_not_mark_new_commit_published(self):
        def run(command, **_kwargs):
            if command[1:3] == ["pr", "list"]:
                return SimpleNamespace(
                    stdout='[{"url": "https://github.test/pull/old", "headRefOid": "old-commit"}]'
                )
            return SimpleNamespace(stdout="https://github.test/pull/new\n")

        with (
            patch.object(agent, "run", side_effect=run) as calls,
            patch.object(agent, "git", return_value="new-commit") as git,
        ):
            agent.heal_issue(ISSUE, self.state, args())
        self.assertTrue(any(c.args[0] == "push" for c in git.call_args_list))
        self.assertTrue(any(c.args[0][1:3] == ["pr", "create"] for c in calls.call_args_list))
        self.assertEqual(agent.load_state()[ISSUE["shortId"]]["pr"], "https://github.test/pull/new")

    def test_pending_publication_is_bounded_and_not_retried_by_dry_run(self):
        entry = {"status": "pending_publish", "attempts": 2}
        self.assertFalse(agent.should_skip(entry))
        self.assertTrue(agent.should_skip(entry, dry_run=True))
        entry["attempts"] = 3
        self.assertTrue(agent.should_skip(entry))

    def test_poll_skips_only_published_or_dismissed_issues(self):
        entries = [
            {"status": "dry_run"},
            {"status": "error"},
            {"status": "fix"},
            {"status": "no_fix"},
            {"status": "published", "pr": "https://github.test/1"},
            {"status": "fix", "pr": "https://github.test/2"},
        ]
        issues = [{**ISSUE, "shortId": f"HTN-BACKEND-{i}"} for i in range(len(entries))]
        agent.save_state(dict(zip([issue["shortId"] for issue in issues], entries)))
        with (
            patch.object(agent, "sentry", return_value=issues),
            patch.object(agent, "PROJECTS", ("htn-backend",)),
            patch.object(agent, "heal_issue") as heal_issue,
        ):
            agent.heal(args())
        self.assertEqual([c.args[0] for c in heal_issue.call_args_list], issues[:3])


class RemoteBaseTests(unittest.TestCase):
    def test_live_base_refreshes_without_publishing_local_commits(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            remote, publisher, local = (root / name for name in ("remote", "publisher", "local"))

            def git(*command, cwd=root, **_kwargs):
                return subprocess.run(
                    ["git", "-c", "commit.gpgsign=false", *command],
                    cwd=cwd,
                    env={
                        **{k: v for k, v in os.environ.items() if not k.startswith("GIT_")},
                        "GIT_CONFIG_GLOBAL": os.devnull,
                        "GIT_CONFIG_NOSYSTEM": "1",
                    },
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()

            git("init", "--bare", str(remote))
            git("init", "-b", "main", str(publisher))
            git("config", "user.email", "test@example.test", cwd=publisher)
            git("config", "user.name", "Test", cwd=publisher)
            git("commit", "--allow-empty", "-m", "initial", cwd=publisher)
            git("remote", "add", "origin", str(remote), cwd=publisher)
            git("push", "origin", "main", cwd=publisher)
            git("clone", "--branch", "main", str(remote), str(local))
            git("config", "user.email", "test@example.test", cwd=local)
            git("config", "user.name", "Test", cwd=local)
            git("commit", "--allow-empty", "-m", "private local work", cwd=local)
            private = git("rev-parse", "HEAD", cwd=local)
            with patch.object(agent, "git", lambda *a, **kw: git(*a, cwd=local)):
                for index in range(2):
                    (publisher / "incidents.jsonl").write_text(f"incident {index}\n")
                    git("add", "incidents.jsonl", cwd=publisher)
                    git("commit", "-m", f"merged fix {index}", cwd=publisher)
                    git("push", "origin", "main", cwd=publisher)
                    expected = git("rev-parse", "HEAD", cwd=publisher)
                    self.assertEqual(agent.base_commit(args()), expected)
                    self.assertNotEqual(expected, private)
                self.assertEqual(agent.base_commit(args(dry_run=True)), private)
            self.assertEqual(git("rev-parse", "HEAD", cwd=local), private)
