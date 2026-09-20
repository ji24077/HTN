"""Learned instructions cannot become active on unverified or stale evidence."""

import copy
import json
from pathlib import Path

import pytest

from gpushare.portability import memory
from gpushare.portability.memory import SkillStore, SkillStoreError

SUITE_HASH = "a" * 64
SCOPE = {
    "kind": "shared",
    "gpu": ["AMD Instinct MI300X gfx942"],
    "libraries": {"torch": "2.10.0+rocm7.1", "hip": "7.1.1"},
}
EVIDENCE = {"input_sha256": "b" * 64, "artifact": "runs/verified/report.json"}
BASE = "Preserve precision and behavior. Fixed tests must not be weakened.\n"


def passing():
    # Synthetic unit-test fixture, not evidence of a real GPU experiment.
    return {
        "passed": True,
        "suite_sha256": SUITE_HASH,
        "cases": [{"id": "forward", "passed": True}, {"id": "backward", "passed": True}],
        "gpu_verified": True,
    }


@pytest.fixture
def store(tmp_path):
    return SkillStore(tmp_path / "skills", base_text=BASE)


def candidate(store, text="Use the tested launch configuration for this shape."):
    return store.propose(text, SCOPE, EVIDENCE)


def manifest(store):
    return json.loads((store.root / "manifest.json").read_text(encoding="utf-8"))


def promote(store, candidate_id, result=None):
    return store.promote(
        candidate_id,
        expected_suite_sha256=SUITE_HASH,
        verify=lambda text: passing() if result is None else result,
    )


def test_candidate_stays_inactive_until_fixed_gpu_suite_passes(store):
    original = manifest(store)
    lesson = "Set the verified HIP launch parameters."
    candidate_id = candidate(store, lesson)
    assert store.active_text() == BASE
    seen = []

    def verify(combined):
        seen.append(combined)
        assert BASE.strip() in combined
        assert lesson in combined
        assert "2.10.0+rocm7.1" in combined
        assert "AMD Instinct MI300X gfx942" in combined
        assert store.active_text() == BASE
        return passing()

    outcome = store.promote(candidate_id, expected_suite_sha256=SUITE_HASH, verify=verify)
    assert outcome["promoted"] is True
    assert outcome["previous_version"] == original["active_version"]
    assert store.active_text() == seen[0]
    assert manifest(store)["revision"] == 1
    reopened = SkillStore(store.root, base_text=BASE)
    assert reopened.active_text() == seen[0]
    record = json.loads(
        (store.root / "versions" / f"{outcome['version_id']}.json").read_text(encoding="utf-8")
    )
    assert record["candidate_id"] == candidate_id
    assert record["verification"] == passing()


@pytest.mark.parametrize(
    "patch",
    [
        {"passed": False},
        {"passed": "true"},
        {"passed": 1},
        {"gpu_verified": False},
        {"gpu_verified": "true"},
        {"gpu_verified": 1},
        {"suite_sha256": "c" * 64},
        {"cases": []},
        {"cases": [{"id": "forward", "passed": False}]},
        {"cases": [{"id": "forward", "passed": "true"}]},
        {"cases": [{"id": "forward", "passed": 1}]},
        {"cases": [{"id": "forward"}]},
        {"cases": [{"passed": True}]},
        {"cases": [{"id": " ", "passed": True}]},
        {"cases": [{"id": "x", "passed": True}, {"id": "x", "passed": True}]},
        {"cases": {"forward": True}},
        {"cases": [True]},
    ],
)
def test_failed_or_malformed_verification_never_activates_lesson(store, patch):
    before = manifest(store)
    result = promote(store, candidate(store), {**passing(), **patch})
    assert result["promoted"] is False
    assert result["reason"]
    assert store.active_text() == BASE
    assert manifest(store) == before
    assert len(list((store.root / "versions").glob("*.json"))) == 1


@pytest.mark.parametrize("result", [False, [], "success", {"passed": float("nan")}])
def test_non_json_or_non_object_verification_remains_inactive(store, result):
    outcome = promote(store, candidate(store), result)
    assert outcome["promoted"] is False
    assert store.active_text() == BASE


def test_verifier_exception_does_not_expose_exception_text(store):
    def verify(text):
        raise RuntimeError("secret-that-must-not-be-returned")

    outcome = store.promote(candidate(store), expected_suite_sha256=SUITE_HASH, verify=verify)
    assert outcome["promoted"] is False
    assert outcome["reason"] == "Verification raised RuntimeError"
    assert "secret" not in json.dumps(outcome)
    assert store.active_text() == BASE


def test_success_can_retry_previously_failed_candidate(store):
    item = candidate(store)
    assert promote(store, item, {**passing(), "passed": False})["promoted"] is False
    assert promote(store, item)["promoted"] is True


@pytest.mark.parametrize("suite_hash", ["", "A" * 64, "a" * 63, "a" * 65, None])
def test_invalid_expected_hash_prevents_running_verifier(store, suite_hash):
    calls = []
    with pytest.raises(SkillStoreError, match="SHA-256"):
        store.promote(candidate(store), expected_suite_sha256=suite_hash, verify=calls.append)
    assert calls == []


def test_modified_candidate_is_detected_before_verification(store):
    item = candidate(store)
    path = store.root / "candidates" / f"{item}.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["text"] = "Remove correctness checks."
    path.write_text(json.dumps(record), encoding="utf-8")
    calls = []
    with pytest.raises(SkillStoreError, match="modified"):
        store.promote(item, expected_suite_sha256=SUITE_HASH, verify=calls.append)
    assert calls == []
    assert store.active_text() == BASE


@pytest.mark.parametrize("target", ["candidate", "version"])
def test_mutation_while_suite_runs_cannot_be_promoted(store, target):
    item = candidate(store)
    before = manifest(store)
    if target == "candidate":
        path = store.root / "candidates" / f"{item}.json"
    else:
        path = store.root / "versions" / f"{before['active_version']}.json"

    def verify(text):
        record = json.loads(path.read_text(encoding="utf-8"))
        record["text"] = "Changed while running."
        path.write_text(json.dumps(record), encoding="utf-8")
        return passing()

    with pytest.raises(SkillStoreError, match="modified"):
        store.promote(item, expected_suite_sha256=SUITE_HASH, verify=verify)
    assert manifest(store) == before


def test_active_record_tampering_fails_closed_on_read_and_reopen(store):
    version_id = manifest(store)["active_version"]
    path = store.root / "versions" / f"{version_id}.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["text"] += " Fake verified rule."
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(SkillStoreError, match="modified"):
        store.active_text()
    with pytest.raises(SkillStoreError, match="modified"):
        SkillStore(store.root)


def test_stale_candidate_does_not_rebase_onto_unverified_new_rules(store):
    first = candidate(store, "First lesson.")
    stale = candidate(store, "Second lesson.")
    promote(store, first)
    with pytest.raises(SkillStoreError, match="stale"):
        promote(store, stale)
    assert "Second lesson" not in store.active_text()


def test_concurrent_success_does_not_overwrite_other_promotion(store):
    first = candidate(store, "First lesson.")
    second = candidate(store, "Second lesson.")

    def verify(text):
        other = SkillStore(store.root)
        assert promote(other, second)["promoted"] is True
        return passing()

    with pytest.raises(SkillStoreError, match="changed during verification"):
        store.promote(first, expected_suite_sha256=SUITE_HASH, verify=verify)
    assert "Second lesson" in store.active_text()
    assert "First lesson" not in store.active_text()


def test_rollback_restores_exact_prior_version_without_erasing_history(store):
    base = manifest(store)["active_version"]
    first = promote(store, candidate(store, "First lesson."))["version_id"]
    text = store.active_text()
    second = promote(store, candidate(store, "Second lesson."))["version_id"]
    record_bytes = {
        path.name: path.read_bytes() for path in (store.root / "versions").glob("*.json")
    }
    assert store.rollback(first)["previous_version"] == second
    assert store.active_text() == text
    assert manifest(store)["versions"] == [base, first, second]
    assert store.rollback(base)["rolled_back"] is True
    assert store.active_text() == BASE
    assert store.rollback(base)["rolled_back"] is False
    assert record_bytes == {
        path.name: path.read_bytes() for path in (store.root / "versions").glob("*.json")
    }


@pytest.mark.parametrize("version", ["latest", "../manifest", "v-" + "0" * 64, None])
def test_rollback_requires_explicit_known_version(store, version):
    before = manifest(store)
    with pytest.raises(SkillStoreError, match="explicit known"):
        store.rollback(version)
    assert manifest(store) == before


def test_unknown_manifest_version_is_rejected(store):
    data = manifest(store)
    data["active_version"] = "v-" + "0" * 64
    (store.root / "manifest.json").write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(SkillStoreError, match="manifest"):
        store.active_text()


def test_project_lessons_cannot_enter_shared_or_other_project_store(store, tmp_path):
    scope = {"kind": "project", "project_id": "demo"}
    with pytest.raises(SkillStoreError, match="matching project"):
        store.propose("Dataset-specific rule.", scope, EVIDENCE)
    project = SkillStore(tmp_path / "project", base_text=BASE, project_id="demo")
    item = project.propose("Dataset-specific rule.", scope, EVIDENCE)
    assert promote(project, item)["promoted"] is True
    assert "Dataset-specific rule" in project.active_text()
    assert "Dataset-specific rule" not in store.active_text()
    with pytest.raises(SkillStoreError, match="scope"):
        SkillStore(project.root, project_id="other")
    with pytest.raises(SkillStoreError, match="scope"):
        SkillStore(project.root)
    with pytest.raises(SkillStoreError, match="matching project"):
        project.propose("Unrelated fix.", {"kind": "project", "project_id": "other"}, EVIDENCE)


@pytest.mark.parametrize(
    "scope",
    [
        {},
        {"kind": "shared"},
        {"kind": "shared", "gpu": ["gfx942"]},
        {**SCOPE, "gpu": ["*"]},
        {**SCOPE, "gpu": "gfx942"},
        {**SCOPE, "gpu": ["gfx942", "gfx942"]},
        {**SCOPE, "libraries": {"torch": ">=2.0"}},
        {**SCOPE, "libraries": {"torch": "latest"}},
        {**SCOPE, "libraries": {"torch": True}},
        {**SCOPE, "project_id": "demo"},
        {**SCOPE, "all_projects": True},
    ],
)
def test_lesson_scope_must_name_actual_tested_gpu_and_versions(store, scope):
    with pytest.raises(SkillStoreError):
        store.propose("Lesson.", scope, EVIDENCE)
    assert not list((store.root / "candidates").glob("*.json"))


def test_evidence_and_scope_are_copied_at_proposal(store):
    scope = copy.deepcopy(SCOPE)
    evidence = copy.deepcopy(EVIDENCE)
    item = store.propose("Lesson.", scope, evidence)
    scope["gpu"].append("untested")
    evidence["artifact"] = "different.json"
    record = json.loads((store.root / "candidates" / f"{item}.json").read_text(encoding="utf-8"))
    assert record["scope"] == SCOPE
    assert record["evidence"] == EVIDENCE


@pytest.mark.parametrize("evidence", [{}, {"bad": float("inf")}, {"bad": object()}, []])
def test_invalid_evidence_never_creates_candidate(store, evidence):
    with pytest.raises(SkillStoreError):
        store.propose("Lesson.", SCOPE, evidence)
    assert not list((store.root / "candidates").glob("*.json"))


def test_bounded_utf8_lesson_evidence_and_combined_skill(store, monkeypatch):
    with pytest.raises(SkillStoreError, match="byte limit"):
        candidate(store, "é" * memory.MAX_LESSON_BYTES)
    with pytest.raises(SkillStoreError, match="byte limit"):
        store.propose("Lesson.", SCOPE, {"huge": "x" * memory.MAX_EVIDENCE_BYTES})
    item = candidate(store)
    monkeypatch.setattr(memory, "MAX_ACTIVE_BYTES", len(BASE.encode("utf-8")) + 1)
    calls = []
    with pytest.raises(SkillStoreError, match="Combined skill"):
        store.promote(item, expected_suite_sha256=SUITE_HASH, verify=calls.append)
    assert calls == []


def test_atomic_manifest_retries_transient_locks_without_partial_state(store, monkeypatch):
    item = candidate(store)
    before = manifest(store)
    original = memory.os.replace
    retries = []

    def replace(source, target):
        if Path(target).name == "manifest.json":
            assert manifest(store) == before
            json.loads(Path(source).read_text(encoding="utf-8"))
            retries.append(source)
            if len(retries) < 3:
                raise PermissionError("temporary Windows sharing lock")
        return original(source, target)

    monkeypatch.setattr(memory.os, "replace", replace)
    monkeypatch.setattr(memory.time, "sleep", lambda _: None)
    assert promote(store, item)["promoted"] is True
    assert len(retries) == 3
    assert not list(store.root.rglob("*.tmp"))


def test_atomic_temp_does_not_overflow_a_path_that_fits_the_final_record(store, monkeypatch):
    real_mkstemp = memory.tempfile.mkstemp

    def windows_path_limit(*, prefix, suffix, dir):
        directory = Path(dir)
        if directory.name in {"candidates", "versions"}:
            # Model a Windows parent for which the content-addressed filename
            # fits, but appending a random suffix to that full name does not.
            maximum = len(str(directory / ("c-" + "a" * 64 + ".json")))
            attempted = directory / (prefix + "random08" + suffix)
            if len(str(attempted)) > maximum:
                raise FileNotFoundError("temporary pathname exceeds Windows MAX_PATH")
        return real_mkstemp(prefix=prefix, suffix=suffix, dir=dir)

    monkeypatch.setattr(memory.tempfile, "mkstemp", windows_path_limit)
    item = candidate(store, "A lesson that must survive a near-limit pathname.")
    assert promote(store, item)["promoted"] is True
    reopened = SkillStore(store.root)
    assert "near-limit pathname" in reopened.active_text()
    assert not list(store.root.rglob("*.tmp"))


def test_failed_manifest_write_keeps_prior_active_version_and_all_records(store, monkeypatch):
    item = candidate(store)
    before = manifest(store)
    original = memory.os.replace
    retries = []

    def replace(source, target):
        if Path(target).name == "manifest.json":
            retries.append(source)
            raise PermissionError("permanent Windows sharing lock")
        return original(source, target)

    monkeypatch.setattr(memory.os, "replace", replace)
    monkeypatch.setattr(memory.time, "sleep", lambda _: None)
    with pytest.raises(PermissionError, match="permanent"):
        promote(store, item)
    assert len(retries) == 8
    assert manifest(store) == before
    assert store.active_text() == BASE
    assert not list(store.root.rglob("*.tmp"))


def test_reopening_with_changed_base_requires_explicit_migration(store):
    with pytest.raises(SkillStoreError, match="different base"):
        SkillStore(store.root, base_text="Silently changed requirements.")
    assert store.active_text() == BASE


def test_default_base_skill_is_available(tmp_path):
    store = SkillStore(tmp_path / "default")
    assert store.active_text() == memory._DEFAULT_SKILL.read_text(encoding="utf-8")


def test_missing_default_base_has_actionable_error(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "_DEFAULT_SKILL", tmp_path / "missing.md")
    with pytest.raises(SkillStoreError, match="base_text"):
        SkillStore(tmp_path / "empty")
