"""CPU-only fault injection for spending, ownership, and cleanup boundaries."""

import copy
import io
import json
import urllib.error

import pytest

from gpushare.agent.runpod_session import (
    APIError,
    RunpodClient,
    Session,
    SessionError,
    UncertainRequest,
    pod_name,
    utc,
)

SESSION_ID = "migration-test-123456"
NOW = 1_790_000_000.0


class Clock:
    def __init__(self):
        self.value = NOW

    def __call__(self):
        return self.value


class FakeClient:
    def __init__(self, clock):
        self.clock = clock
        self.pods = {}
        self.creates = 0
        self.deletes = []
        self.fail_create = None
        self.uncertain_create = None
        self.reject_quote = False
        self.keep_after_delete = False
        self.actual_cost = None

    def list_pods(self):
        return copy.deepcopy(list(self.pods.values()))

    def check_quote(self, record, *, allow_unavailable_target_probe=False):
        if self.reject_quote:
            raise SessionError("GPU quote changed")

    def create_pod(self, payload):
        self.creates += 1
        if self.fail_create == self.creates:
            raise APIError(400)
        pod_id = f"pod-{self.creates:05d}"
        pod = {"id": pod_id, "name": payload["name"], "gpu": payload["gpu"],
               "cloud": payload["cloud"], "createdAt": utc(self.clock()), "status": "RUNNING"}
        self.pods[pod_id] = pod
        if self.actual_cost is not None:
            pod["cost"] = self.actual_cost
        if self.uncertain_create == self.creates:
            raise UncertainRequest("Request timed out")
        return copy.deepcopy(pod)

    def get_pod(self, pod_id):
        return copy.deepcopy(self.pods.get(pod_id))

    def terminate_pod(self, pod_id):
        self.deletes.append(pod_id)
        if not self.keep_after_delete:
            self.pods.pop(pod_id, None)


def make_plan():
    pods = []
    for role, gpu, rate in [
        ("source", "NVIDIA GeForce RTX 4090", 0.74),
        ("target", "AMD Instinct MI300X OAM", 2.39),
    ]:
        pods.append({"role": role, "hourly_gpu_usd": rate, "quoted_at_utc": utc(NOW),
                     "available": True, "payload": {
                         "name": pod_name(SESSION_ID, role), "image": "official/image:pinned",
                         "gpu": {"id": gpu, "count": 1}, "cloud": "SECURE", "disk": 50,
                         "dataCenterIds": ["EU-RO-1"], "ports": ["22/tcp"],
                         "env": {"PUBLIC_KEY": "ssh-ed25519 public-test-value"},
                         "entrypoint": ["/bin/bash", "-lc"], "cmd": ["exec sshd -D"],
                     }})
    return {"budget_usd": 15, "max_duration_seconds": 5400,
            "storage_allowance_per_hour": 0.10, "pods": pods}


def latency_plan():
    from gpushare.agent.runpod_session import PROFILES

    plan = make_plan()
    template = copy.deepcopy(plan["pods"][0])
    plan.update(profile="inference-latency", storage_allowance_per_hour=.15, budget_usd=13.5)
    plan["pods"] = []
    for role, (gpu, cloud, centers, rate) in PROFILES["inference-latency"].items():
        item = copy.deepcopy(template)
        item.update(role=role, hourly_gpu_usd=rate)
        item["payload"].update(name=pod_name(SESSION_ID, role), gpu={"id": gpu, "count": 1},
                               cloud=cloud, dataCenterIds=centers)
        plan["pods"].append(item)
    return plan


@pytest.mark.parametrize("key", ["rtx5090", "a5000", "a6000", "a40", "l4", "l40s", "a100", "h100", "rtx5080"])
def test_hardware_expansion_uses_its_reservation_and_cleans_only_owned_gpu(tmp_path, key):
    from gpushare.agent.gpu_matrix import BY_KEY, NEW_TARGET_KEYS, expansion_plan

    campaign = expansion_plan([*NEW_TARGET_KEYS, "rtx5080"], prior_reserve_usd=4.25)
    reservation = next(r for r in campaign["reservations"] if r["key"] == key)
    target = BY_KEY[key]
    clock = Clock()
    client = FakeClient(clock)
    client.pods["other-pod"] = {"id": "other-pod", "name": "teammate-owned"}
    session = Session(tmp_path, SESSION_ID, client, clock=clock, sleep=lambda _: None)
    plan = make_plan()
    plan.update(profile=target.profile, budget_usd=reservation["budget_usd"],
                max_duration_seconds=reservation["max_duration_seconds"],
                storage_allowance_per_hour=reservation["storage_allowance_per_hour"])
    plan["pods"] = plan["pods"][:1]
    plan["pods"][0].update(hourly_gpu_usd=target.hourly_limit)
    plan["pods"][0]["payload"].update(gpu={"id": target.gpu_id, "count": 1, "minCudaVersion": "12.8"},
                                       cloud=target.cloud, dataCenterIds=list(target.regions))
    session.prepare(plan)
    session.heartbeat()
    session.create(plan)
    assert client.creates == 1
    assert session.cleanup()["state"] == "closed"
    assert set(client.pods) == {"other-pod"}


def test_latency_profile_owns_and_cleans_exactly_three_approved_pods(tmp_path):
    clock = Clock()
    client = FakeClient(clock)
    session = Session(tmp_path, SESSION_ID, client, clock=clock, sleep=lambda _: None)
    plan = latency_plan()
    session.prepare(plan)
    session.heartbeat()
    session.create(plan)
    assert len(session.read()["pods"]) == 3
    assert client.creates == 3
    session.cleanup()
    assert len(client.deletes) == 3 and not client.pods


@pytest.mark.parametrize("gpu", ["3090", "4090", "amd"])
def test_independent_gpu_session_leaves_existing_comparison_pods_untouched(tmp_path, gpu):
    clock = Clock()
    client = FakeClient(clock)
    client.pods["other-4090"] = {"id": "other-4090", "name": "existing-comparison"}
    session = Session(tmp_path, SESSION_ID, client, clock=clock, sleep=lambda _: None)
    plan = latency_plan() if gpu == "3090" else make_plan()
    plan["profile"] = f"inference-latency-{gpu}"
    role = "target" if gpu == "amd" else "source"
    plan["pods"] = [item for item in plan["pods"] if item["role"] == role]
    session.prepare(plan)
    session.heartbeat()
    session.create(plan)
    assert client.creates == 1
    assert session.cleanup()["state"] == "closed"
    assert client.deletes == ["pod-00001"]
    assert set(client.pods) == {"other-4090"}


@pytest.mark.parametrize("change", ["cloud", "count", "price", "gpu", "region", "budget", "driver", "unknown"])
def test_latency_profile_rejects_unapproved_rentals_before_create(tmp_path, change):
    clock = Clock()
    client = FakeClient(clock)
    session = Session(tmp_path, SESSION_ID, client, clock=clock)
    plan = latency_plan()
    item = plan["pods"][0]
    if change == "cloud":
        item["payload"]["cloud"] = "SECURE"
    elif change == "count":
        item["payload"]["gpu"]["count"] = 2
    elif change == "price":
        item["hourly_gpu_usd"] = 1
    elif change == "gpu":
        item["payload"]["gpu"]["id"] = "NVIDIA H100"
    elif change == "region":
        plan["pods"][1]["payload"]["dataCenterIds"] = ["US-OTHER"]
    elif change == "budget":
        plan["budget_usd"] = 1
    elif change == "driver":
        item["payload"]["gpu"]["minCudaVersion"] = "not-a-version"
    else:
        plan["profile"] = "anything"
    with pytest.raises(SessionError):
        session.prepare(plan)
    assert client.creates == 0


def test_community_quote_checks_driver_constraint_before_creation(monkeypatch):
    from gpushare.agent.runpod_session import RunpodClient

    client = RunpodClient("test-key")
    calls = []
    record = {"role": "source", "gpu_id": "NVIDIA GeForce RTX 3090", "gpu_count": 1,
              "cloud": "COMMUNITY", "data_center_ids": [], "hourly_gpu_usd": .22,
              "min_cuda_version": "12.4"}

    def catalog(method, path):
        calls.append((method, path))
        return {"gpus": [{"id": record["gpu_id"], "community": True,
                          "price": {"community": .22}, "availability": "LOW"}]}

    monkeypatch.setattr(client, "request", catalog)
    assert client.check_quote(record)["approved_data_center_available"]
    assert calls[0][0] == "GET" and "minCudaVersion=12.4" in calls[0][1]


@pytest.fixture
def prepared(tmp_path):
    clock = Clock()
    client = FakeClient(clock)
    session = Session(tmp_path, SESSION_ID, client, clock=clock, sleep=lambda _: None)
    plan = make_plan()
    session.prepare(plan)
    session.heartbeat()
    return session, client, clock, plan


def test_budget_rejected_before_any_create(tmp_path):
    clock = Clock()
    client = FakeClient(clock)
    session = Session(tmp_path, SESSION_ID, client, clock=clock)
    plan = make_plan()
    plan["budget_usd"] = 4
    with pytest.raises(SessionError, match="budget"):
        session.prepare(plan)
    assert client.creates == 0
    assert not session.path.exists()


def test_fixed_deadline_and_no_secrets_or_environment_in_journal(prepared):
    session, _, clock, _ = prepared
    journal = session.read()
    assert journal["deadline"] == clock() + 5400
    raw = session.path.read_text()
    assert "PUBLIC_KEY" not in raw
    assert "public-test-value" not in raw
    assert '"env"' not in raw
    assert session._budget(journal, clock()) == pytest.approx(4.845)


def test_partial_failure_cleans_only_created_pod(prepared):
    session, client, _, plan = prepared
    client.pods["old-12345"] = {"id": "old-12345", "name": "existing-other-user-work"}
    client.fail_create = 2
    with pytest.raises(APIError):
        session.create(plan)
    assert client.deletes == ["pod-00001"]
    assert "old-12345" in client.pods
    assert session.read()["state"] == "closed"


def test_uncertain_create_reconciles_without_retrying_post(prepared):
    session, client, _, plan = prepared
    client.uncertain_create = 1
    assert session.create(plan)["state"] == "active"
    assert client.creates == 2
    assert [r["id"] for r in session.read()["pods"]] == ["pod-00001", "pod-00002"]
    assert session.cleanup()["state"] == "closed"


@pytest.mark.parametrize("uncertain", [None, 1])
def test_actual_cost_guard_covers_confirmed_and_reconciled_creates(prepared, uncertain):
    session, client, _, plan = prepared
    client.actual_cost = 10
    client.uncertain_create = uncertain
    with pytest.raises(SessionError, match="cost exceeds"):
        session.create(plan)
    assert client.creates == 1
    assert client.deletes == ["pod-00001"]
    assert session.read()["state"] == "closed"


def test_deadline_reached_before_first_create_never_posts(prepared):
    session, client, clock, plan = prepared
    clock.value += 5400
    session.heartbeat()
    with pytest.raises(SessionError, match="deadline"):
        session.create(plan)
    assert client.creates == 0


def test_existing_session_name_is_never_adopted_or_deleted(prepared):
    session, client, _, plan = prepared
    client.pods["old-12345"] = {"id": "old-12345", "name": pod_name(SESSION_ID, "source")}
    with pytest.raises(SessionError, match="already exists"):
        session.create(plan)
    assert client.creates == 0
    assert session.cleanup()["state"] == "closed"
    assert client.deletes == []


def test_timeout_without_visible_pod_remains_pending_and_never_reposts(prepared):
    session, client, _, plan = prepared
    def timeout(payload):
        client.creates += 1
        raise UncertainRequest("uncertain")
    client.create_pod = timeout
    with pytest.raises(SessionError, match="uncertain"):
        session.create(plan)
    assert session.read()["state"] == "needs_cleanup"
    session.watchdog_tick()
    assert client.creates == 1


def test_cleanup_requested_during_create_is_not_overwritten_active(prepared):
    session, client, _, plan = prepared
    original = client.create_pod
    def trigger_cleanup(payload):
        pod = original(payload)
        session.mutate(lambda j: j.update(cleanup_requested=True))
        return pod
    client.create_pod = trigger_cleanup
    with pytest.raises(SessionError, match="cleanup"):
        session.create(plan)
    assert session.read()["state"] == "closed"
    assert client.deletes == ["pod-00001"]


def test_watchdog_disk_failure_uses_snapshot_and_recovers(prepared, monkeypatch, capsys):
    session, client, clock, plan = prepared
    session.create(plan)
    clock.value += 5401
    original = session._save
    failures = [True]
    def failing_save(journal):
        if failures:
            failures.pop()
            raise OSError("private filesystem path must not be logged")
        return original(journal)
    monkeypatch.setattr(session, "_save", failing_save)
    assert session.watchdog(poll_seconds=1)["state"] == "closed"
    assert set(client.deletes) == {"pod-00001", "pod-00002"}
    error = capsys.readouterr().err
    assert "Journal filesystem failure" in error
    assert "private filesystem" not in error


def test_emergency_cleanup_still_refuses_foreign_resources(prepared):
    session, client, _, plan = prepared
    session.create(plan)
    snapshot = session.read()
    client.pods["pod-00001"]["name"] = "unrelated-work"
    with pytest.raises(SessionError, match="ownership"):
        session.emergency_cleanup(snapshot)
    assert client.deletes == []


def test_expired_watchdog_deletes_and_verifies_both(prepared):
    session, client, clock, plan = prepared
    session.create(plan)
    clock.value += 5401
    assert session.watchdog_tick()["state"] == "closed"
    assert set(client.deletes) == {"pod-00001", "pod-00002"}
    assert client.pods == {}


def test_watchdog_is_required_before_create(prepared):
    session, client, clock, plan = prepared
    clock.value += 31
    with pytest.raises(SessionError, match="watchdog"):
        session.create(plan)
    assert client.creates == 0


@pytest.mark.parametrize("change", [
    lambda j: j["pods"][0].update(id="../../account"),
    lambda j: j["pods"][0].update(owner_session="another-session"),
    lambda j: j["pods"][0].update(name="unrelated-existing-pod"),
    lambda j: j.update(session_id="another-session"),
    lambda j: j["baseline_ids"].append(j["pods"][0]["id"]),
])
def test_unsafe_journal_cannot_trigger_delete(prepared, change):
    session, client, _, plan = prepared
    session.create(plan)
    journal = session.read()
    change(journal)
    session.path.write_text(json.dumps(journal))
    with pytest.raises(SessionError):
        session.cleanup()
    assert client.deletes == []


def test_provider_owner_mismatch_never_deletes_that_resource(prepared):
    session, client, _, plan = prepared
    session.create(plan)
    client.pods["pod-00001"]["name"] = "other-work"
    assert session.cleanup()["state"] == "needs_cleanup"
    assert client.deletes == ["pod-00002"]


def test_unconfirmed_termination_stays_pending(prepared):
    session, client, _, plan = prepared
    session.create(plan)
    client.keep_after_delete = True
    assert session.cleanup()["state"] == "needs_cleanup"
    assert all(r["state"] == "created" for r in session.read()["pods"])


def test_changed_quote_rejected_before_post(prepared):
    session, client, _, plan = prepared
    client.reject_quote = True
    with pytest.raises(SessionError, match="quote changed"):
        session.create(plan)
    assert client.creates == 0


def test_modified_plan_rejected_before_post(prepared):
    session, client, _, plan = prepared
    plan["pods"][0]["payload"]["image"] = "changed/image:tag"
    with pytest.raises(SessionError, match="Plan changed"):
        session.create(plan)
    assert client.creates == 0


def test_creation_cannot_be_reinvoked(prepared):
    session, client, _, plan = prepared
    session.create(plan)
    with pytest.raises(SessionError, match="only once"):
        session.create(plan)
    assert client.creates == 2


class Response(io.BytesIO):
    def __init__(self, raw=b"", status=200):
        super().__init__(raw)
        self.status = status


def test_rest_204_empty_body_and_custom_entrypoint():
    seen = []
    def opener(request, timeout):
        seen.append(request)
        return Response(status=204)
    client = RunpodClient("private-test-key", opener=opener)
    assert client.terminate_pod("pod-12345") is None
    assert seen[0].method == "DELETE"
    assert seen[0].full_url == "https://api.runpod.io/v2/pods/pod-12345"
    client.create_pod({"entrypoint": ["bash", "-lc"], "cmd": ["exec sshd -D"]})
    assert json.loads(seen[1].data)["entrypoint"] == ["bash", "-lc"]


def test_http_errors_omit_provider_body_and_credentials():
    def opener(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 400, "unsafe message", {},
                                     io.BytesIO(b'{"env":{"SECRET":"unsafe"}}'))
    client = RunpodClient("private-test-key", opener=opener)
    with pytest.raises(APIError) as caught:
        client.create_pod({"name": "test"})
    assert str(caught.value) == "Runpod API returned HTTP 400"


@pytest.mark.parametrize("availability,price,center,passes", [
    ("LOW", 0.74, "EU-RO-1", True),
    ("NONE", 0.74, "EU-RO-1", False),
    ("LOW", 0.75, "EU-RO-1", False),
    ("LOW", 0.74, "OTHER", False),
])
def test_live_quote_price_and_stock(availability, price, center, passes):
    payload = {"gpus": [{"id": "NVIDIA GeForce RTX 4090", "price": {"secure": price},
                          "availability": availability,
                          "dataCenters": [{"id": center, "availability": availability}]}]}
    client = RunpodClient("private-test-key", opener=lambda *a, **k: Response(json.dumps(payload).encode()))
    record = {"gpu_id": "NVIDIA GeForce RTX 4090", "gpu_count": 1, "cloud": "SECURE",
              "hourly_gpu_usd": 0.74, "data_center_ids": ["EU-RO-1"]}
    if passes:
        client.check_quote(record)
    else:
        with pytest.raises(SessionError):
            client.check_quote(record)


class ProbeClient(FakeClient):
    """Exercise real quote validation against an entirely in-memory catalog."""

    def __init__(self, clock, *, target_price=2.39, source_stock="LOW"):
        super().__init__(clock)
        catalog = {"gpus": [
            {"id": "AMD Instinct MI300X OAM", "secure": True,
             "price": {"secure": target_price}, "availability": "NONE", "dataCenters": []},
            {"id": "NVIDIA GeForce RTX 4090", "secure": True,
             "price": {"secure": 0.74}, "availability": source_stock,
             "dataCenters": [{"id": "EU-RO-1", "availability": source_stock}]},
        ]}
        self.catalog_client = RunpodClient(
            "private-test-key", opener=lambda *a, **k: Response(json.dumps(catalog).encode())
        )

    def check_quote(self, record, *, allow_unavailable_target_probe=False):
        return self.catalog_client.check_quote(
            record, allow_unavailable_target_probe=allow_unavailable_target_probe
        )


def probe_plan():
    plan = make_plan()
    plan["pods"].reverse()  # Probe AMD before any NVIDIA rental is attempted.
    plan["pods"][0]["available"] = False
    return plan


@pytest.mark.parametrize("flag,unavailable_role", [(False, "target"), (True, "source")])
def test_probe_permission_does_not_relax_normal_or_source_preparation(tmp_path, flag, unavailable_role):
    clock = Clock()
    client = ProbeClient(clock)
    session = Session(tmp_path, SESSION_ID, client, clock=clock)
    plan = make_plan()
    next(item for item in plan["pods"] if item["role"] == unavailable_role)["available"] = False
    with pytest.raises(SessionError, match="unavailable"):
        session.prepare(plan, allow_unavailable_target_probe=flag)
    assert client.creates == 0


def test_explicit_unavailable_probe_posts_once_and_rejection_closes_cleanly(tmp_path):
    clock = Clock()
    client = ProbeClient(clock)
    client.fail_create = 1
    session = Session(tmp_path, SESSION_ID, client, clock=clock, sleep=lambda _: None)
    plan = probe_plan()
    session.prepare(plan, allow_unavailable_target_probe=True)
    session.heartbeat()
    with pytest.raises(APIError):
        session.create(plan)
    journal = session.read()
    assert client.creates == 1 and client.deletes == []
    assert journal["state"] == "closed"
    target = journal["pods"][0]
    assert target["role"] == "target" and target["state"] == "rejected"
    assert target["quoted_available"] is False
    assert target["last_catalog_check"]["availability"] == "NONE"
    assert target["last_catalog_check"]["availability_probe_permitted"] is True
    assert journal["pods"][1]["state"] == "planned"


def test_unavailable_target_probe_still_rejects_changed_price(tmp_path):
    clock = Clock()
    client = ProbeClient(clock, target_price=2.40)
    session = Session(tmp_path, SESSION_ID, client, clock=clock)
    plan = probe_plan()
    session.prepare(plan, allow_unavailable_target_probe=True)
    session.heartbeat()
    with pytest.raises(SessionError, match="quote changed"):
        session.create(plan)
    assert client.creates == 0
    assert session.read()["state"] == "closed"


def test_target_probe_never_bypasses_source_live_availability(tmp_path):
    clock = Clock()
    client = ProbeClient(clock, source_stock="NONE")
    session = Session(tmp_path, SESSION_ID, client, clock=clock, sleep=lambda _: None)
    plan = probe_plan()
    session.prepare(plan, allow_unavailable_target_probe=True)
    session.heartbeat()
    with pytest.raises(SessionError, match="no longer available"):
        session.create(plan)
    assert client.creates == 1  # AMD only; unavailable NVIDIA never gets a POST.
    assert client.deletes == ["pod-00001"]
    assert session.read()["state"] == "closed"


def test_source_cannot_use_target_probe_even_via_direct_client():
    record = {"role": "source", "gpu_id": "NVIDIA GeForce RTX 4090", "gpu_count": 1,
              "cloud": "SECURE", "data_center_ids": ["EU-RO-1"], "hourly_gpu_usd": .74}
    with pytest.raises(SessionError, match="restricted"):
        ProbeClient(Clock()).check_quote(record, allow_unavailable_target_probe=True)
