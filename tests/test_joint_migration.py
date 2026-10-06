"""Joint multi-session migration scheduling: order feasibility under a hard
capacity limit, shared target reserved once, old loads pinned while still
used, failure/cancel releasing only the failed candidate, sequential vs
batch peak/waiting, and save/reopen with idempotent completion."""

import pytest

from precswitch import (
    Budget, Engine, GroupingRule, InputEvent, Mode, Outcome, PrecisionScheme,
    StateKind, SwitchPolicy, load_artifact, save_artifact,
)
from precswitch.joint import (
    JointMigrationExecutor, MigrationStep, check_joint_batch,
    check_joint_order, compare_strategies, find_feasible_order,
)


def build_schemes():
    a = PrecisionScheme("a", StateKind.COMPATIBLE, load_cost=20, state_cost=10)
    b = PrecisionScheme("b", StateKind.COMPATIBLE, load_cost=10, state_cost=100)
    c = PrecisionScheme("c", StateKind.COMPATIBLE, load_cost=5, state_cost=10)
    e = PrecisionScheme("e", StateKind.COMPATIBLE, load_cost=60, state_cost=10)
    big = PrecisionScheme("big", StateKind.COMPATIBLE, load_cost=80, state_cost=20)
    for src, dst in [(a, b), (a, c), (e, c), (a, big), (b, a), (c, a)]:
        src.migrators[dst.name] = (
            lambda st, d=dst.name: {"scheme": d, "history": list(st["history"])})
    return a, b, c, e, big


@pytest.fixture
def rule():
    return GroupingRule(lambda p: "hot" if p < 0 else "cold", version="v1")


def make_engine(rule, group_schemes, limit):
    return Engine(Mode.STATEFUL, rule, SwitchPolicy(min_hold=1, version="p1"),
                  group_schemes, budget=Budget(limit=limit))


def start_sessions(eng, ids, payload=1):
    for i, sid in enumerate(ids):
        r = eng.feed(InputEvent(sid, i, payload))
        assert r.outcome is Outcome.EMITTED


def test_order_feasibility_and_shared_target_reserved_once(rule):
    a, b, c, _, _ = build_schemes()
    eng = make_engine(rule, {"cold": a, "hot": b}, limit=300)
    start_sessions(eng, ["s1", "s2", "s3"])
    assert eng.budget.report().total == 20 + 30  # a shared once + 3 states

    steps = [MigrationStep("s1", b), MigrationStep("s2", b), MigrationStep("s3", c)]
    plan = check_joint_order(eng.budget, steps)
    assert plan.feasible
    # step 1: old a load still pinned by s2/s3, b reserved, peak 50+110=160
    assert plan.checks[0].projection.peak_total == 160
    assert plan.checks[0].projection.steady_total == 150  # a not released early
    # step 2: b's shared load already counted once -- peak grows by state only
    assert plan.checks[1].projection.peak_total == 250
    # step 3: a released only after its last user (s3) migrates away
    assert plan.checks[2].projection.steady_total == 10 + 5 + 200 + 10
    assert plan.peak_total == 255
    assert plan.waiting == 3
    # planning is a simulation: the real ledger is untouched
    assert eng.budget.report().total == 50
    assert not any(s.pending_switch for s in eng.sessions.values())


def test_order_matters_and_search_finds_feasible_order(rule):
    a, b, c, e, big = build_schemes()
    eng = make_engine(rule, {"cold": a, "hot": b}, limit=150)
    start_sessions(eng, ["s1"])
    eng.budget.session_started("s2", e)  # s2 runs scheme e (60 load)
    from precswitch.engine import _Session
    eng.sessions["s2"] = _Session("s2", e, "cold", state=e.fresh_state())
    assert eng.budget.report().total == 20 + 60 + 20

    steps = [MigrationStep("s1", big), MigrationStep("s2", c)]
    # big first: peak 100 + 80 + 20 = 200 > 150
    assert not check_joint_order(eng.budget, steps).feasible
    # freeing e first makes room: 100+15=115, steady 45, then 45+100=145
    plan = find_feasible_order(eng.budget, steps)
    assert plan is not None and plan.feasible
    assert [s.step.session_id for s in plan.checks] == ["s2", "s1"]
    assert plan.peak_total == 145
    # the search itself executed nothing
    assert eng.budget.report().total == 100


def test_no_feasible_order_reported(rule):
    a, b, c, e, big = build_schemes()
    eng = make_engine(rule, {"cold": a, "hot": b}, limit=140)
    start_sessions(eng, ["s1"])
    eng.budget.session_started("s2", e)
    from precswitch.engine import _Session
    eng.sessions["s2"] = _Session("s2", e, "cold", state=e.fresh_state())

    steps = [MigrationStep("s1", big), MigrationStep("s2", c)]
    # even after freeing e, the big migration peaks at 45+100=145 > 140
    assert find_feasible_order(eng.budget, steps) is None
    assert eng.budget.report().total == 100  # nothing half-done


def test_compare_sequential_vs_batch_peak_and_waiting(rule):
    a, b, c, _, _ = build_schemes()
    eng = make_engine(rule, {"cold": a, "hot": b}, limit=300)
    start_sessions(eng, ["s1", "s2", "s3"])
    steps = [MigrationStep("s1", b), MigrationStep("s2", b), MigrationStep("s3", c)]

    cmp = compare_strategies(eng.budget, steps)
    assert cmp.sequential.feasible and cmp.batch.feasible
    # batch overlaps every conversion peak: 50 + 110 + 100 + 15 = 275
    assert cmp.batch.peak_total == 275
    assert cmp.sequential.peak_total == 255
    assert cmp.batch.waiting == 0
    assert cmp.sequential.waiting == 3
    # a compared-but-unexecuted plan frees nothing
    assert eng.budget.report().total == 50


def test_sequential_execution_failure_releases_only_failed_candidate(rule):
    a, b, c, _, _ = build_schemes()
    a2 = PrecisionScheme("a2", StateKind.COMPATIBLE, load_cost=20, state_cost=10)
    a2.migrators["b"] = lambda st: {"scheme": "b", "history": list(st["history"])}

    def broken(_state):
        raise RuntimeError("checksum mismatch")

    a.migrators["b"] = broken
    eng = make_engine(rule, {"cold": a, "hot": b}, limit=500)
    start_sessions(eng, ["s1"])
    eng.budget.session_started("s2", a2)
    from precswitch.engine import _Session
    eng.sessions["s2"] = _Session("s2", a2, "cold", state=a2.fresh_state())
    before = eng.budget.report().total  # 40 shared + 20 state

    ex = JointMigrationExecutor(eng)
    rep = ex.execute([MigrationStep("s1", b), MigrationStep("s2", b)])
    by_sid = {o.step.session_id: o for o in rep.outcomes}
    assert by_sid["s1"].status == "failed"
    assert "checksum mismatch" in by_sid["s1"].detail
    assert by_sid["s2"].status == "committed"
    # failed candidate's reservation fully released; old session usable
    assert eng.sessions["s1"].scheme.name == "a"
    assert eng.sessions["s1"].state["scheme"] == "a"
    # ledger: a + a2? no -- s2 left a2 (last user): a(20)+b(10) shared,
    # states s1 a-state(10) + s2 b-state(100)
    assert eng.budget.report().total == 20 + 10 + 10 + 100
    r = eng.feed(InputEvent("s1", 5, 3))
    assert r.outcome is Outcome.EMITTED and r.scheme == "a"


def test_denied_step_reports_whether_the_rest_can_still_advance(rule):
    a, b, c, _, _ = build_schemes()
    eng = make_engine(rule, {"cold": a, "hot": b}, limit=160)
    start_sessions(eng, ["s1", "s2", "s3"])
    steps = [MigrationStep("s1", b), MigrationStep("s2", b), MigrationStep("s3", c)]

    rep = JointMigrationExecutor(eng).execute(steps)
    by_sid = {o.step.session_id: o for o in rep.outcomes}
    assert by_sid["s1"].status == "committed"   # peak exactly 160
    assert by_sid["s2"].status == "denied"      # peak 250 > 160
    assert by_sid["s3"].status == "denied"      # peak 165 > 160
    # from the real post-denial ledger: neither remaining step can advance
    assert rep.continuation == (("s2", False), ("s3", False))
    # denied steps touched nothing: s2/s3 still on a, a still shared
    assert eng.sessions["s2"].scheme.name == "a"
    assert eng.budget.report().total == 150


def test_batch_cancel_releases_only_that_candidate_and_is_idempotent(rule):
    a, b, c, _, _ = build_schemes()
    eng = make_engine(rule, {"cold": a, "hot": b}, limit=400)
    start_sessions(eng, ["s1", "s2"])
    steps = [MigrationStep("s1", b), MigrationStep("s2", b)]

    rep = JointMigrationExecutor(eng).execute(steps, batch=True, cancel=["s2"])
    by_sid = {}
    for o in rep.outcomes:
        by_sid.setdefault(o.step.session_id, []).append(o.status)
    assert by_sid["s1"] == ["reserved", "committed"]
    assert by_sid["s2"] == ["reserved", "cancelled"]
    # s1 on b (shared 10, state 100); s2 back on a (shared 20, state 10)
    assert eng.sessions["s1"].scheme.name == "b"
    assert eng.sessions["s2"].scheme.name == "a"
    assert eng.sessions["s2"].state["scheme"] == "a"
    assert eng.budget.report().total == 10 + 20 + 100 + 10
    # batch peak held both b states plus both a states and both loads
    assert rep.peak_total == 30 + 220

    # duplicate completion/cancellation notifications are no-ops
    depth = len(eng.history)
    total = eng.budget.report().total
    eng.resume_pending_switch("s1")
    eng.cancel_pending_switch("s1")
    eng.cancel_pending_switch("s2")
    eng.resume_pending_switch("s2")
    assert len(eng.history) == depth
    assert eng.budget.report().total == total
    assert eng.sessions["s1"].scheme.name == "b"
    assert eng.sessions["s2"].scheme.name == "a"


def test_batch_save_reopen_resume_and_rollback(tmp_path, rule):
    a, b, c, _, _ = build_schemes()
    group_schemes = {"cold": a, "hot": b}
    eng = make_engine(rule, group_schemes, limit=400)
    start_sessions(eng, ["s1", "s2"])
    steps = [MigrationStep("s1", b), MigrationStep("s2", b)]

    # reserve both without committing: interruption mid-batch
    ex = JointMigrationExecutor(eng)
    outcomes = []
    assert ex._reserve(steps[0], outcomes)
    assert ex._reserve(steps[1], outcomes)
    assert eng.budget.report().total == 30 + 220  # real retained peak
    path = str(tmp_path / "mid-batch.json")
    save_artifact(eng, path)

    # Reopen and resume s1, roll back s2.
    eng2 = load_artifact(path, rule, group_schemes)
    assert eng2.budget.report().total == 250  # rebuilt from real resources
    eng2.resume_pending_switch("s1")
    eng2.cancel_pending_switch("s2")
    assert eng2.sessions["s1"].scheme.name == "b"
    assert eng2.sessions["s2"].scheme.name == "a"
    assert eng2.budget.report().total == 140

    # Loading the same artifact again must not double-load or double-release:
    # the second reopen is an independent ledger with the same real peak.
    eng3 = load_artifact(path, rule, group_schemes)
    assert eng3.budget.report().total == 250
    eng3.cancel_pending_switch("s1")
    eng3.cancel_pending_switch("s2")
    assert eng3.budget.report().total == 20 + 20  # both rolled back to a
    assert eng2.budget.report().total == 140      # other ledger untouched

    # duplicate resolution after reopen stays idempotent
    depth = len(eng2.history)
    eng2.resume_pending_switch("s1")
    eng2.cancel_pending_switch("s2")
    assert len(eng2.history) == depth
    assert eng2.budget.report().total == 140
