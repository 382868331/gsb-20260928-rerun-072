"""Switch eligibility must account for persistent state, the steady-state
footprint, and the real migration peak. These tests trace a rejection, a
release-then-retry, migration failure recovery, and an interrupted
migration that is saved, reopened, cancelled/resumed with shared reuse."""

import pytest

from precswitch import (
    Budget, Engine, GroupingRule, InputEvent, Mode, Outcome,
    PrecisionScheme, StateKind, SwitchPolicy,
)


def build_schemes(b_migrator=None):
    # a and b are mutually compatible; the migrator is injected so tests can
    # simulate a broken migration.
    a = PrecisionScheme("a", StateKind.COMPATIBLE, load_cost=20, state_cost=10)
    b = PrecisionScheme(
        "b", StateKind.COMPATIBLE, load_cost=10, state_cost=100,
        migrators={"a": lambda st: {"scheme": "a", "history": list(st["history"])}},
    )
    a.migrators["b"] = b_migrator or (lambda st: {"scheme": "b", "history": list(st["history"])})
    c = PrecisionScheme("c", StateKind.COMPATIBLE, load_cost=5, state_cost=10)
    return a, b, c


def engine(rule, group_schemes, budget, **kw):
    return Engine(
        Mode.STATEFUL, rule, SwitchPolicy(min_hold=1, version="p1"),
        group_schemes, budget=budget, **kw,
    )


@pytest.fixture
def rule():
    return GroupingRule(lambda p: "hot" if p < 0 else "cold", version="v1")


def test_persistent_state_counts_in_switch_eligibility(rule):
    # budget 50; shared load 30 is already occupied (a + b); target b's
    # persistent state is 100. The completed switch would total 130 and the
    # migration peak 140: either figure alone proves the denial.
    a, b, _ = build_schemes()
    budget = Budget(limit=50)
    group_schemes = {"cold": a, "hot": b}
    from precswitch.engine import _Session
    budget._retain(a)
    budget._retain(b)  # another candidate already has b partially loaded
    budget.session_started("s", a)
    sess = _Session("s", a, "cold", state=a.fresh_state(), inputs_since_switch=1,
                    outputs=[0])
    eng = engine(rule, group_schemes, budget)
    eng.sessions["s"] = sess

    rep_before = budget.report()
    assert rep_before.total == 40  # 30 shared + 10 persistent state

    projection = budget.project_switch("s", b)
    assert projection.steady_total == 130
    assert projection.peak_total == 140
    assert not projection.fits

    r = eng.feed(InputEvent("s", 1, -1))
    assert r.outcome is Outcome.ROLLED_BACK
    assert "130" in r.detail and "50" in r.detail
    # old state retained, still usable, nothing half-switched
    assert eng.sessions["s"].scheme.name == "a"
    assert eng.sessions["s"].state["scheme"] == "a"
    assert budget.report().total == 40


def test_release_then_retry_succeeds(rule):
    a, b, _ = build_schemes()
    budget = Budget(limit=240)
    group_schemes = {"cold": a, "hot": b}
    eng = engine(rule, group_schemes, budget)

    # A neighbour session keeps b loaded: 10 shared. s on a: 20 shared + 10 state.
    eng.feed(InputEvent("s", 0, 1))
    budget.session_started("neighbour", b)
    assert budget.report().total == 20 + 10 + 10 + 100

    # Switch s to b: peak 240 fits exactly (b load already shared).
    r = eng.feed(InputEvent("s", 1, -1))
    assert r.outcome is Outcome.SWITCHED_CONTINUED
    # Steady state: a released (s was its only user), b shared, two b states.
    assert budget.report().total == 10 + 200


def test_release_frees_shared_load_for_retry(rule):
    a, b, c = build_schemes()
    budget = Budget(limit=140)
    group_schemes = {"cold": a, "hot": b}
    eng = engine(rule, group_schemes, budget)
    eng.feed(InputEvent("s", 0, 1))
    budget.session_started("holder", a)  # extra pin keeps a loaded
    # With both holders active the migration peak is 155 > 140: denied even
    # though, once the unrelated holders are released, the same switch fits.
    budget.session_started("c-holder", c)
    r = eng.feed(InputEvent("s", 1, -1))
    assert r.outcome is Outcome.ROLLED_BACK
    assert eng.sessions["s"].scheme.name == "a"

    # Release unrelated holder; retry with a fresh qualifying input.
    budget.session_ended("c-holder")
    budget.session_ended("holder")
    eng.sessions["s"].inputs_since_switch = 1
    r2 = eng.feed(InputEvent("s", 2, -2))
    assert r2.outcome is Outcome.SWITCHED_CONTINUED
    assert eng.sessions["s"].scheme.name == "b"
    # steady: a unloaded entirely, b loaded, b state only
    assert budget.report().total == 110


def test_migration_failure_keeps_old_scheme(rule):
    def broken(_state):
        raise RuntimeError("checksum mismatch")

    a, b, _ = build_schemes(b_migrator=broken)
    budget = Budget(limit=500)
    group_schemes = {"cold": a, "hot": b}
    eng = engine(rule, group_schemes, budget)
    eng.feed(InputEvent("s", 0, 1))
    shared_after_start = budget.report().shared_scheme_cost

    r = eng.feed(InputEvent("s", 1, -1))
    assert r.outcome is Outcome.SWITCH_FAILED
    assert "checksum mismatch" in r.detail
    # reservation released, old state and scheme fully intact and usable
    assert eng.sessions["s"].scheme.name == "a"
    assert eng.sessions["s"].state == {"scheme": "a", "history": []}
    assert budget.report().shared_scheme_cost == shared_after_start
    r2 = eng.feed(InputEvent("s", 2, 2))
    assert r2.outcome is Outcome.EMITTED and r2.scheme == "a"


def test_interrupted_migration_save_reopen_cancel_and_resume(tmp_path, rule):
    from precswitch import load_artifact, save_artifact

    a, b, c = build_schemes()
    budget = Budget(limit=300)
    group_schemes = {"cold": a, "hot": b}
    eng = engine(rule, group_schemes, budget, interrupt_before_commit=True)
    eng.feed(InputEvent("s", 0, 1))
    # Another engine session already runs b: a candidate reusing that shared
    # load must not be charged b's load again at the peak.
    eng.feed(InputEvent("b-user", 0, -9))

    r = eng.feed(InputEvent("s", 1, -1))
    assert r.outcome is Outcome.SWITCHED_RESTARTED
    sess = eng.sessions["s"]
    assert sess.pending_switch
    # migration peak: a(20) + b(10) shared; states a(10) + b(100) + b(100)
    assert budget.report().total == 240
    # The historical, already-completed switch is not reinterpreted.
    path = str(tmp_path / "mid.json")
    save_artifact(eng, path)

    eng2 = load_artifact(path, rule, group_schemes)
    assert eng2.sessions["s"].pending_target == "b"
    assert eng2.budget.report().total == 240  # real retained resources

    # While the peak is held, an even bigger candidate cannot sneak in.
    big = PrecisionScheme("big", StateKind.COMPATIBLE, load_cost=5, state_cost=90)
    assert not eng2.budget.project_switch("b-user", big).fits

    # Cancel the candidate: only the old scheme/state remain, stream continues.
    eng2.cancel_pending_switch("s")
    assert eng2.sessions["s"].scheme.name == "a"
    assert eng2.sessions["s"].state["scheme"] == "a"
    assert eng2.budget.report().total == 20 + 10 + 10 + 100
    # the cancellation is its own fact, not a completed switch
    assert eng2.history[-1].outcome is Outcome.SWITCH_CANCELLED
    assert eng2.history[-1].scheme == "a"
    assert not any(
        h.outcome is Outcome.SWITCH_COMMITTED and h.session_id == "s"
        for h in eng2.history
    )
    r2 = eng2.feed(InputEvent("s", 2, -2))
    assert r2.outcome is Outcome.SWITCHED_CONTINUED and r2.scheme == "b"

    # Reopen the original mid-flight artifact again and this time resume it.
    eng3 = load_artifact(path, rule, group_schemes)
    eng3.resume_pending_switch("s")
    assert eng3.sessions["s"].scheme.name == "b"
    assert eng3.sessions["s"].state["scheme"] == "b"
    # steady: a released (s was its last user), b shared by two sessions
    assert eng3.budget.report().total == 10 + 200
    # reservation, migration in flight and confirmation are separate facts
    assert eng3.history[2].outcome is Outcome.SWITCH_RESERVED
    assert eng3.history[3].outcome is Outcome.SWITCHED_RESTARTED
    assert eng3.history[-1].outcome is Outcome.SWITCH_COMMITTED
    assert eng3.history[-1].scheme == "b"
    # repeated resume/cancel keep the same candidate identity: no new history
    depth = len(eng3.history)
    eng3.resume_pending_switch("s")
    eng3.cancel_pending_switch("s")
    assert len(eng3.history) == depth
    assert eng3.sessions["s"].scheme.name == "b"


def test_steady_fits_but_peak_overflows_names_only_the_peak(rule):
    # steady footprint inside the budget must not be reported as over.
    a = PrecisionScheme("a", StateKind.COMPATIBLE, load_cost=20, state_cost=10)
    d = PrecisionScheme("d", StateKind.COMPATIBLE, load_cost=5, state_cost=60)
    a.migrators["d"] = lambda st: {"scheme": "d", "history": list(st["history"])}
    d.migrators["a"] = lambda st: {"scheme": "a", "history": list(st["history"])}
    budget = Budget(limit=140)
    group_schemes = {"cold": a, "hot": d}
    eng = engine(rule, group_schemes, budget)
    eng.feed(InputEvent("s", 0, 1))
    budget.session_started("neighbour", d)  # d's shared load already paid
    assert budget.report().total == 20 + 5 + 10 + 60

    projection = budget.project_switch("s", d)
    assert projection.steady_total == 125 <= 140  # steady fits
    assert projection.peak_total == 155 > 140     # only the peak blocks

    r = eng.feed(InputEvent("s", 1, -1))
    assert r.outcome is Outcome.ROLLED_BACK
    assert "migration peak 155" in r.detail
    assert "steady footprint 125 itself fits" in r.detail
    assert "steady footprint 125 exceed" not in r.detail
    assert eng.sessions["s"].scheme.name == "a"


def test_shared_target_load_commit_and_cancel_after_reopen(tmp_path, rule):
    # Two sessions reserve switches into the same target scheme (shared load
    # paid once); after save/reopen one candidate is confirmed, the other
    # cancelled. Ledger and history must tell the same story.
    from precswitch import load_artifact, save_artifact

    a, b, _ = build_schemes()
    budget = Budget(limit=400)
    group_schemes = {"cold": a, "hot": b}
    eng = engine(rule, group_schemes, budget, interrupt_before_commit=True)
    eng.feed(InputEvent("s1", 0, 1))
    eng.feed(InputEvent("s2", 0, 2))
    r1 = eng.feed(InputEvent("s1", 1, -1))
    r2 = eng.feed(InputEvent("s2", 1, -2))
    assert r1.outcome is Outcome.SWITCHED_RESTARTED
    assert r2.outcome is Outcome.SWITCHED_RESTARTED
    # both candidates pending: shared a(20)+b(10); states a+a+b(100)+b(100)
    assert budget.report().total == 30 + 220
    path = str(tmp_path / "mid.json")
    save_artifact(eng, path)

    eng2 = load_artifact(path, rule, group_schemes)
    assert eng2.budget.report().total == 250  # real retained peak rebuilt
    eng2.resume_pending_switch("s1")
    eng2.cancel_pending_switch("s2")
    # s1 committed to b (b state 100, b shared 10); s2 back on a (shared 20,
    # state 10); the cancelled candidate's reservation is fully released
    assert eng2.sessions["s1"].scheme.name == "b"
    assert eng2.sessions["s2"].scheme.name == "a"
    assert eng2.sessions["s2"].state["scheme"] == "a"
    assert eng2.budget.report().total == 10 + 20 + 100 + 10

    # history: each candidate's ending recorded exactly once, no rewrite
    committed = [h for h in eng2.history if h.outcome is Outcome.SWITCH_COMMITTED]
    cancelled = [h for h in eng2.history if h.outcome is Outcome.SWITCH_CANCELLED]
    assert [h.session_id for h in committed] == ["s1"]
    assert [h.session_id for h in cancelled] == ["s2"]
    assert committed[0].scheme == "b" and cancelled[0].scheme == "a"
    reserved = [h for h in eng2.history if h.outcome is Outcome.SWITCH_RESERVED]
    assert sorted(h.session_id for h in reserved) == ["s1", "s2"]

    # repeated complete/cancel keep candidate identity: history unchanged
    depth = len(eng2.history)
    eng2.resume_pending_switch("s1")
    eng2.cancel_pending_switch("s1")  # must not rewrite s1's committed ending
    eng2.cancel_pending_switch("s2")
    eng2.resume_pending_switch("s2")  # must not rewrite s2's cancelled ending
    assert len(eng2.history) == depth
    assert eng2.sessions["s1"].scheme.name == "b"
    assert eng2.sessions["s2"].scheme.name == "a"
    assert eng2.budget.report().total == 140
