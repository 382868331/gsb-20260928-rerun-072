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


def test_peak_only_rejection_names_real_blocker(rule):
    # Steady footprint fits, only the migration peak breaches: the message
    # must name the temporary coexistence rather than calling both figures
    # exceeded.
    a, b, _ = build_schemes()
    # a load 20 + state 10 fits; target b state 100: steady = 10+100 = 110;
    # peak adds the new state and b's load on top of the old footprint:
    # 30 + 100 + 10 = 140.
    budget = Budget(limit=120)
    group_schemes = {"cold": a, "hot": b}
    eng = engine(rule, group_schemes, budget)
    eng.feed(InputEvent("s", 0, 1))

    projection = budget.project_switch("s", b)
    assert projection.steady_total == 110  # fits within 120
    assert projection.peak_total == 140    # does not
    assert not projection.fits
    assert not projection.steady_exceeded and projection.peak_exceeded

    r = eng.feed(InputEvent("s", 1, -1))
    assert r.outcome is Outcome.ROLLED_BACK
    detail = r.detail
    assert "steady footprint 110 fits within budget 120" in detail
    assert "migration peak 140" in detail
    assert "temporary coexistence" in detail
    assert "not the steady-state footprint" in detail
    assert "110 and the migration peak 140 exceed" not in detail
    assert budget.report().total == 30  # nothing reserved, old state usable


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
    candidate_id = "s@1->b"
    assert sess.candidate_id == candidate_id
    # Reservation and in-flight migration are separately traceable facts.
    assert [h.outcome for h in eng.history[-2:]] == [
        Outcome.SWITCH_RESERVED, Outcome.SWITCHED_RESTARTED,
    ]
    assert all(h.candidate_id == candidate_id for h in eng.history[-2:])
    # migration peak: a(20) + b(10) shared; states a(10) + b(100) + b(100)
    assert budget.report().total == 240
    # The historical, already-completed switch is not reinterpreted.
    path = str(tmp_path / "mid.json")
    save_artifact(eng, path)

    eng2 = load_artifact(path, rule, group_schemes)
    assert eng2.sessions["s"].pending_target == "b"
    assert eng2.sessions["s"].candidate_id == candidate_id
    assert eng2.budget.report().total == 240  # real retained resources

    # While the peak is held, an even bigger candidate cannot sneak in.
    big = PrecisionScheme("big", StateKind.COMPATIBLE, load_cost=5, state_cost=90)
    assert not eng2.budget.project_switch("b-user", big).fits

    # Cancel the candidate: only the old scheme/state remain, stream continues.
    cancel_result = eng2.cancel_pending_switch("s")
    assert cancel_result.outcome is Outcome.SWITCH_CANCELLED
    assert cancel_result.candidate_id == candidate_id
    assert eng2.sessions["s"].scheme.name == "a"
    assert eng2.sessions["s"].state["scheme"] == "a"
    # A cancellation is its own fact and must never look like completion.
    assert eng2.history[-1].outcome is Outcome.SWITCH_CANCELLED
    assert eng2.history[-1].scheme == "a"
    assert eng2.history[-1].candidate_id == candidate_id
    assert eng2.sessions["s"].last_candidate_id == candidate_id
    # Repeating the cancel does not append another fact and does not switch.
    with pytest.raises(ValueError):
        eng2.cancel_pending_switch("s")
    assert eng2.sessions["s"].scheme.name == "a"
    assert [h.outcome for h in eng2.history].count(Outcome.SWITCH_CANCELLED) == 1
    assert eng2.budget.report().total == 20 + 10 + 10 + 100
    r2 = eng2.feed(InputEvent("s", 2, -2))
    assert r2.outcome is Outcome.SWITCHED_CONTINUED and r2.scheme == "b"
    # A genuinely new attempt gets a distinct candidate identity.
    assert r2.candidate_id == "s@2->b"

    # Reopen the original mid-flight artifact again and this time resume it.
    eng3 = load_artifact(path, rule, group_schemes)
    commit_result = eng3.resume_pending_switch("s")
    assert commit_result.outcome is Outcome.SWITCH_COMMITTED
    assert commit_result.candidate_id == candidate_id
    assert eng3.sessions["s"].scheme.name == "b"
    assert eng3.sessions["s"].state["scheme"] == "b"
    # steady: a released (s was its last user), b shared by two sessions
    assert eng3.budget.report().total == 10 + 200
    # Facts recorded at switch time are preserved, not rewritten on resume;
    # the later commit is appended and attributed to the same candidate.
    assert [h.outcome for h in eng3.history[2:4]] == [
        Outcome.SWITCH_RESERVED, Outcome.SWITCHED_RESTARTED,
    ]
    assert eng3.history[-1].outcome is Outcome.SWITCH_COMMITTED
    assert eng3.history[-1].candidate_id == candidate_id
    with pytest.raises(ValueError):
        eng3.resume_pending_switch("s")


def _history_for(engine, session_id):
    return [h for h in engine.history if h.session_id == session_id]


def test_two_sessions_shared_target_cancel_and_commit_ledger(tmp_path, rule):
    """Two sessions sharing a target load: save both mid-flight, then in
    two reopened engines cancel one candidate and commit the other. The
    resource ledger must match exactly the facts appended to history."""
    from precswitch import load_artifact, save_artifact

    a, b, _ = build_schemes()
    budget = Budget(limit=400)
    group_schemes = {"cold": a, "hot": b}
    eng = engine(rule, group_schemes, budget, interrupt_before_commit=True)
    eng.feed(InputEvent("s1", 0, 1))
    eng.feed(InputEvent("s2", 0, 1))
    # A third session keeps b's shared load independently of the candidates.
    eng.feed(InputEvent("b-user", 0, -9))

    r1 = eng.feed(InputEvent("s1", 1, -1))
    r2 = eng.feed(InputEvent("s2", 1, -1))
    assert r1.outcome is r2.outcome is Outcome.SWITCHED_RESTARTED
    assert eng.sessions["s1"].candidate_id == "s1@1->b"
    assert eng.sessions["s2"].candidate_id == "s2@1->b"
    # Peak held twice, b load shared once:
    # shared a(20)+b(10); states b-user(100)+s1(a10+b100)+s2(a10+b100) = 320
    assert budget.report().total == 350
    path = str(tmp_path / "two.json")
    save_artifact(eng, path)

    # Session A engine: cancel s1's candidate.
    eng_cancel = load_artifact(path, rule, group_schemes)
    assert eng_cancel.budget.report().total == 350
    cr = eng_cancel.cancel_pending_switch("s1")
    assert cr.candidate_id == "s1@1->b"
    # Released: s1's built b state (100). Shared loads unchanged: s1 keeps
    # a, s2 is still pending on both, b-user keeps b. states 220 + 30.
    assert eng_cancel.budget.report().total == 250
    s1_facts = [h.outcome for h in _history_for(eng_cancel, "s1")]
    assert s1_facts == [
        Outcome.EMITTED,
        Outcome.SWITCH_RESERVED,
        Outcome.SWITCHED_RESTARTED,
        Outcome.SWITCH_CANCELLED,
    ]
    assert all(
        h.candidate_id == "s1@1->b"
        for h in _history_for(eng_cancel, "s1")[1:]
    )
    assert eng_cancel.sessions["s1"].scheme.name == "a"
    # s2's candidate is untouched in this engine.
    assert eng_cancel.sessions["s2"].candidate_id == "s2@1->b"

    # Session B engine: commit s2's candidate.
    eng_commit = load_artifact(path, rule, group_schemes)
    rr = eng_commit.resume_pending_switch("s2")
    assert rr.candidate_id == "s2@1->b"
    # s2's a state (10) is released; a load stays because s1 is still
    # pending with old scheme a. states: 100 + (10+100) + 100 = 310 + 30.
    assert eng_commit.budget.report().total == 340
    s2_facts = [h.outcome for h in _history_for(eng_commit, "s2")]
    assert s2_facts == [
        Outcome.EMITTED,
        Outcome.SWITCH_RESERVED,
        Outcome.SWITCHED_RESTARTED,
        Outcome.SWITCH_COMMITTED,
    ]
    assert eng_commit.sessions["s2"].scheme.name == "b"
    assert eng_commit.sessions["s2"].state["scheme"] == "b"
    # No cancellation fact exists for s2 and no commit fact for s1.
    assert Outcome.SWITCH_COMMITTED not in [
        h.outcome for h in _history_for(eng_commit, "s1")]
    assert Outcome.SWITCH_CANCELLED not in [
        h.outcome for h in _history_for(eng_commit, "s2")]
