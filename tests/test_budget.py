from precswitch import Budget, InputEvent, Mode, Outcome

from conftest import make_engine


def test_budget_counts_shared_and_active_sessions(schemes):
    budget = Budget(limit=1000)
    budget.session_started("a", schemes["mid"])
    budget.session_started("b", schemes["mid"])
    budget.session_started("c", schemes["high"])
    rep = budget.report()
    # shared load counted once per scheme, state counted per active session
    assert rep.shared_scheme_cost == 20 + 40
    assert rep.session_state_cost == 5 + 5 + 15
    assert rep.total == 85


def test_idle_session_still_counts(schemes):
    budget = Budget(limit=100)
    budget.session_started("idle", schemes["high"])
    rep = budget.report()
    assert rep.session_state_cost == 15  # no new input, still counted
    budget.session_ended("idle")
    assert budget.report().session_state_cost == 0


def test_budget_denied_on_short_stream(rule, group_schemes, schemes):
    # enough for mid (20+5) but not high (40+15)
    budget = Budget(limit=30)
    eng = make_engine(Mode.STATEFUL, rule, group_schemes, min_hold=1, budget=budget)
    r0 = eng.feed(InputEvent("s", 0, 1))
    assert r0.outcome is Outcome.EMITTED
    r1 = eng.feed(InputEvent("s", 1, -1))
    assert r1.outcome is Outcome.ROLLED_BACK
    assert "budget" in r1.detail
    # session still on old scheme, state intact
    assert eng.sessions["s"].scheme.name == "mid"


def test_budget_denied_at_session_start(rule, group_schemes, schemes):
    budget = Budget(limit=10)
    eng = make_engine(Mode.STATEFUL, rule, group_schemes, budget=budget)
    r = eng.feed(InputEvent("s", 0, 1))
    assert r.outcome is Outcome.BUDGET_DENIED

