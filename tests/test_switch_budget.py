"""Persistent state must enter switch eligibility; migration peaks are
reserved before old state is touched, and every terminal switch state has
a matching resource basis."""

from precswitch import (
    Budget, GroupingRule, InputEvent, Mode, Outcome, PrecisionScheme,
    StateKind, load_artifact, save_artifact,
)

from conftest import make_engine


RULE = GroupingRule(
    lambda p: None if p == "?" else ("risk" if p < 0 else "normal"),
    version="v1",
)


def _schemes():
    old = PrecisionScheme(
        "old", StateKind.COMPATIBLE, load_cost=20, state_cost=5,
        migrators={
            "target": lambda st: {"scheme": "target",
                                   "history": list(st["history"])},
            "boom": lambda st: (_boom()),
        },
    )
    target = PrecisionScheme(
        "target", StateKind.INCOMPATIBLE, load_cost=40, state_cost=15,
    )
    boom = PrecisionScheme(
        "boom", StateKind.COMPATIBLE, load_cost=10, state_cost=5,
    )
    keeper = PrecisionScheme(
        "keeper", StateKind.STATELESS, load_cost=30, state_cost=0,
    )
    return old, target, boom, keeper


def _boom():
    raise RuntimeError("migration blew up")


def _engine(budget, groups):
    return make_engine(Mode.STATEFUL, RULE, groups, min_hold=1, budget=budget)


def test_persistent_state_counts_in_switch_eligibility():
    # budget 50; shared load already totals 30; new persistent state 100.
    old = PrecisionScheme("old", StateKind.STATELESS, load_cost=0, state_cost=0)
    heavy = PrecisionScheme("heavy", StateKind.INCOMPATIBLE,
                            load_cost=0, state_cost=100)
    base = PrecisionScheme("base", StateKind.STATELESS,
                           load_cost=30, state_cost=0)
    budget = Budget(limit=50)
    budget.load_scheme(base)  # other shared usage: total is now 30
    budget.session_started("s", old)
    assert budget.report().total == 30
    quote = budget.quote_switch("s", heavy)
    assert not quote.fits
    assert quote.steady_total == 130
    assert quote.peak_total == 130
    assert "persistent state" in quote.reason

    # Through the engine the same switch is refused; old scheme kept.
    groups = {"normal": old, "risk": heavy}
    eng2 = _engine(Budget(limit=50), groups)
    eng2.budget.load_scheme(base)
    r0 = eng2.feed(InputEvent("s", 0, 1))
    assert r0.outcome is Outcome.EMITTED
    r1 = eng2.feed(InputEvent("s", 1, -1))
    assert r1.outcome is Outcome.ROLLED_BACK
    assert "130 > budget 50" in r1.detail
    assert eng2.sessions["s"].scheme.name == "old"
    rep = eng2.budget.report()
    assert rep.total == 30 and rep.fits  # nothing extra retained


def test_migration_peak_protected_even_when_steady_state_fits():
    # Two sessions share the old scheme; releasing it on switch settles
    # within budget, but the overlap peak does not.
    old = PrecisionScheme(
        "old", StateKind.COMPATIBLE, load_cost=40, state_cost=5,
        migrators={"t": lambda st: {"scheme": "t", "history": []}},
    )
    t = PrecisionScheme("t", StateKind.STATELESS, load_cost=10, state_cost=20)
    budget = Budget(limit=75)
    budget.session_started("s1", old)
    budget.session_started("s2", old)  # shared load counted once
    assert budget.report().total == 50
    quote = budget.quote_switch("s1", t)
    assert quote.steady_fits and not quote.peak_fits
    assert quote.steady_total == 75 and quote.peak_total == 80
    assert "migration peak" in quote.reason


def test_release_shared_load_then_retry_succeeds():
    old, target, _boom_s, keeper = _schemes()
    budget = Budget(limit=100)
    eng = _engine(budget, {"normal": old, "risk": target})
    budget.session_started("keeper-session", keeper)
    r0 = eng.feed(InputEvent("s", 0, 1))
    assert r0.outcome is Outcome.EMITTED
    # shared 20+30, state 5; switch peak needs +40+15 -> 105 > 100
    r1 = eng.feed(InputEvent("s", 1, -1))
    assert r1.outcome is Outcome.ROLLED_BACK
    assert eng.sessions["s"].scheme.name == "old"
    assert budget.report().total == 55
    # release the unrelated shared load and retry: now fits
    budget.session_ended("keeper-session")
    r2 = eng.feed(InputEvent("s", 2, -1))
    assert r2.outcome is Outcome.SWITCHED_CONTINUED
    rep = budget.report()  # old load released: 40 shared + 15 state
    assert rep.shared_scheme_cost == 40
    assert rep.session_state_cost == 15
    assert rep.total == 55 and rep.fits


def test_abort_candidate_releases_only_reservation_and_other_candidate_reuses():
    old, target, _b, _k = _schemes()
    budget = Budget(limit=1000)
    budget.session_started("s1", old)
    res = budget.begin_switch("s1", target)
    # peak is really charged while migration is pending: old+new states,
    # and both shared schemes loaded
    rep = budget.report()
    assert rep.shared_scheme_cost == 20 + 40
    assert rep.session_state_cost == 5 + 15
    # another candidate to the same scheme reuses the already-loaded share
    budget.session_started("s2", old)
    quote2 = budget.quote_switch("s2", target)
    assert quote2.extra_shared == 0
    budget.abort_switch(res)
    rep = budget.report()  # target reservation fully released
    assert rep.shared_scheme_cost == 20
    assert rep.session_state_cost == 5 + 5
    # old scheme still usable after cancellation
    assert budget.can_afford_session(old)


def test_migration_failure_releases_reservation_and_keeps_old_state():
    old, _t, boom, _k = _schemes()
    budget = Budget(limit=1000)
    eng = _engine(budget, {"normal": old, "risk": boom})
    eng.feed(InputEvent("s", 0, 1))
    before = budget.report().total
    r = eng.feed(InputEvent("s", 1, -1))
    assert r.outcome is Outcome.SWITCH_FAILED
    assert "migration blew up" in r.detail
    sess = eng.sessions["s"]
    assert sess.scheme.name == "old"
    assert sess.state["scheme"] == "old"  # old state intact, not dropped
    assert budget.report().total == before  # candidate resources released
    # stream continues on the retained old scheme
    r2 = eng.feed(InputEvent("s", 2, 1))
    assert r2.outcome is Outcome.EMITTED and r2.scheme == "old"


def test_migration_saved_midway_reopens_with_real_retained_resources(
    tmp_path,
):
    old, target, _b, _k = _schemes()
    groups = {"normal": old, "risk": target}
    budget = Budget(limit=1000)
    eng = _engine(budget, groups)
    eng.feed(InputEvent("s", 0, 1))
    reserved = eng.begin_migration("s", "risk", 1)
    assert reserved.outcome is Outcome.SWITCH_RESERVED
    sess = eng.sessions["s"]
    assert sess.state["scheme"] == "old"  # old state still live
    assert sess.pending_state["scheme"] == "target"
    peak = budget.report().total  # both states + both shares: 60+20
    assert peak == 80

    path = str(tmp_path / "mid.json")
    save_artifact(eng, path)
    eng2 = load_artifact(path, RULE, groups)
    # the reopened budget accounts exactly the retained migration peak
    assert eng2.budget.report().total == peak
    assert eng2.sessions["s"].scheme.name == "old"
    assert eng2.sessions["s"].pending_state["scheme"] == "target"
    done = eng2.commit_migration("s", 1)
    assert done.outcome is Outcome.SWITCHED_CONTINUED
    assert eng2.sessions["s"].scheme.name == "target"
    rep = eng2.budget.report()
    assert rep.shared_scheme_cost == 40 and rep.session_state_cost == 15
    # historical reservation is preserved, not reinterpreted
    assert eng2.history[1].outcome is Outcome.SWITCH_RESERVED


def test_migration_saved_midway_can_still_be_aborted(tmp_path):
    old, target, _b, _k = _schemes()
    groups = {"normal": old, "risk": target}
    budget = Budget(limit=1000)
    eng = _engine(budget, groups)
    eng.feed(InputEvent("s", 0, 1))
    eng.begin_migration("s", "risk", 1)
    path = str(tmp_path / "mid2.json")
    save_artifact(eng, path)
    eng2 = load_artifact(path, RULE, groups)
    r = eng2.abort_migration("s", 2)
    assert r.outcome is Outcome.SWITCH_ABORTED
    assert eng2.sessions["s"].scheme.name == "old"
    assert eng2.sessions["s"].pending_state is None
    rep = eng2.budget.report()  # back to the pre-switch accounting
    assert rep.shared_scheme_cost == 20 and rep.session_state_cost == 5
    assert rep.total == 25
