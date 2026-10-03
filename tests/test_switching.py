from precswitch import Engine, GroupingRule, InputEvent, Mode, Outcome, SwitchPolicy

from conftest import make_engine


def feed_stream(engine, sid, payloads):
    return [engine.feed(InputEvent(sid, i, p)) for i, p in enumerate(payloads)]


def test_stateful_switch_normal_to_high_risk(rule, group_schemes, schemes):
    eng = make_engine(Mode.STATEFUL, rule, group_schemes, min_hold=1)
    res = feed_stream(eng, "s1", [1, 2, -5, -6])
    assert [r.outcome for r in res] == [
        Outcome.EMITTED, Outcome.EMITTED,
        Outcome.SWITCHED_CONTINUED, Outcome.EMITTED,
    ]
    assert res[2].scheme == "high"
    # prior outputs remain attributed to the old scheme
    assert res[2].preview.old_scheme_outputs == [0, 1]


def test_min_hold_blocks_ping_pong(rule, group_schemes, schemes):
    eng = make_engine(Mode.STATEFUL, rule, group_schemes, min_hold=10)
    res = feed_stream(eng, "s1", [1, -1, 1, -1, 1, -1])
    # group keeps changing but min_hold=10 prevents every switch
    assert all(r.outcome in (Outcome.EMITTED, Outcome.HELD_BY_MIN_HOLD) for r in res)
    assert sum(r.outcome is Outcome.HELD_BY_MIN_HOLD for r in res) == 3
    assert eng.sessions["s1"].scheme.name == "mid"


def test_ping_pong_with_short_hold_thrashes(rule, group_schemes, schemes):
    # min_hold=1 lets every cross-group flip attempt a switch; when the
    # target state cannot migrate the rollback is explicit, not silent.
    eng = make_engine(Mode.STATEFUL, rule, group_schemes, min_hold=1)
    res = feed_stream(eng, "s1", [1, -1, 1, -1])
    assert [r.outcome for r in res] == [
        Outcome.EMITTED, Outcome.SWITCHED_CONTINUED,
        Outcome.ROLLED_BACK, Outcome.EMITTED,
    ]
    # high -> mid cannot migrate (high is incompatible): explicit rollback
    assert res[2].scheme == "high"


def test_min_hold_allows_switch_after_hold(rule, group_schemes, schemes):
    eng = make_engine(Mode.STATEFUL, rule, group_schemes, min_hold=2)
    res = feed_stream(eng, "s1", [1, 1, -1, -1, -1])
    # after 2 inputs under mid, min_hold=2 is met: switch allowed at pos 2
    assert res[2].outcome is Outcome.SWITCHED_CONTINUED
    eng2 = make_engine(Mode.STATEFUL, rule, group_schemes, min_hold=3)
    res2 = feed_stream(eng2, "s1", [1, 1, -1, -1, -1])
    assert res2[2].outcome is Outcome.HELD_BY_MIN_HOLD
    assert res2[3].outcome is Outcome.SWITCHED_CONTINUED


def test_uniform_mode_ignores_groups(rule, group_schemes, schemes):
    eng = make_engine(Mode.UNIFORM, rule, group_schemes, schemes=schemes)
    res = feed_stream(eng, "s1", [1, -1, 1, -1])
    assert all(r.scheme == "low" for r in res)
    assert all(r.outcome is Outcome.EMITTED for r in res)


def test_independent_mode_picks_per_input(rule, group_schemes, schemes):
    eng = make_engine(Mode.INDEPENDENT, rule, group_schemes)
    res = feed_stream(eng, "s1", [1, -1, 1, -1])
    assert [r.scheme for r in res] == ["mid", "high", "mid", "high"]
    assert "s1" not in eng.sessions  # no session state retained


def test_uncovered_input_explicit(rule, group_schemes, schemes):
    eng = make_engine(Mode.STATEFUL, rule, group_schemes)
    res = feed_stream(eng, "s1", [1, "?", 2])
    assert res[1].outcome is Outcome.UNCOVERED_INPUT
    assert res[1].scheme is None
    assert res[2].outcome is Outcome.EMITTED  # stream continues


def test_causal_decisions_only(rule, group_schemes, schemes):
    # identical prefixes must produce identical decisions regardless of what
    # comes later: switching cannot depend on future inputs/errors.
    eng1 = make_engine(Mode.STATEFUL, rule, group_schemes, min_hold=1)
    eng2 = make_engine(Mode.STATEFUL, rule, group_schemes, min_hold=1)
    r1 = feed_stream(eng1, "s", [1, -1, 1])
    r2 = feed_stream(eng2, "s", [1, -1, -9])
    assert [(r.outcome, r.scheme) for r in r1[:2]] == [(r.outcome, r.scheme) for r in r2[:2]]

