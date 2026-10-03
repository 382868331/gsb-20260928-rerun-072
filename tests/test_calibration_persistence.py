import pytest

from precswitch import (
    Calibration, GroupingRule, InputEvent, Mode, Outcome, load_artifact, save_artifact,
)
from precswitch.calibration import CalibrationError

from conftest import make_engine


def test_calibration_bound_to_rule_and_policy():
    cal = Calibration(rule_version="v1", policy_version="p1")
    cal.record("stateful", "mid", 0.01, 0.05)
    assert cal.error_range("stateful", "mid", "v1", "p1") == (0.01, 0.05)
    with pytest.raises(CalibrationError):
        cal.error_range("stateful", "mid", "v2", "p1")  # regrouped: not endorsed
    with pytest.raises(CalibrationError):
        cal.error_range("stateful", "mid", "v1", "p2")  # new hold rules: not endorsed


def test_save_reload_continues_session_path(tmp_path, rule, group_schemes, schemes):
    eng = make_engine(Mode.STATEFUL, rule, group_schemes, min_hold=1)
    for i, p in enumerate([1, 2, -1]):
        eng.feed(InputEvent("s1", i, p))
    path = str(tmp_path / "artifact.json")
    save_artifact(eng, path)

    eng2 = load_artifact(path, rule, group_schemes)
    assert eng2.sessions["s1"].scheme.name == "high"
    assert eng2.sessions["s1"].old_scheme_outputs == [0, 1]
    # resume: next input continues the original path
    r = eng2.feed(InputEvent("s1", 3, -2))
    assert r.outcome is Outcome.EMITTED and r.scheme == "high"
    r2 = eng2.feed(InputEvent("s1", 4, 5))
    # high -> mid cannot migrate (high incompatible): explicit rollback
    assert r2.outcome is Outcome.ROLLED_BACK and r2.scheme == "high"


def test_reload_rejects_changed_rule(tmp_path, group_schemes, schemes):
    rule_v1 = GroupingRule(lambda p: "high_risk" if p < 0 else "normal", version="v1")
    eng = make_engine(Mode.STATEFUL, rule_v1, group_schemes)
    eng.feed(InputEvent("s1", 0, 1))
    path = str(tmp_path / "a.json")
    save_artifact(eng, path)
    rule_v2 = GroupingRule(lambda p: "high_risk" if p < 10 else "normal", version="v2")
    with pytest.raises(ValueError):
        load_artifact(path, rule_v2, group_schemes)

