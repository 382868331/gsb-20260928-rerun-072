import pytest

from precswitch import (
    InputEvent, Mode, Outcome, SwitchPreview, UnexplainableStateError,
)
from precswitch.preview import preview_switch

from conftest import make_engine


def test_preview_compatible_migration(schemes):
    prev = preview_switch(schemes["mid"], schemes["high"], schemes["mid"].fresh_state(), 5, [0, 1])
    assert prev.can_continue
    assert prev.restart_at is None
    assert prev.old_scheme_outputs == [0, 1]


def test_preview_incompatible_requires_restart(schemes):
    st = schemes["high"].fresh_state()
    prev = preview_switch(schemes["high"], schemes["locked"], st, 7, [0, 1, 2])
    assert not prev.can_continue
    assert prev.restart_at == 7
    assert prev.old_scheme_outputs == [0, 1, 2]
    assert "restart" in prev.reason


def test_unexplainable_state_raises(schemes):
    with pytest.raises(UnexplainableStateError):
        preview_switch(schemes["mid"], schemes["high"], {"scheme": "ghost"}, 0, [])


def test_stateful_rollback_on_unmigratable_state(rule, schemes):
    # normal -> mid (compatible), high_risk -> locked (incompatible, no migrator)
    eng = make_engine(
        Mode.STATEFUL, rule,
        {"normal": schemes["mid"], "high_risk": schemes["locked"]},
        min_hold=1,
    )
    res = [eng.feed(InputEvent("s", i, p)) for i, p in enumerate([1, 1, -1])]
    # mid -> locked cannot migrate; engine rolls back instead of dropping state
    assert res[2].outcome is Outcome.ROLLED_BACK
    assert res[2].scheme == "mid"
    assert eng.sessions["s"].state["scheme"] == "mid"
    assert res[2].preview is not None and not res[2].preview.can_continue

