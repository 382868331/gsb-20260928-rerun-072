import pytest

from precswitch import (
    Budget, Engine, GroupingRule, Mode, PrecisionScheme, StateKind, SwitchPolicy,
)


@pytest.fixture
def schemes():
    low = PrecisionScheme("low", StateKind.STATELESS, load_cost=10, state_cost=0)
    mid = PrecisionScheme(
        "mid", StateKind.COMPATIBLE, load_cost=20, state_cost=5,
        migrators={"high": lambda st: {"scheme": "high", "history": list(st["history"])}},
    )
    high = PrecisionScheme("high", StateKind.INCOMPATIBLE, load_cost=40, state_cost=15)
    locked = PrecisionScheme("locked", StateKind.INCOMPATIBLE, load_cost=40, state_cost=15)
    return {"low": low, "mid": mid, "high": high, "locked": locked}


@pytest.fixture
def rule():
    # payload: int; negative -> high_risk, else normal. "?" is uncovered.
    return GroupingRule(
        lambda p: None if p == "?" else ("high_risk" if p < 0 else "normal"),
        version="v1",
    )


@pytest.fixture
def group_schemes(schemes):
    return {"normal": schemes["mid"], "high_risk": schemes["high"]}


def make_engine(mode, rule, group_schemes, min_hold=1, budget=None, schemes=None):
    return Engine(
        mode=mode,
        rule=rule,
        policy=SwitchPolicy(min_hold=min_hold, version="p1"),
        group_schemes=group_schemes,
        budget=budget,
        uniform_scheme=(schemes["low"] if schemes else None),
    )

