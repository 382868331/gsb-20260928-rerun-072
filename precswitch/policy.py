from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SwitchPolicy:
    """Rule deciding when a session may switch precision scheme.

    min_hold is the minimum number of inputs a session must process
    under its current scheme before switching away is allowed. Decisions
    are causal: they may only use inputs already seen, never future error.
    """

    min_hold: int = 1
    version: str = "p1"

    def may_switch(self, current_group: str, target_group: str, inputs_since_switch: int) -> bool:
        if current_group == target_group:
            return False
        return inputs_since_switch >= self.min_hold

