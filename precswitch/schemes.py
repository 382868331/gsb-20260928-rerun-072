from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Callable, Dict


class StateKind(enum.Enum):
    """How a scheme's internal session state relates to other schemes."""

    STATELESS = "stateless"          # no internal state; freely interchangeable
    COMPATIBLE = "compatible"        # state migratable via a migrator function
    INCOMPATIBLE = "incompatible"    # state cannot be carried across schemes


@dataclass(frozen=True)
class PrecisionScheme:
    """A precision scheme that may hold per-session internal state.

    state_cost is the per-active-session memory footprint of its state;
    load_cost is the shared cost of keeping the scheme itself loaded
    (counted once no matter how many sessions use it).
    """

    name: str
    kind: StateKind
    load_cost: int = 0
    state_cost: int = 0
    # migrators[target_scheme_name] maps this scheme's state into the target's.
    migrators: Dict[str, Callable[[Any], Any]] = field(default_factory=dict, compare=False)

    def fresh_state(self) -> Any:
        if self.kind is StateKind.STATELESS:
            return None
        return {"scheme": self.name, "history": []}

    def can_migrate_to(self, target: "PrecisionScheme") -> bool:
        if target.name == self.name:
            return True
        if self.kind is StateKind.STATELESS or target.kind is StateKind.STATELESS:
            return True
        if self.kind is StateKind.COMPATIBLE and target.name in self.migrators:
            return True
        return False

    def migrate_state_to(self, target: "PrecisionScheme", state: Any) -> Any:
        if not self.can_migrate_to(target):
            raise ValueError("state of %s cannot migrate to %s" % (self.name, target.name))
        if target.name == self.name:
            return state
        if self.kind is StateKind.STATELESS:
            return target.fresh_state()
        if target.kind is StateKind.STATELESS:
            return None
        return self.migrators[target.name](state)

