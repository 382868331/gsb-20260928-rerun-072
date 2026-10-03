from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Set

from .schemes import PrecisionScheme


@dataclass(frozen=True)
class BudgetReport:
    total: int
    shared_scheme_cost: int
    session_state_cost: int
    limit: int
    fits: bool


class Budget:
    """Accounts memory as shared loaded schemes plus per-active-session state.

    A session's state counts while the session is active, even when it has
    no new input arriving -- idle does not mean free.
    """

    def __init__(self, limit: int):
        self.limit = limit
        self._loaded: Dict[str, PrecisionScheme] = {}
        self._active_sessions: Dict[str, PrecisionScheme] = {}

    def load_scheme(self, scheme: PrecisionScheme) -> None:
        self._loaded[scheme.name] = scheme

    def session_started(self, session_id: str, scheme: PrecisionScheme) -> None:
        self.load_scheme(scheme)
        self._active_sessions[session_id] = scheme

    def session_switched(self, session_id: str, scheme: PrecisionScheme) -> None:
        self.session_started(session_id, scheme)

    def session_ended(self, session_id: str) -> None:
        self._active_sessions.pop(session_id, None)

    def report(self) -> BudgetReport:
        used_names: Set[str] = set(self._loaded)
        shared = sum(self._loaded[n].load_cost for n in used_names)
        state = sum(s.state_cost for s in self._active_sessions.values())
        total = shared + state
        return BudgetReport(total, shared, state, self.limit, total <= self.limit)

    def can_afford_session(self, scheme: PrecisionScheme) -> bool:
        extra_shared = 0 if scheme.name in self._loaded else scheme.load_cost
        current = self.report()
        return current.total + extra_shared + scheme.state_cost <= self.limit

