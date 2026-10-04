from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .schemes import PrecisionScheme


@dataclass(frozen=True)
class BudgetReport:
    total: int
    shared_scheme_cost: int
    session_state_cost: int
    limit: int
    fits: bool


@dataclass(frozen=True)
class SwitchProjection:
    """Resource basis of a planned switch.

    steady_total is the footprint after the switch completes; peak_total is
    the migration peak when the old state has not been released yet and the
    new state has already been built. Both must fit.
    """

    steady_total: int
    peak_total: int
    limit: int

    @property
    def fits(self) -> bool:
        return self.steady_total <= self.limit and self.peak_total <= self.limit

    @property
    def blocking_total(self) -> int:
        return max(self.steady_total, self.peak_total)

    @property
    def steady_exceeded(self) -> bool:
        return self.steady_total > self.limit

    @property
    def peak_exceeded(self) -> bool:
        return self.peak_total > self.limit

    def describe_block(self) -> str:
        """Name the resource that actually blocks the switch.

        When the steady footprint fits but the migration peak does not,
        the steady figure must not be reported as exceeded: the blocker is
        the temporary coexistence of the old state (and possibly the old
        shared load) with the already-built new state and target load.
        """
        if not self.steady_exceeded and self.peak_exceeded:
            return (
                "steady footprint %d fits within budget %d, but the "
                "migration peak %d exceeds it; the blocking resource is the "
                "temporary coexistence during migration (old state retained "
                "while the new state is already built, plus the overlapping "
                "shared scheme loads), not the steady-state footprint"
                % (self.steady_total, self.limit, self.peak_total)
            )
        if self.steady_exceeded and not self.peak_exceeded:
            return (
                "migration peak %d fits within budget %d, but the steady "
                "footprint %d after the switch exceeds it; the blocking "
                "resource is the target scheme's persistent steady-state "
                "cost"
                % (self.peak_total, self.limit, self.steady_total)
            )
        return (
            "both the steady footprint %d and the migration peak %d exceed "
            "budget %d"
            % (self.steady_total, self.peak_total, self.limit)
        )


class Budget:
    """Accounts memory as shared loaded schemes plus per-active-session state.

    A session's state counts while the session is active, even when it has
    no new input arriving -- idle does not mean free. Shared scheme loads
    are reference counted: once no active session (or in-flight switch)
    needs a scheme, its load is released and becomes available again.
    """

    def __init__(self, limit: int):
        self.limit = limit
        self._schemes: Dict[str, PrecisionScheme] = {}
        self._refs: Dict[str, int] = {}  # active sessions per loaded scheme
        self._active_sessions: Dict[str, PrecisionScheme] = {}
        # sid -> (source, target): migration in flight, both states retained
        self._pending: Dict[str, Tuple[PrecisionScheme, PrecisionScheme]] = {}

    def load_scheme(self, scheme: PrecisionScheme) -> None:
        # Explicit pin (e.g. the uniform scheme loaded before any session).
        self._schemes[scheme.name] = scheme
        self._refs[scheme.name] = self._refs.get(scheme.name, 0) + 1

    def session_started(self, session_id: str, scheme: PrecisionScheme) -> None:
        self._retain(scheme)
        self._active_sessions[session_id] = scheme

    def session_ended(self, session_id: str) -> None:
        scheme = self._active_sessions.pop(session_id, None)
        pending = self._pending.pop(session_id, None)
        if scheme is not None:
            self._release(scheme)
        if pending is not None:
            self._release(pending[1])

    # -- two-phase switching -------------------------------------------

    def project_switch(
        self, session_id: str, target: PrecisionScheme
    ) -> SwitchProjection:
        source = self._active_sessions[session_id]
        current = self.report()

        # Steady state: this session's old state is gone; source scheme may
        # be unloaded when this was its last active user.
        steady_shared = current.shared_scheme_cost
        if not self._is_loaded(target):
            steady_shared += target.load_cost
        if self._refs.get(source.name, 0) <= 1:
            steady_shared -= source.load_cost
        steady_shared = max(steady_shared, 0)
        steady_state = (
            current.session_state_cost - source.state_cost + target.state_cost
        )
        steady_total = steady_shared + steady_state

        # Migration peak: old state still retained, new state already built,
        # and the target scheme must be loaded while the source is still live.
        peak_total = current.total + target.state_cost
        if not self._is_loaded(target):
            peak_total += target.load_cost
        return SwitchProjection(steady_total, peak_total, self.limit)

    def begin_switch(self, session_id: str, target: PrecisionScheme) -> SwitchProjection:
        """Reserve the migration peak before any old state is touched."""
        projection = self.project_switch(session_id, target)
        if not projection.fits:
            raise BudgetExceeded(projection)
        source = self._active_sessions[session_id]
        self._retain(target)
        self._pending[session_id] = (source, target)
        return projection

    def commit_switch(self, session_id: str) -> None:
        source, target = self._pending.pop(session_id)
        self._active_sessions[session_id] = target
        self._release(source)

    def cancel_switch(self, session_id: str) -> None:
        """A candidate that is not adopted releases everything it reserved."""
        if session_id not in self._pending:
            return
        _, target = self._pending.pop(session_id)
        self._release(target)

    def pending_switch(
        self, session_id: str
    ) -> Optional[Tuple[PrecisionScheme, PrecisionScheme]]:
        return self._pending.get(session_id)

    def report(self) -> BudgetReport:
        shared = sum(s.load_cost for s in self._schemes.values())
        state = sum(s.state_cost for s in self._active_sessions.values())
        # New state built by an in-flight migration is retained until commit.
        state += sum(target.state_cost for _, target in self._pending.values())
        total = shared + state
        return BudgetReport(total, shared, state, self.limit, total <= self.limit)

    def can_afford_session(self, scheme: PrecisionScheme) -> bool:
        extra_shared = 0 if self._is_loaded(scheme) else scheme.load_cost
        current = self.report()
        return current.total + extra_shared + scheme.state_cost <= self.limit

    # -- internals ------------------------------------------------------

    def _retain(self, scheme: PrecisionScheme) -> None:
        self._schemes[scheme.name] = scheme
        self._refs[scheme.name] = self._refs.get(scheme.name, 0) + 1

    def _release(self, scheme: PrecisionScheme) -> None:
        refs = self._refs.get(scheme.name, 0) - 1
        if refs <= 0:
            self._refs.pop(scheme.name, None)
            # An in-flight migration still needs the scheme even with no
            # active-session reference.
            if not any(
                source.name == scheme.name or target.name == scheme.name
                for source, target in self._pending.values()
            ):
                self._schemes.pop(scheme.name, None)
        else:
            self._refs[scheme.name] = refs

    def _is_loaded(self, scheme: PrecisionScheme) -> bool:
        return scheme.name in self._schemes


class BudgetExceeded(Exception):
    """Raised when a switch reservation would breach steady or peak budget."""

    def __init__(self, projection: SwitchProjection):
        self.projection = projection
        super().__init__(projection.describe_block())
