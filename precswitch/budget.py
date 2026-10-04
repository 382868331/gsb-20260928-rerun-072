from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Union

from .schemes import PrecisionScheme


@dataclass(frozen=True)
class BudgetReport:
    total: int
    shared_scheme_cost: int
    session_state_cost: int
    limit: int
    fits: bool


@dataclass(frozen=True)
class SwitchQuote:
    """Resource basis for one session's proposed scheme switch.

    Two distinct occupancies must both fit:

    * steady_total: usage after the switch settles (new persistent state
      replaces the old one, shared load is released when no longer used);
    * peak_total: usage while migration is in flight, when the old state
      and the freshly reserved new state coexist, plus any extra shared
      scheme load.
    """

    session_id: str
    from_scheme: str
    to_scheme: str
    limit: int
    current_total: int
    steady_total: int
    peak_total: int
    extra_shared: int
    old_state_cost: int
    new_state_cost: int
    steady_fits: bool
    peak_fits: bool

    @property
    def fits(self) -> bool:
        return self.steady_fits and self.peak_fits

    @property
    def reason(self) -> str:
        if self.fits:
            return ""
        blockers = []
        if not self.steady_fits:
            blockers.append(
                "steady-state usage would be %d > budget %d "
                "(shared load plus new persistent state %d of scheme %s)"
                % (self.steady_total, self.limit,
                   self.new_state_cost, self.to_scheme)
            )
        if not self.peak_fits:
            blockers.append(
                "migration peak would be %d > budget %d while old state %d "
                "and new state %d coexist (extra shared load %d)"
                % (self.peak_total, self.limit, self.old_state_cost,
                   self.new_state_cost, self.extra_shared)
            )
        return (
            "switch %s -> %s denied: %s; old scheme and its state are "
            "retained and remain usable"
            % (self.from_scheme, self.to_scheme, "; ".join(blockers))
        )


class BudgetDeniedError(Exception):
    """Raised when a switch reservation has no resource basis. The old
    state is never touched before this point."""

    def __init__(self, quote: SwitchQuote):
        super().__init__(quote.reason)
        self.quote = quote


@dataclass(frozen=True)
class _Reservation:
    session_id: str
    old: PrecisionScheme
    target: PrecisionScheme
    target_newly_loaded: bool


class Budget:
    """Accounts memory as shared loaded schemes plus per-active-session state.

    A session's state counts while the session is active, even when it has
    no new input arriving -- idle does not mean free.

    Scheme switches use an explicit reservation protocol: begin reserves
    the real migration peak (old and new state coexist, extra shared load
    retained); commit settles to the steady state; abort releases only the
    candidate reservation and keeps the old scheme usable.
    """

    def __init__(self, limit: int):
        self.limit = limit
        self._schemes: Dict[str, PrecisionScheme] = {}
        self._load_refs: Dict[str, int] = {}
        self._active_sessions: Dict[str, PrecisionScheme] = {}
        self._migrating: Dict[str, _Reservation] = {}

    def _retain(self, scheme: PrecisionScheme) -> bool:
        """Keep a scheme's shared load counted once; True if newly loaded."""
        refs = self._load_refs.get(scheme.name, 0)
        self._load_refs[scheme.name] = refs + 1
        if refs == 0:
            self._schemes[scheme.name] = scheme
            return True
        return False

    def _release(self, name: str) -> None:
        refs = self._load_refs.get(name, 0)
        if refs <= 1:
            self._load_refs.pop(name, None)
            self._schemes.pop(name, None)
        else:
            self._load_refs[name] = refs - 1

    def load_scheme(self, scheme: PrecisionScheme) -> None:
        self._retain(scheme)

    def session_started(self, session_id: str, scheme: PrecisionScheme) -> None:
        self._retain(scheme)
        self._active_sessions[session_id] = scheme

    def session_ended(self, session_id: str) -> None:
        if session_id in self._migrating:
            self.abort_switch(session_id)
        scheme = self._active_sessions.pop(session_id, None)
        if scheme is not None:
            self._release(scheme.name)

    # -- switch reservation protocol ------------------------------------

    def quote_switch(
        self, session_id: str, target: PrecisionScheme
    ) -> SwitchQuote:
        old = self._active_sessions[session_id]
        rep = self.report()
        if target.name == old.name:
            return SwitchQuote(
                session_id=session_id,
                from_scheme=old.name,
                to_scheme=target.name,
                limit=self.limit,
                current_total=rep.total,
                steady_total=rep.total,
                peak_total=rep.total,
                extra_shared=0,
                old_state_cost=old.state_cost,
                new_state_cost=target.state_cost,
                steady_fits=rep.fits,
                peak_fits=rep.fits,
            )
        extra_shared = (
            target.load_cost
            if self._load_refs.get(target.name, 0) == 0
            else 0
        )
        # Migration peak: target load is reserved while the old scheme is
        # still retained; new state is allocated before old state releases.
        peak_total = rep.total + extra_shared + target.state_cost
        # Steady state: this session's old state is gone; the old shared
        # load is released only when no other session retains the scheme.
        new_shared = (
            rep.shared_scheme_cost
            - (old.load_cost if self._load_refs.get(old.name, 0) <= 1 else 0)
            + extra_shared
        )
        new_state_total = (
            rep.session_state_cost - old.state_cost + target.state_cost
        )
        steady_total = new_shared + new_state_total
        return SwitchQuote(
            session_id=session_id,
            from_scheme=old.name,
            to_scheme=target.name,
            limit=self.limit,
            current_total=rep.total,
            steady_total=steady_total,
            peak_total=peak_total,
            extra_shared=extra_shared,
            old_state_cost=old.state_cost,
            new_state_cost=target.state_cost,
            steady_fits=steady_total <= self.limit,
            peak_fits=peak_total <= self.limit,
        )

    def begin_switch(
        self, session_id: str, target: PrecisionScheme
    ) -> _Reservation:
        """Reserve the migration peak. Callers mutate session state only
        after this succeeds, so a denial never loses the old state."""
        quote = self.quote_switch(session_id, target)
        if not quote.fits:
            raise BudgetDeniedError(quote)
        newly_loaded = self._retain(target)
        reservation = _Reservation(
            session_id, self._active_sessions[session_id],
            target, newly_loaded,
        )
        self._migrating[session_id] = reservation
        return reservation

    def commit_switch(self, reservation: _Reservation) -> None:
        """Settle: install the new scheme/state, release old resources."""
        sid = reservation.session_id
        pending = self._migrating.pop(sid, None)
        if pending is None or pending != reservation:
            raise ValueError("no pending switch reservation for %s" % sid)
        self._active_sessions[sid] = reservation.target
        self._release(reservation.old.name)

    def abort_switch(self, reservation: Union[_Reservation, str]) -> None:
        """Cancel a candidate: release exactly what begin reserved; the
        old active scheme/state keep counting untouched."""
        sid = (reservation.session_id
               if isinstance(reservation, _Reservation) else reservation)
        pending = self._migrating.pop(sid, None)
        if pending is None:
            raise ValueError("no pending switch reservation for %s" % sid)
        if pending.target_newly_loaded:
            self._release(pending.target.name)

    def pending_switch(self, session_id: str) -> Optional[_Reservation]:
        return self._migrating.get(session_id)

    def report(self) -> BudgetReport:
        shared = sum(scheme.load_cost for scheme in self._schemes.values())
        state = sum(
            scheme.state_cost for scheme in self._active_sessions.values()
        ) + sum(
            reservation.target.state_cost
            for reservation in self._migrating.values()
        )
        total = shared + state
        return BudgetReport(total, shared, state, self.limit, total <= self.limit)

    def can_afford_session(self, scheme: PrecisionScheme) -> bool:
        extra_shared = 0 if scheme.name in self._schemes else scheme.load_cost
        current = self.report()
        return current.total + extra_shared + scheme.state_cost <= self.limit
