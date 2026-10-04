from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .budget import Budget, BudgetDeniedError
from .policy import SwitchPolicy
from .preview import SwitchPreview, preview_switch
from .schemes import PrecisionScheme
from .stream import GroupingRule, InputEvent, UncoveredInputError


class Mode(enum.Enum):
    UNIFORM = "uniform"            # one fixed scheme for everything
    INDEPENDENT = "independent"    # re-pick group/scheme per input, no session memory
    STATEFUL = "stateful"          # session keeps state; policy-gated causal switching


class Outcome(enum.Enum):
    EMITTED = "emitted"                    # output produced under current scheme
    SWITCHED_CONTINUED = "switched_continued"
    SWITCH_RESERVED = "switch_reserved"    # peak reserved, migration pending
    SWITCH_ABORTED = "switch_aborted"      # candidate dropped, old state kept
    SWITCH_FAILED = "switch_failed"        # migration raised; resources rolled back
    SWITCHED_RESTARTED = "switched_restarted"
    HELD_BY_MIN_HOLD = "held_by_min_hold"  # switch condition met but min_hold blocks it
    ROLLED_BACK = "rolled_back"            # switch attempted, not feasible, kept old scheme
    UNCOVERED_INPUT = "uncovered_input"    # grouping rule does not cover this input
    BUDGET_DENIED = "budget_denied"        # not enough memory to run this input


@dataclass
class StepResult:
    session_id: str
    position: int
    outcome: Outcome
    scheme: Optional[str]
    group: Optional[str] = None
    preview: Optional[SwitchPreview] = None
    detail: str = ""


@dataclass
class _Session:
    session_id: str
    scheme: PrecisionScheme
    group: str
    state: object = None
    inputs_since_switch: int = 0
    outputs: List[int] = field(default_factory=list)  # positions under current scheme
    old_scheme_outputs: List[int] = field(default_factory=list)
    migrating_to: Optional[str] = None
    pending_state: object = None


class Engine:
    """Runs session input streams under one of the three selection modes.

    Switching decisions are causal: at position i only inputs 0..i (already
    seen) and the fixed rules are consulted. Selecting a historical path by
    later-observed error is not possible in this API.
    """

    def __init__(
        self,
        mode: Mode,
        rule: GroupingRule,
        policy: SwitchPolicy,
        group_schemes: Dict[str, PrecisionScheme],
        budget: Optional[Budget] = None,
        uniform_scheme: Optional[PrecisionScheme] = None,
    ):
        self.mode = mode
        self.rule = rule
        self.policy = policy
        self.group_schemes = group_schemes
        self.budget = budget
        self.uniform_scheme = uniform_scheme or next(iter(group_schemes.values()))
        self.sessions: Dict[str, _Session] = {}
        self.history: List[StepResult] = []
        self._pending: Dict[str, dict] = {}
        if budget is not None and mode is Mode.UNIFORM:
            budget.load_scheme(self.uniform_scheme)

    # -- public API -----------------------------------------------------

    def feed(self, event: InputEvent) -> StepResult:
        try:
            group = self.rule.group_of(event.payload)
        except UncoveredInputError as exc:
            return self._record(StepResult(
                event.session_id, event.position, Outcome.UNCOVERED_INPUT, None,
                detail=str(exc),
            ))
        if self.mode is Mode.UNIFORM:
            return self._feed_uniform(event, group)
        if self.mode is Mode.INDEPENDENT:
            return self._feed_independent(event, group)
        return self._feed_stateful(event, group)

    def end_session(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)
        if self.budget is not None:
            self.budget.session_ended(session_id)

    # -- modes ----------------------------------------------------------

    def _feed_uniform(self, event: InputEvent, group: str) -> StepResult:
        scheme = self.uniform_scheme
        if not self._check_budget(event.session_id, scheme):
            return self._budget_denied(event, group, scheme)
        if self.budget is not None and event.session_id not in self.sessions:
            self.budget.session_started(event.session_id, scheme)
        self.sessions.setdefault(event.session_id, _Session(event.session_id, scheme, group))
        sess = self.sessions[event.session_id]
        sess.outputs.append(event.position)
        return self._record(StepResult(
            event.session_id, event.position, Outcome.EMITTED, scheme.name, group,
        ))

    def _feed_independent(self, event: InputEvent, group: str) -> StepResult:
        # No session memory: scheme chosen from this input alone. Any state
        # is discarded between inputs, so only stateless-equivalent outputs.
        scheme = self.group_schemes[group]
        if not self._check_budget(event.session_id + "@%d" % event.position, scheme):
            return self._budget_denied(event, group, scheme)
        return self._record(StepResult(
            event.session_id, event.position, Outcome.EMITTED, scheme.name, group,
            detail="scheme picked from this input only; no session state kept",
        ))

    def _feed_stateful(self, event: InputEvent, group: str) -> StepResult:
        sess = self.sessions.get(event.session_id)
        if sess is None:
            scheme = self.group_schemes[group]
            if not self._check_budget(event.session_id, scheme):
                return self._budget_denied(event, group, scheme)
            sess = _Session(event.session_id, scheme, group, state=scheme.fresh_state())
            self.sessions[event.session_id] = sess
            if self.budget is not None:
                self.budget.session_started(event.session_id, scheme)
            sess.outputs.append(event.position)
            return self._record(StepResult(
                event.session_id, event.position, Outcome.EMITTED, scheme.name, group,
            ))

        sess.inputs_since_switch += 1
        if group == sess.group:
            sess.outputs.append(event.position)
            return self._record(StepResult(
                event.session_id, event.position, Outcome.EMITTED, sess.scheme.name, group,
            ))

        target = self.group_schemes[group]
        if not self.policy.may_switch(sess.group, group, sess.inputs_since_switch):
            sess.outputs.append(event.position)
            return self._record(StepResult(
                event.session_id, event.position, Outcome.HELD_BY_MIN_HOLD,
                sess.scheme.name, group,
                detail="group changed to %s but min_hold=%d not reached"
                       % (group, self.policy.min_hold),
            ))

        prev = preview_switch(
            sess.scheme, target, sess.state, event.position,
            sess.old_scheme_outputs + sess.outputs,
        )
        if not prev.can_continue:
            # Cannot carry state and cannot silently drop it: roll back,
            # keep the old scheme, record the explicit rollback choice.
            sess.outputs.append(event.position)
            return self._record(StepResult(
                event.session_id, event.position, Outcome.ROLLED_BACK,
                sess.scheme.name, group, preview=prev,
                detail=prev.reason,
            ))

        # Reserve resources before touching any state: denial must leave
        # the old scheme and its persistent state fully usable.
        if self.budget is not None:
            try:
                reservation = self.budget.begin_switch(
                    event.session_id, target
                )
            except BudgetDeniedError as exc:
                sess.outputs.append(event.position)
                return self._record(StepResult(
                    event.session_id, event.position, Outcome.ROLLED_BACK,
                    sess.scheme.name, group, preview=prev,
                    detail=exc.quote.reason,
                ))
        else:
            reservation = None

        try:
            new_state = sess.scheme.migrate_state_to(target, sess.state)
        except Exception as exc:  # migration failure: keep old scheme alive
            if self.budget is not None:
                self.budget.abort_switch(reservation)
            sess.outputs.append(event.position)
            return self._record(StepResult(
                event.session_id, event.position, Outcome.SWITCH_FAILED,
                sess.scheme.name, group, preview=prev,
                detail="state migration %s -> %s failed (%s); released the "
                       "candidate reservation and kept the old scheme/state"
                       % (sess.scheme.name, target.name, exc),
            ))

        self._commit_switch(event.session_id, sess, target, group, new_state,
                            reservation)
        sess.outputs.append(event.position)
        return self._record(StepResult(
            event.session_id, event.position, Outcome.SWITCHED_CONTINUED,
            target.name, group, preview=prev,
        ))

    def _commit_switch(self, session_id, sess, target, group, new_state,
                       reservation) -> None:
        if self.budget is not None:
            self.budget.commit_switch(reservation)
        sess.state = new_state
        sess.migrating_to = None
        sess.pending_state = None
        sess.old_scheme_outputs = sess.old_scheme_outputs + sess.outputs
        sess.outputs = []
        sess.scheme = target
        sess.group = group
        sess.inputs_since_switch = 0

    # -- multi-step migration API (persistable across save/reload) ------

    def begin_migration(self, session_id: str, group: str,
                        position: int) -> StepResult:
        """Reserve the migration peak and move the new state into a
        pending slot; the old state stays live until commit. The pending
        reservation survives save/reload and reflects the true retained
        resources while migration is in flight."""
        sess = self.sessions[session_id]
        target = self.group_schemes[group]
        prev = preview_switch(
            sess.scheme, target, sess.state, position,
            sess.old_scheme_outputs + sess.outputs,
        )
        if not prev.can_continue:
            return self._record(StepResult(
                session_id, position, Outcome.ROLLED_BACK,
                sess.scheme.name, group, preview=prev,
                detail=prev.reason,
            ))
        if self.budget is not None:
            try:
                reservation = self.budget.begin_switch(session_id, target)
            except BudgetDeniedError as exc:
                return self._record(StepResult(
                    session_id, position, Outcome.ROLLED_BACK,
                    sess.scheme.name, group, preview=prev,
                    detail=exc.quote.reason,
                ))
        else:
            reservation = None
        try:
            new_state = sess.scheme.migrate_state_to(target, sess.state)
        except Exception as exc:
            if self.budget is not None:
                self.budget.abort_switch(reservation)
            return self._record(StepResult(
                session_id, position, Outcome.SWITCH_FAILED,
                sess.scheme.name, group, preview=prev,
                detail="state migration %s -> %s failed (%s); released the "
                       "candidate reservation and kept the old scheme/state"
                       % (sess.scheme.name, target.name, exc),
            ))
        sess.migrating_to = target.name
        sess.pending_state = new_state
        self._pending[session_id] = {
            "target": target, "group": group, "position": position,
            "reservation": reservation, "preview": prev,
        }
        return self._record(StepResult(
            session_id, position, Outcome.SWITCH_RESERVED,
            sess.scheme.name, group, preview=prev,
            detail="migration peak reserved; new state pending, old "
                   "scheme/state remain live and usable",
        ))

    def commit_migration(self, session_id: str, position: int) -> StepResult:
        pending = getattr(self, "_pending", {}).get(session_id)
        if pending is None:
            raise ValueError("no pending migration for %s" % session_id)
        sess = self.sessions[session_id]
        self._commit_switch(session_id, sess, pending["target"],
                            pending["group"], sess.pending_state,
                            pending["reservation"])
        del self._pending[session_id]
        sess.outputs.append(position)
        return self._record(StepResult(
            session_id, position, Outcome.SWITCHED_CONTINUED,
            sess.scheme.name, pending["group"], preview=pending["preview"],
        ))

    def abort_migration(self, session_id: str, position: int) -> StepResult:
        pending = getattr(self, "_pending", {}).pop(session_id, None)
        if pending is None:
            raise ValueError("no pending migration for %s" % session_id)
        sess = self.sessions[session_id]
        if self.budget is not None:
            self.budget.abort_switch(pending["reservation"])
        sess.migrating_to = None
        sess.pending_state = None
        sess.outputs.append(position)
        return self._record(StepResult(
            session_id, position, Outcome.SWITCH_ABORTED,
            sess.scheme.name, sess.group, preview=pending["preview"],
            detail="candidate %s released; only the old scheme and its "
                   "state remain charged and usable" % pending["target"].name,
        ))

    # -- helpers --------------------------------------------------------

    def _check_budget(self, session_id: str, scheme: PrecisionScheme) -> bool:
        if self.budget is None:
            return True
        return self.budget.can_afford_session(scheme)

    def _budget_denied(self, event: InputEvent, group: str, scheme: PrecisionScheme) -> StepResult:
        return self._record(StepResult(
            event.session_id, event.position, Outcome.BUDGET_DENIED, None, group,
            detail="cannot afford scheme %s within budget" % scheme.name,
        ))

    def _record(self, result: StepResult) -> StepResult:
        self.history.append(result)
        return result
