from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .budget import Budget
from .policy import SwitchPolicy
from .preview import SwitchPreview, preview_switch
from .schemes import PrecisionScheme
from .stream import GroupingRule, InputEvent, UncoveredInputError


class Mode(enum.Enum):
    UNIFORM = "uniform"            # one fixed scheme for everything
    INDEPENDENT = "independent"    # re-pick group/scheme per input, no session memory
    STATEFUL = "stateful"  # session keeps state; policy-gated causal switching


class Outcome(enum.Enum):
    EMITTED = "emitted"                    # output produced under current scheme
    SWITCHED_CONTINUED = "switched_continued"
    SWITCHED_RESTARTED = "switched_restarted"
    HELD_BY_MIN_HOLD = "held_by_min_hold"  # switch condition met but min_hold blocks it
    ROLLED_BACK = "rolled_back"            # switch attempted, not feasible, kept old scheme
    SWITCH_FAILED = "switch_failed"        # migration broke mid-way; old scheme kept
    UNCOVERED_INPUT = "uncovered_input"    # grouping rule does not cover this input
    BUDGET_DENIED = "budget_denied"        # not enough memory to run this input


class MigrationFailed(Exception):
    """A migrator raised while building the new state.

    The old state must stay usable: the migration reservation is released
    and the session keeps running under its old scheme.
    """


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
    # Non-None while a switch is reserved but not committed: old state in
    # ``state`` and the already-built new state in ``migrating_state`` coexist.
    migrating_state: object = None
    pending_target: Optional[str] = None

    @property
    def pending_switch(self) -> bool:
        return self.pending_target is not None


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
        interrupt_before_commit: bool = False,
    ):
        self.mode = mode
        self.rule = rule
        self.policy = policy
        self.group_schemes = group_schemes
        self.budget = budget
        self.uniform_scheme = uniform_scheme or next(iter(group_schemes.values()))
        self.interrupt_before_commit = interrupt_before_commit
        self.schemes_by_name = {
            sc.name: sc for sc in [self.uniform_scheme, *group_schemes.values()]
        }
        self.sessions: Dict[str, _Session] = {}
        self.history: List[StepResult] = []
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

    def resume_pending_switch(self, session_id: str) -> None:
        """Commit a switch left in flight before a save/reopen."""
        sess = self._require_pending(session_id)
        target = self.schemes_by_name[sess.pending_target]
        self._commit_pending_switch(session_id, sess, target)

    def cancel_pending_switch(self, session_id: str) -> None:
        """Abandon a candidate: release its reservation, keep old state."""
        sess = self._require_pending(session_id)
        if self.budget is not None:
            self.budget.cancel_switch(session_id)
        sess.migrating_state = None
        sess.pending_target = None

    # -- modes ----------------------------------------------------------

    def _feed_uniform(self, event: InputEvent, group: str) -> StepResult:
        scheme = self.uniform_scheme
        if not self.budget_can_afford(scheme):
            return self._budget_denied(event, group, scheme)
        if self.budget is not None and event.session_id not in self.sessions:
            self.budget.session_started(event.session_id, scheme)
        sess = self.sessions.setdefault(
            event.session_id, _Session(event.session_id, scheme, group))
        sess.outputs.append(event.position)
        return self._record(StepResult(
            event.session_id, event.position, Outcome.EMITTED, scheme.name, group,
        ))

    def _feed_independent(self, event: InputEvent, group: str) -> StepResult:
        # No session memory: scheme chosen from this input alone. Any state
        # is discarded between inputs, so only stateless-equivalent outputs.
        scheme = self.group_schemes[group]
        if not self.budget_can_afford(scheme):
            return self._budget_denied(event, group, scheme)
        return self._record(StepResult(
            event.session_id, event.position, Outcome.EMITTED, scheme.name, group,
            detail="scheme picked from this input only; no session state kept",
        ))

    def _feed_stateful(self, event: InputEvent, group: str) -> StepResult:
        sess = self.sessions.get(event.session_id)
        if sess is None:
            scheme = self.group_schemes[group]
            if not self.budget_can_afford(scheme):
                return self._budget_denied(event, group, scheme)
            sess = _Session(event.session_id, scheme, group, state=scheme.fresh_state())
            self.sessions[event.session_id] = sess
            if self.budget is not None:
                self.budget.session_started(event.session_id, scheme)
            sess.outputs.append(event.position)
            return self._record(StepResult(
                event.session_id, event.position, Outcome.EMITTED, scheme.name, group,
            ))

        if sess.pending_switch:
            # Resources for both states are still reserved; feeding more
            # inputs would hide that the switch was never completed.
            raise RuntimeError(
                "session %s has an in-flight switch; resume or cancel it "
                "before feeding more inputs" % event.session_id)

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
            # Cannot carry state and cannot silently drop it: keep the old
            # scheme and state, record the explicit rollback choice.
            sess.outputs.append(event.position)
            return self._record(StepResult(
                event.session_id, event.position, Outcome.ROLLED_BACK,
                sess.scheme.name, group, preview=prev,
                detail=prev.reason,
            ))

        if self.budget is not None:
            projection = self.budget.project_switch(event.session_id, target)
            if not projection.fits:
                # Reject before touching anything: the old state is never
                # dropped first and the failure is announced afterwards.
                sess.outputs.append(event.position)
                return self._record(StepResult(
                    event.session_id, event.position, Outcome.ROLLED_BACK,
                    sess.scheme.name, group, preview=prev,
                    detail="switch blocked: steady footprint %d and migration "
                           "peak %d exceed budget %d; old scheme and state "
                           "retained and remain usable"
                           % (projection.steady_total, projection.peak_total,
                              projection.limit),
                ))

        try:
            committed = self._perform_switch(
                event.session_id, sess, target,
                interrupt_before_commit=self.interrupt_before_commit,
            )
        except MigrationFailed as exc:
            sess.outputs.append(event.position)
            return self._record(StepResult(
                event.session_id, event.position, Outcome.SWITCH_FAILED,
                sess.scheme.name, group, preview=prev,
                detail="migration failed (%s); reserved resources released, "
                       "old scheme and state retained and remain usable" % exc,
            ))

        if not committed:
            # Interrupted between state migration and commit. The artifact
            # must be saved so the protected migration peak survives reopen.
            return self._record(StepResult(
                event.session_id, event.position, Outcome.SWITCHED_RESTARTED,
                target.name, group, preview=prev,
                detail="migration peak in flight: old state and new state "
                       "both retained; save the artifact, then resume or cancel",
            ))

        sess.outputs.append(event.position)
        return self._record(StepResult(
            event.session_id, event.position, Outcome.SWITCHED_CONTINUED,
            target.name, group, preview=prev,
        ))

    # -- two-phase switching --------------------------------------------

    def _perform_switch(
        self, session_id: str, sess: _Session, target: PrecisionScheme,
        interrupt_before_commit: bool = False,
    ) -> bool:
        """Returns False when the switch is deliberately left in flight.

        Phase 1 reserves the migration peak (target shared load plus the new
        state) while the old state is still fully in place. Phase 2 commits
        only after the new state was built successfully.
        """
        source = sess.scheme
        if self.budget is not None:
            self.budget.begin_switch(session_id, target)
        try:
            new_state = source.migrate_state_to(target, sess.state)
        except Exception as exc:
            if self.budget is not None:
                self.budget.cancel_switch(session_id)
            raise MigrationFailed(str(exc)) from exc
        # Old and new state coexist here: the protected migration peak.
        sess.migrating_state = new_state
        sess.pending_target = target.name
        if interrupt_before_commit:
            return False
        self._commit_pending_switch(session_id, sess, target)
        return True

    def _commit_pending_switch(
        self, session_id: str, sess: _Session, target: PrecisionScheme,
    ) -> None:
        new_state = sess.migrating_state
        sess.old_scheme_outputs = sess.old_scheme_outputs + sess.outputs
        sess.outputs = []
        sess.state = new_state
        sess.migrating_state = None
        sess.pending_target = None
        sess.scheme = target
        sess.group = self._group_of_scheme(target)
        sess.inputs_since_switch = 0
        if self.budget is not None:
            self.budget.commit_switch(session_id)

    # -- helpers --------------------------------------------------------

    def _require_pending(self, session_id: str) -> _Session:
        sess = self.sessions.get(session_id)
        if sess is None or not sess.pending_switch:
            raise ValueError("session %s has no in-flight switch" % session_id)
        return sess

    def _group_of_scheme(self, target: PrecisionScheme) -> str:
        for group, scheme in self.group_schemes.items():
            if scheme.name == target.name:
                return group
        raise ValueError("scheme %s is not a group candidate" % target.name)

    def budget_can_afford(self, scheme: PrecisionScheme) -> bool:
        if self.budget is None:
            return True
        return self.budget.can_afford_session(scheme)

    def _budget_denied(self, event: InputEvent, group: str, scheme: PrecisionScheme) -> StepResult:
        return self._record(StepResult(
            event.session_id, event.position, Outcome.BUDGET_DENIED, None, group,
            detail="cannot afford scheme %s within budget %d"
                   % (scheme.name, self.budget.limit),
        ))

    def _record(self, result: StepResult) -> StepResult:
        self.history.append(result)
        return result
