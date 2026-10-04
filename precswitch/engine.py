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
    SWITCH_RESERVED = "switch_reserved"           # peak reserved; migration candidate created
    SWITCH_COMMITTED = "switch_committed"         # pending candidate adopted after reopen
    SWITCH_CANCELLED = "switch_cancelled"         # pending candidate abandoned; old state kept
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
    candidate_id: Optional[str] = None


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
    pending_position: Optional[int] = None
    # Stable identity of the in-flight candidate; survives save/reopen so a
    # later commit or cancel is attributed to the exact same switch attempt.
    candidate_id: Optional[str] = None
    # Identity of the most recently resolved candidate, kept so a repeated
    # resume/cancel can name the attempt instead of silently duplicating it.
    last_candidate_id: Optional[str] = None

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

    def resume_pending_switch(self, session_id: str) -> StepResult:
        """Commit a switch left in flight before a save/reopen.

        The commit is attributed to the candidate identity stored when the
        peak was reserved. Repeating it does not fabricate another fact:
        once the candidate is gone the call raises instead of appending.
        """
        sess = self._require_session_with_switch(session_id)
        if not sess.pending_switch:
            raise ValueError(
                "session %s candidate %s was already resolved; no pending "
                "switch remains" % (session_id, sess.last_candidate_id))
        target = self.schemes_by_name[sess.pending_target]
        candidate_id = sess.candidate_id
        position = sess.pending_position
        self._commit_pending_switch(session_id, sess, target)
        return self._record(StepResult(
            session_id, position, Outcome.SWITCH_COMMITTED, target.name,
            sess.group, candidate_id=candidate_id,
            detail="candidate %s committed after resume: old state and old "
                   "shared load released, target state effective" % candidate_id,
        ))

    def cancel_pending_switch(self, session_id: str) -> StepResult:
        """Abandon a candidate: release its reservation, keep old state.

        The cancellation is recorded as its own fact under the candidate's
        stable identity; it is never rewritten as a completed switch. A
        repeated cancel raises instead of appending a duplicate fact.
        """
        sess = self._require_session_with_switch(session_id)
        if not sess.pending_switch:
            raise ValueError(
                "session %s candidate %s was already resolved; no pending "
                "switch remains" % (session_id, sess.last_candidate_id))
        candidate_id = sess.candidate_id
        position = sess.pending_position
        old_name = sess.scheme.name
        if self.budget is not None:
            self.budget.cancel_switch(session_id)
        sess.migrating_state = None
        sess.pending_target = None
        sess.pending_position = None
        sess.last_candidate_id = candidate_id
        sess.candidate_id = None
        return self._record(StepResult(
            session_id, position, Outcome.SWITCH_CANCELLED, old_name, sess.group,
            candidate_id=candidate_id,
            detail="candidate %s cancelled before commit: migration "
                   "reservation released, old scheme and old state retained "
                   "and remain usable; target never became effective"
                   % candidate_id,
        ))

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
                    detail="switch blocked: %s; old scheme and state "
                           "retained and remain usable"
                           % projection.describe_block(),
                ))

        candidate_id = self._new_candidate_id(event.session_id, event.position, target)
        try:
            committed = self._perform_switch(
                event.session_id, sess, target,
                candidate_id=candidate_id,
                position=event.position,
                interrupt_before_commit=self.interrupt_before_commit,
            )
        except MigrationFailed as exc:
            sess.outputs.append(event.position)
            return self._record(StepResult(
                event.session_id, event.position, Outcome.SWITCH_FAILED,
                sess.scheme.name, group, preview=prev, candidate_id=candidate_id,
                detail="migration failed (%s); reserved resources released, "
                       "old scheme and state retained and remain usable" % exc,
            ))

        if not committed:
            # Interrupted between state migration and commit. The artifact
            # must be saved so the protected migration peak survives reopen.
            self._record(StepResult(
                event.session_id, event.position, Outcome.SWITCH_RESERVED,
                sess.scheme.name, group, preview=prev, candidate_id=candidate_id,
                detail="candidate %s reserved the migration peak and built "
                       "the new state while retaining the old state"
                       % candidate_id,
            ))
            return self._record(StepResult(
                event.session_id, event.position, Outcome.SWITCHED_RESTARTED,
                target.name, group, preview=prev, candidate_id=candidate_id,
                detail="candidate %s migration in flight: old state and new "
                       "state both retained; save the artifact, then resume "
                       "or cancel under the same candidate identity"
                       % candidate_id,
            ))

        sess.outputs.append(event.position)
        return self._record(StepResult(
            event.session_id, event.position, Outcome.SWITCHED_CONTINUED,
            target.name, group, preview=prev, candidate_id=candidate_id,
        ))

    # -- two-phase switching --------------------------------------------

    def _perform_switch(
        self, session_id: str, sess: _Session, target: PrecisionScheme,
        candidate_id: str, position: int,
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
        sess.pending_position = position
        sess.candidate_id = candidate_id
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
        sess.pending_position = None
        sess.last_candidate_id = sess.candidate_id
        sess.candidate_id = None
        sess.scheme = target
        sess.group = self._group_of_scheme(target)
        sess.inputs_since_switch = 0
        if self.budget is not None:
            self.budget.commit_switch(session_id)

    # -- helpers --------------------------------------------------------

    def _require_session_with_switch(self, session_id: str) -> _Session:
        sess = self.sessions.get(session_id)
        if sess is None:
            raise ValueError("session %s has no in-flight switch" % session_id)
        return sess

    def _new_candidate_id(
        self, session_id: str, position: int, target: PrecisionScheme
    ) -> str:
        # Deterministic and stable across save/reopen: the same reserved
        # candidate keeps one identity whether it is later committed or
        # cancelled, and a genuinely new switch at a later position gets a
        # distinct identity.
        return "%s@%d->%s" % (session_id, position, target.name)

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
