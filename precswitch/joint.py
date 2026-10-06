"""Joint multi-session migration resource scheduling research.

A researcher picks a set of migration candidates (session -> target scheme)
and an order. The planner simulates the whole sequence against a clone of
the live budget -- old resources still occupied, the target reserved only
once even when several sessions share it, the conversion peak of each step,
and shared loads released only when their last user migrates away -- and
answers whether the order fits the hard capacity limit, or searches for an
order that does. Planning never touches the real ledger: a plan that was
not executed must not be read as already released.
"""
from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .budget import Budget, SwitchProjection
from .engine import Engine, MigrationFailed
from .schemes import PrecisionScheme

#: Permutation search is exact up to this many steps; beyond it a greedy
#: pass tries to extend a fitting prefix one step at a time.
_EXACT_SEARCH_LIMIT = 8


@dataclass(frozen=True)
class MigrationStep:
    """One candidate: migrate ``session_id`` to scheme ``target``."""

    session_id: str
    target: PrecisionScheme


@dataclass(frozen=True)
class StepCheck:
    step: MigrationStep
    projection: Optional[SwitchProjection]
    feasible: bool
    detail: str = ""


@dataclass(frozen=True)
class JointPlan:
    """Simulated outcome of a candidate order. Pure projection: nothing
    here was executed, so nothing here released or reserved anything."""

    strategy: str                      # "sequential" | "batch"
    steps: Tuple[MigrationStep, ...]
    checks: Tuple[StepCheck, ...]
    feasible: bool
    peak_total: int
    waiting: int
    limit: int

    def blocking_details(self) -> Tuple[str, ...]:
        return tuple(c.detail for c in self.checks if not c.feasible)


def check_joint_order(budget: Budget, steps: Sequence[MigrationStep]) -> JointPlan:
    """Simulate migrating one session at a time, in the given order.

    At each step the old state is still occupied, the target reservation is
    added (shared target loads count once), and the conversion peak must
    fit before the step is taken. A session still on the old scheme keeps
    that shared load pinned, so it is never released early.
    """
    sim = copy.deepcopy(budget)
    checks: List[StepCheck] = []
    peak = sim.report().total
    feasible = True
    for index, step in enumerate(steps):
        if step.session_id not in sim._active_sessions:
            checks.append(StepCheck(step, None, False,
                                    "session %s is not active" % step.session_id))
            feasible = False
            continue
        projection = sim.project_switch(step.session_id, step.target)
        if not projection.fits:
            blocking = " and ".join(projection.blocking_parts())
            checks.append(StepCheck(
                step, projection, False,
                "step %d blocked: %s exceed budget %d"
                % (index, blocking, projection.limit)))
            feasible = False
            continue
        sim.begin_switch(step.session_id, step.target)
        sim.commit_switch(step.session_id)
        checks.append(StepCheck(step, projection, True))
        peak = max(peak, projection.peak_total)
    waiting = len(steps) * (len(steps) - 1) // 2  # each step waits for all before it
    return JointPlan("sequential", tuple(steps), tuple(checks), feasible,
                     peak, waiting, budget.limit)


def check_joint_batch(budget: Budget, steps: Sequence[MigrationStep]) -> JointPlan:
    """Simulate reserving every candidate up front, then committing all.

    No step waits for another to finish before starting (waiting 0), but
    every conversion peak overlaps: old states stay occupied and all target
    reservations coexist until the commit phase drains them.
    """
    sim = copy.deepcopy(budget)
    checks: List[StepCheck] = []
    peak = sim.report().total
    feasible = True
    reserved: List[str] = []
    for index, step in enumerate(steps):
        if step.session_id not in sim._active_sessions:
            checks.append(StepCheck(step, None, False,
                                    "session %s is not active" % step.session_id))
            feasible = False
            continue
        projection = sim.project_switch(step.session_id, step.target)
        if not projection.fits:
            blocking = " and ".join(projection.blocking_parts())
            checks.append(StepCheck(
                step, projection, False,
                "batch reservation %d blocked: %s exceed budget %d"
                % (index, blocking, projection.limit)))
            feasible = False
            continue
        sim.begin_switch(step.session_id, step.target)
        reserved.append(step.session_id)
        checks.append(StepCheck(step, projection, True))
        peak = max(peak, sim.report().total)
    for session_id in reserved:
        sim.commit_switch(session_id)
    return JointPlan("batch", tuple(steps), tuple(checks), feasible,
                     peak, 0, budget.limit)


def find_feasible_order(
    budget: Budget, steps: Sequence[MigrationStep]
) -> Optional[JointPlan]:
    """Search for an order that fits the hard capacity limit.

    Returns a feasible sequential JointPlan, or None when no order fits --
    the caller should then report that no feasible order exists rather than
    silently dropping candidates.
    """
    steps = tuple(steps)
    if len(steps) <= _EXACT_SEARCH_LIMIT:
        for perm in itertools.permutations(steps):
            plan = check_joint_order(budget, perm)
            if plan.feasible:
                return plan
        return None
    # Greedy fallback for larger sets: repeatedly append any step that fits.
    remaining = list(steps)
    chosen: List[MigrationStep] = []
    sim = copy.deepcopy(budget)
    while remaining:
        progressed = False
        for step in list(remaining):
            if step.session_id not in sim._active_sessions:
                return None
            projection = sim.project_switch(step.session_id, step.target)
            if projection.fits:
                sim.begin_switch(step.session_id, step.target)
                sim.commit_switch(step.session_id)
                chosen.append(step)
                remaining.remove(step)
                progressed = True
        if not progressed:
            return None
    return check_joint_order(budget, chosen)


@dataclass(frozen=True)
class StrategyComparison:
    """Peak vs waiting for migrating one-by-one or as a batch. Both are
    simulations: neither plan has executed, so neither freed anything."""

    sequential: JointPlan
    batch: JointPlan


def compare_strategies(
    budget: Budget, steps: Sequence[MigrationStep]
) -> StrategyComparison:
    return StrategyComparison(check_joint_order(budget, steps),
                              check_joint_batch(budget, steps))


@dataclass(frozen=True)
class StepOutcome:
    step: MigrationStep
    status: str  # committed | failed | denied | cancelled | reserved | skipped
    detail: str = ""


@dataclass(frozen=True)
class JointReport:
    strategy: str
    outcomes: Tuple[StepOutcome, ...]
    peak_total: int
    waiting: int
    limit: int
    # After a failure/cancel, whether each not-yet-executed step could
    # still proceed from the real post-failure ledger.
    continuation: Tuple[Tuple[str, bool], ...] = ()


class JointMigrationExecutor:
    """Executes a planned joint migration against a live engine.

    A step that fails or is cancelled releases only the resources that
    belong to its own candidate; sessions still on their old scheme stay
    usable, and shared target loads survive while another candidate or
    session still references them. Completion/cancellation of an already
    resolved candidate is an idempotent no-op (handled by the engine), so
    duplicate notifications never release or load twice.
    """

    def __init__(self, engine: Engine):
        self.engine = engine

    def execute(
        self,
        steps: Sequence[MigrationStep],
        batch: bool = False,
        cancel: Sequence[str] = (),
    ) -> JointReport:
        budget = self.engine.budget
        if budget is None:
            raise ValueError("joint migration research requires a budget")
        for step in steps:
            # Research candidates may sit outside the grouping rule; make
            # them resolvable after a save/reopen without re-registering.
            self.engine.schemes_by_name.setdefault(step.target.name, step.target)
        peak = budget.report().total
        outcomes: List[StepOutcome] = []

        def track():
            nonlocal peak
            peak = max(peak, budget.report().total)

        if batch:
            reserved: List[MigrationStep] = []
            for step in steps:
                if self._reserve(step, outcomes):
                    reserved.append(step)
                track()
            cancel_set = set(cancel)
            for step in reserved:
                sid = step.session_id
                if sid in cancel_set:
                    self.engine.cancel_pending_switch(sid)
                    outcomes.append(StepOutcome(
                        step, "cancelled",
                        "candidate cancelled; only its own reservation "
                        "released, old session retained"))
                else:
                    self.engine.resume_pending_switch(sid)
                    outcomes.append(StepOutcome(step, "committed"))
                track()
            strategy = "batch"
            waiting = 0
        else:
            for step in steps:
                self._run_single(step, outcomes)
                track()
            strategy = "sequential"
            waiting = len(steps) * (len(steps) - 1) // 2

        continuation = self._continuation(steps, outcomes)
        return JointReport(strategy, tuple(outcomes), peak, waiting,
                           budget.limit, continuation)

    # -- internals ------------------------------------------------------

    def _reserve(self, step: MigrationStep, outcomes: List[StepOutcome]) -> bool:
        engine, budget = self.engine, self.engine.budget
        sess = engine.sessions.get(step.session_id)
        if sess is None or sess.pending_switch:
            outcomes.append(StepOutcome(
                step, "skipped",
                "session missing or already has an in-flight switch"))
            return False
        projection = budget.project_switch(step.session_id, step.target)
        if not projection.fits:
            outcomes.append(StepOutcome(
                step, "denied",
                "reservation blocked: %s exceed budget %d; old session "
                "untouched"
                % (" and ".join(projection.blocking_parts()), projection.limit)))
            return False
        try:
            engine._perform_switch(
                step.session_id, sess, step.target,
                interrupt_before_commit=True,
                position=sess.last_position, group=None)
        except MigrationFailed as exc:
            outcomes.append(StepOutcome(
                step, "failed",
                "migration failed (%s); only this candidate's reservation "
                "released, old session retained and usable" % exc))
            return False
        outcomes.append(StepOutcome(step, "reserved",
                                    "migration peak reserved, old state intact"))
        return True

    def _run_single(self, step: MigrationStep, outcomes: List[StepOutcome]) -> None:
        engine, budget = self.engine, self.engine.budget
        sess = engine.sessions.get(step.session_id)
        if sess is None or sess.pending_switch:
            outcomes.append(StepOutcome(
                step, "skipped",
                "session missing or already has an in-flight switch"))
            return
        projection = budget.project_switch(step.session_id, step.target)
        if not projection.fits:
            outcomes.append(StepOutcome(
                step, "denied",
                "step blocked: %s exceed budget %d; old session retained "
                "and usable"
                % (" and ".join(projection.blocking_parts()), projection.limit)))
            return
        try:
            engine._perform_switch(
                step.session_id, sess, step.target,
                interrupt_before_commit=False,
                position=sess.last_position, group=None)
        except MigrationFailed as exc:
            outcomes.append(StepOutcome(
                step, "failed",
                "migration failed (%s); only this candidate's reservation "
                "released, old session retained and usable" % exc))
            return
        outcomes.append(StepOutcome(step, "committed"))

    def _continuation(
        self, steps: Sequence[MigrationStep], outcomes: List[StepOutcome]
    ) -> Tuple[Tuple[str, bool], ...]:
        """For steps never executed (e.g. after a batch reservation was
        denied), report from the real current ledger whether each could
        still proceed -- without executing anything."""
        done = {o.step.session_id for o in outcomes
                if o.status in ("committed", "cancelled", "reserved")}
        remaining = [s for s in steps if s.session_id not in done]
        if not remaining:
            return ()
        plan = check_joint_order(self.engine.budget, remaining)
        return tuple((c.step.session_id, c.feasible) for c in plan.checks)
