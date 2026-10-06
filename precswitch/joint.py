from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .budget import BudgetExceeded
from .engine import Engine, MigrationFailed, StepResult
from .schemes import PrecisionScheme


@dataclass(frozen=True)
class MigrationRequest:
    session_id: str
    target_name: str


@dataclass(frozen=True)
class PlannedStep:
    session_id: str
    target_name: str
    peak_total: int
    total_after_commit: Optional[int]
    waiting_after_step: int


@dataclass(frozen=True)
class JointMigrationPlan:
    strategy: str
    feasible: bool
    order: Tuple[str, ...]
    peak_total: int
    final_total: int
    limit: int
    max_waiting: int
    waiting_units: int
    steps: Tuple[PlannedStep, ...] = field(default_factory=tuple)
    reason: str = ""

    def as_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "feasible": self.feasible,
            "order": list(self.order),
            "peak_total": self.peak_total,
            "final_total": self.final_total,
            "limit": self.limit,
            "max_waiting": self.max_waiting,
            "waiting_units": self.waiting_units,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class StrategyComparison:
    serial: JointMigrationPlan
    batch: JointMigrationPlan

    @property
    def feasible_strategies(self) -> Tuple[JointMigrationPlan, ...]:
        return tuple(p for p in (self.serial, self.batch) if p.feasible)


@dataclass
class JointMigrationResult:
    committed: Tuple[str, ...] = ()
    pending: Tuple[str, ...] = ()
    failed: Tuple[str, ...] = ()
    details: Dict[str, str] = field(default_factory=dict)

    @property
    def successful(self) -> bool:
        return not self.failed


def compare_migration_strategies(
    engine: Engine,
    requests: Sequence[MigrationRequest],
) -> StrategyComparison:
    """Research serial and all-at-once batch migrations without reserving."""
    requests = _validate_requests(engine, requests)
    serial = _best_plan(engine, requests, "serial")
    batch = _best_plan(engine, requests, "batch")
    return StrategyComparison(serial=serial, batch=batch)


def plan_joint_migration(
    engine: Engine,
    requests: Sequence[MigrationRequest],
    strategy: str = "batch",
    order: Optional[Sequence[str]] = None,
) -> JointMigrationPlan:
    """Return a read-only resource plan; no target is loaded by this call."""
    validated = _validate_requests(engine, requests)
    if order is not None:
        by_id = {r.session_id: r for r in validated}
        if tuple(order) != tuple(by_id):
            missing = set(by_id) - set(order)
            extra = set(order) - set(by_id)
            if missing or extra:
                raise ValueError("invalid migration order: missing=%s extra=%s" %
                                 (sorted(missing), sorted(extra)))
        validated = [by_id[sid] for sid in order]
    if strategy == "serial":
        return _simulate(engine, validated, "serial")
    if strategy == "batch":
        return _simulate(engine, validated, "batch")
    raise ValueError("strategy must be 'serial' or 'batch'")


class JointMigrationCoordinator:
    """Executes a researched plan through the engine's two-phase ledger."""

    def __init__(self, engine: Engine, requests: Sequence[MigrationRequest]):
        self.engine = engine
        self.requests = _validate_requests(engine, requests, allow_pending=True)
        self.candidate_ids: Dict[str, int] = {}
        self.committed: List[str] = []
        self.failed: List[str] = []

    def execute_serial(self) -> JointMigrationResult:
        pending: List[str] = []
        details: Dict[str, str] = {}
        for request in self.requests:
            if not self._ensure_pending(request, pending, details):
                continue
            self._commit(request.session_id, pending, details)
        return self._result(pending, details)

    def execute_batch(self) -> JointMigrationResult:
        pending: List[str] = []
        details: Dict[str, str] = {}
        for request in self.requests:
            self._ensure_pending(request, pending, details)

        # A failed or explicitly cancelled candidate must not block candidates
        # whose old states are still intact and whose reservation still fits.
        for session_id in list(pending):
            self._commit(session_id, pending, details)
        return self._result(pending, details)

    def commit(self, session_id: str) -> Optional[StepResult]:
        sess = self.engine.sessions[session_id]
        if not sess.pending_switch:
            return None
        self.engine.resume_pending_switch(session_id)
        if session_id not in self.committed:
            self.committed.append(session_id)
        return self.engine.history[-1]

    def rollback(self, session_id: str) -> Optional[StepResult]:
        sess = self.engine.sessions[session_id]
        if not sess.pending_switch:
            return None
        self.engine.cancel_pending_switch(session_id)
        return self.engine.history[-1]

    def rollback_pending(self) -> List[StepResult]:
        return [
            result for request in self.requests
            for sid in [request.session_id]
            if (result := self.rollback(sid)) is not None
        ]

    def _ensure_pending(
        self,
        request: MigrationRequest,
        pending: List[str],
        details: Dict[str, str],
    ) -> bool:
        sess = self.engine.sessions[request.session_id]
        if sess.pending_switch:
            if sess.pending_target != request.target_name:
                details[request.session_id] = (
                    "pending target %s does not match requested %s"
                    % (sess.pending_target, request.target_name))
                self.failed.append(request.session_id)
                return False
            if request.session_id not in pending:
                pending.append(request.session_id)
            self.candidate_ids[request.session_id] = sess.candidate_id
            details[request.session_id] = (
                "existing half-finished candidate %s resumed" % sess.candidate_id)
            return True
        return self._begin(request, details, pending)

    def _begin(
        self,
        request: MigrationRequest,
        details: Dict[str, str],
        pending: Optional[List[str]] = None,
    ) -> bool:
        sess = self.engine.sessions.get(request.session_id)
        target = self.engine.schemes_by_name[request.target_name]
        if sess is None:
            details[request.session_id] = "unknown session"
            self.failed.append(request.session_id)
            return False
        if sess.pending_switch:
            details[request.session_id] = "session already has an in-flight migration"
            self.failed.append(request.session_id)
            return False
        if not sess.scheme.can_migrate_to(target):
            details[request.session_id] = (
                "%s state cannot migrate to %s" % (sess.scheme.name, target.name))
            self.failed.append(request.session_id)
            return False
        try:
            result = self.engine.begin_pending_switch(request.session_id, target)
        except (BudgetExceeded, MigrationFailed) as exc:
            candidate = sess.candidate_id
            if candidate is not None:
                self.candidate_ids[request.session_id] = candidate
            details[request.session_id] = str(exc)
            self.failed.append(request.session_id)
            return False
        self.candidate_ids[request.session_id] = sess.candidate_id
        details[request.session_id] = result.detail
        if pending is not None:
            pending.append(request.session_id)
        return True

    def _commit(
        self, session_id: str, pending: List[str], details: Dict[str, str]
    ) -> None:
        result = self.commit(session_id)
        if result is None:
            return
        pending.remove(session_id)
        details[session_id] = result.detail

    def _result(
        self, pending: List[str], details: Dict[str, str]
    ) -> JointMigrationResult:
        for session_id, reason in details.items():
            sess = self.engine.sessions.get(session_id)
            if sess is not None and sess.pending_switch and session_id not in pending:
                pending.append(session_id)
        return JointMigrationResult(
            committed=tuple(self.committed),
            pending=tuple(pending),
            failed=tuple(self.failed),
            details=dict(details),
        )


def _best_plan(
    engine: Engine, requests: Sequence[MigrationRequest], strategy: str
) -> JointMigrationPlan:
    candidates = []
    for ordered in _permutations(requests):
        plan = _simulate(engine, ordered, strategy)
        if plan.feasible:
            candidates.append(plan)
    if candidates:
        return min(candidates, key=lambda p: (p.peak_total, p.waiting_units, p.order))
    return _simulate(engine, requests, strategy)


def _permutations(requests: Sequence[MigrationRequest]):
    for ordered_ids in permutations(r.session_id for r in requests):
        by_id = {r.session_id: r for r in requests}
        yield [by_id[sid] for sid in ordered_ids]


def _simulate(
    engine: Engine,
    requests: Sequence[MigrationRequest],
    strategy: str,
) -> JointMigrationPlan:
    limit = engine.budget.limit if engine.budget is not None else None
    if engine.budget is None:
        raise ValueError("joint migration research requires a budget")
    current = engine.budget.report()
    refs = dict(engine.budget._refs)
    loaded = set(engine.budget._schemes)
    shared_cost = current.shared_scheme_cost
    state_cost = current.session_state_cost
    peak = current.total
    steps: List[PlannedStep] = []
    waiting_units = 0
    max_waiting = 0

    def scheme(name: str) -> PrecisionScheme:
        return engine.schemes_by_name[name]

    attempted = 0
    failed = 0

    def begin(request: MigrationRequest, waiting: int) -> bool:
        nonlocal shared_cost, state_cost, peak, waiting_units, max_waiting
        nonlocal attempted, failed
        attempted += 1
        target = scheme(request.target_name)
        previous_refs = refs.get(target.name, 0)
        was_loaded = target.name in loaded
        if target.name not in loaded:
            loaded.add(target.name)
            shared_cost += target.load_cost
        refs[target.name] = refs.get(target.name, 0) + 1
        state_cost += target.state_cost
        total = shared_cost + state_cost
        peak = max(peak, total)
        waiting_units += waiting
        max_waiting = max(max_waiting, waiting)
        steps.append(PlannedStep(
            request.session_id, target.name, total, None, waiting))
        if total > limit:
            failed += 1
            state_cost -= target.state_cost
            if previous_refs <= 0:
                refs.pop(target.name, None)
                if not was_loaded:
                    loaded.discard(target.name)
                    shared_cost -= target.load_cost
            else:
                refs[target.name] = previous_refs
            steps.pop()
            return False
        return True

    def commit(request: MigrationRequest) -> None:
        nonlocal shared_cost, state_cost
        sess = engine.sessions[request.session_id]
        source_name = sess.scheme.name
        target_name = request.target_name
        state_cost -= scheme(source_name).state_cost
        refs[source_name] = refs.get(source_name, 0) - 1
        if refs[source_name] <= 0:
            refs.pop(source_name, None)
            if source_name in loaded:
                loaded.discard(source_name)
                shared_cost -= scheme(source_name).load_cost
        total = shared_cost + state_cost
        steps[-1] = PlannedStep(
            request.session_id, target_name, steps[-1].peak_total, total,
            steps[-1].waiting_after_step,
        )

    if strategy == "serial":
        for request in requests:
            if not begin(request, 0):
                break
            commit(request)
    else:
        begin_results = []
        for index, request in enumerate(requests):
            begin_results.append((request, begin(request, index)))
        waiting = len(requests) - 1
        for index, (request, began) in enumerate(begin_results):
            if not began:
                continue
            commit(request)
            waiting -= 1
            max_waiting = max(max_waiting, waiting)
    final_total = shared_cost + state_cost

    feasible = peak <= limit and final_total <= limit and failed == 0
    reason = ""
    if not feasible:
        blockers = []
        if peak > limit:
            blockers.append("migration peak %d exceeds hard limit %d" % (peak, limit))
        if final_total > limit:
            blockers.append("steady footprint %d exceeds hard limit %d" % (final_total, limit))
        if attempted < len(requests):
            blockers.append("one or more candidates cannot reserve within the hard limit")
        reason = "; ".join(blockers)
    return JointMigrationPlan(
        strategy=strategy,
        feasible=feasible,
        order=tuple(r.session_id for r in requests),
        peak_total=peak,
        final_total=shared_cost + state_cost,
        limit=limit,
        max_waiting=max_waiting,
        waiting_units=waiting_units,
        steps=tuple(steps),
        reason=reason,
    )


def _validate_requests(
    engine: Engine, requests: Iterable[MigrationRequest], allow_pending: bool = False
) -> List[MigrationRequest]:
    validated = list(requests)
    if not validated:
        raise ValueError("at least one migration request is required")
    ids = [r.session_id for r in validated]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate session in joint migration requests")
    for request in validated:
        if request.session_id not in engine.sessions:
            raise ValueError("unknown session %s" % request.session_id)
        if request.target_name not in engine.schemes_by_name:
            raise ValueError("unknown target scheme %s" % request.target_name)
        sess = engine.sessions[request.session_id]
        if sess.pending_switch and not allow_pending:
            raise ValueError("session %s already has an in-flight migration" % request.session_id)
        if sess.pending_switch and sess.pending_target != request.target_name:
            raise ValueError(
                "session %s has a pending migration to %s, not %s"
                % (request.session_id, sess.pending_target, request.target_name))
    return validated
