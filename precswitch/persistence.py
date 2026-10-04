from __future__ import annotations

import json
from typing import Dict

from .budget import Budget
from .engine import Engine, Mode, _Session
from .policy import SwitchPolicy
from .schemes import PrecisionScheme, StateKind
from .stream import GroupingRule


def save_artifact(engine: Engine, path: str) -> None:
    """Save the complete artifact so a rerun resumes the original session
    paths: live sessions (scheme, group, state, outputs), rule/policy
    versions, mode and budget."""
    data = {
        "mode": engine.mode.value,
        "rule_version": engine.rule.version,
        "policy": {"min_hold": engine.policy.min_hold, "version": engine.policy.version},
        "uniform_scheme": engine.uniform_scheme.name,
        "sessions": {
            sid: {
                "scheme": s.scheme.name,
                "group": s.group,
                "state": s.state,
                "migrating_state": s.migrating_state,
                "pending_target": s.pending_target,
                "pending_position": s.pending_position,
                "candidate_id": s.candidate_id,
                "last_candidate_id": s.last_candidate_id,
                "inputs_since_switch": s.inputs_since_switch,
                "outputs": s.outputs,
                "old_scheme_outputs": s.old_scheme_outputs,
            }
            for sid, s in engine.sessions.items()
        },
        "history": [
            {
                "session_id": h.session_id,
                "position": h.position,
                "outcome": h.outcome.value,
                "scheme": h.scheme,
                "group": h.group,
                "candidate_id": h.candidate_id,
            }
            for h in engine.history
        ],
        "budget": None if engine.budget is None else {"limit": engine.budget.limit},
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def load_artifact(
    path: str,
    rule: GroupingRule,
    group_schemes: Dict[str, PrecisionScheme],
) -> Engine:
    """Rebuild an engine from a saved artifact. The supplied grouping rule
    must match the artifact's rule version, otherwise the saved session
    paths were decided under different rules and cannot be endorsed."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if rule.version != data["rule_version"]:
        raise ValueError(
            "artifact sessions ran under rule %s, current rule is %s"
            % (data["rule_version"], rule.version)
        )
    policy = SwitchPolicy(
        min_hold=data["policy"]["min_hold"], version=data["policy"]["version"]
    )
    budget = None
    if data["budget"] is not None:
        budget = Budget(limit=data["budget"]["limit"])
    schemes_by_name = {sc.name: sc for sc in group_schemes.values()}
    engine = Engine(
        mode=Mode(data["mode"]),
        rule=rule,
        policy=policy,
        group_schemes=group_schemes,
        budget=budget,
        uniform_scheme=schemes_by_name.get(data["uniform_scheme"]),
    )
    from .engine import StepResult, Outcome

    for sid, s in data["sessions"].items():
        scheme = schemes_by_name[s["scheme"]]
        sess = _Session(
            session_id=sid,
            scheme=scheme,
            group=s["group"],
            state=s["state"],
            migrating_state=s.get("migrating_state"),
            pending_target=s.get("pending_target"),
            pending_position=s.get("pending_position"),
            candidate_id=s.get("candidate_id"),
            last_candidate_id=s.get("last_candidate_id"),
            inputs_since_switch=s["inputs_since_switch"],
            outputs=list(s["outputs"]),
            old_scheme_outputs=list(s["old_scheme_outputs"]),
        )
        engine.sessions[sid] = sess
        if budget is not None:
            if sess.pending_switch:
                # A switch was interrupted before commit: the migration
                # peak was genuinely retained and must be rebuilt exactly.
                target = schemes_by_name[sess.pending_target]
                # session_started accounts for the old state; register the
                # pending pair first so the target pin survives, then pin
                # the target's shared load and add the new state's cost.
                budget.session_started(sid, scheme)
                budget._pending[sid] = (scheme, target)
                budget._retain(target)
            else:
                budget.session_started(sid, scheme)
    for h in data["history"]:
        engine.history.append(StepResult(
            h["session_id"], h["position"], Outcome(h["outcome"]), h["scheme"], h["group"],
            candidate_id=h.get("candidate_id"),
        ))
    return engine
