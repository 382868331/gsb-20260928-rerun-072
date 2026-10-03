"""有状态输入流的精度方案切换。

切换决策只使用当前输入之前已经冻结的规则和已观察输入；状态迁移若没有
显式迁移器，只能在明确位置重启，绝不把旧状态无声丢弃后声称连续。
"""
import copy
import hashlib
import json

import numpy as np

from .runner import run_quant


SESSION_POLICY_FORMAT = "edgequant.session-policy.v1"


def normalize_session_policy(policy=None):
    policy = policy or {}
    min_hold = int(policy.get("min_hold_inputs", 0))
    if min_hold < 0:
        raise ValueError("min_hold_inputs 不能为负数")
    state_bits = int(policy.get("state_bits_per_session", 0))
    if state_bits < 0:
        raise ValueError("state_bits_per_session 不能为负数")
    return {
        "format": SESSION_POLICY_FORMAT,
        "min_hold_inputs": min_hold,
        "state_bits_per_session": state_bits,
        "switch_inputs": policy.get("switch_inputs", "current-only"),
        "unmatched_state_action": policy.get(
            "unmatched_state_action", "restart-at-boundary"),
        "inactive_after_inputs": policy.get("inactive_after_inputs"),
    }


def session_policy_fingerprint(policy):
    raw = json.dumps(normalize_session_policy(policy), sort_keys=True,
                     ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def plan_state_bits(plan_entry, policy):
    spec = plan_entry.get("session_state", {}) if plan_entry else {}
    return int(spec.get("state_bits", policy["state_bits_per_session"]))


def state_transition(from_plan, to_plan, migrations=None):
    if from_plan == to_plan:
        return {"from_plan": from_plan, "to_plan": to_plan,
                "compatible": True, "action": "continue",
                "explained": True,
                "description": "同一方案，直接延续内部会话状态。"}
    migrations = migrations or {}
    migrator = migrations.get("%s->%s" % (from_plan, to_plan),
                              migrations.get((from_plan, to_plan)))
    if migrator is not None:
        return {"from_plan": from_plan, "to_plan": to_plan,
                "compatible": True, "action": "migrate",
                "explained": True, "migration": migrator,
                "description": "使用已注册迁移器显式转换内部状态。"}
    return {"from_plan": from_plan, "to_plan": to_plan,
            "compatible": False, "action": "restart-at-boundary",
            "explained": True,
            "description": "两套方案内部状态不可直接互换；新方案从切换"
                           "位置重新开始，旧输出仍归属于旧方案。"}


def preview_switch(session, desired_plan, position, policy, migrations=None):
    current = session["plan"]
    held_until = int(session.get("held_until", 0))
    transition = state_transition(current, desired_plan, migrations)
    within_hold = position < held_until
    if within_hold and desired_plan != current:
        transition = copy.deepcopy(transition)
        transition.update({
            "action": "retain-min-hold",
            "compatible": False,
            "description": "仍处于最短保持范围，本输入继续使用当前方案。"})
    elif within_hold:
        transition["within_min_hold"] = True
    transition.update({
        "position": position, "within_min_hold": within_hold,
        "previous_outputs_plan": current,
        "previous_output_positions": list(
            session.get("output_positions", {}).get(current, [])),
        "restart_position": position if
        transition["action"] == "restart-at-boundary" else None})
    return transition


class SessionDeploymentRuntime:
    def __init__(self, deployment, policy=None, migrations=None):
        self.deployment = deployment
        self.policy = normalize_session_policy(
            policy or deployment.get("session_policy"))
        self.migrations = migrations or {}
        self.sessions = {}
        restored = deployment.get("session_runtime", {}).get("sessions", {})
        for sid, session in restored.items():
            self.sessions[sid] = copy.deepcopy(session)

    @classmethod
    def from_dict(cls, doc, migrations=None):
        return cls(doc, doc.get("session_policy"), migrations)

    def to_dict(self):
        doc = copy.deepcopy(self.deployment)
        doc["session_runtime"] = {
            "format": "edgequant.session-runtime.v1",
            "sessions": self.sessions,
            "storage_snapshot": self.storage_snapshot()}
        return doc

    def _fresh_session(self, session_id, plan, position):
        return {
            "session_id": session_id, "plan": plan, "started_at": position,
            "last_seen_at": position,
            "held_until": position + self.policy["min_hold_inputs"],
            "active": True, "state": self._initial_state(plan, position),
            "switches": [], "rejections": [],
            "output_positions": {plan: []},
            "state_lineage": [{"plan": plan, "position": position,
                               "action": "initialize"}]}

    def _initial_state(self, plan, position):
        qmodel = self.deployment["plans"][plan]
        return {"plan": plan, "began_at": position,
                "samples_seen": 0,
                "checkpoint_sha256_16": hashlib.sha256(
                    (plan + ":" + str(position)).encode()
                ).hexdigest()[:16],
                "quant_scheme": qmodel.get("quant_scheme")}

    def _assignment(self, x):
        from .routing import route_inputs
        fallback = self.deployment.get("fallback_plan", "unified")
        assignment = route_inputs(x, self.deployment["route_policy"]["rules"],
                                  fallback)[0]
        if assignment["plan"] not in self.deployment["plans"]:
            assignment = {**assignment, "status": "fallback-plan-missing",
                          "requested_plan": assignment["plan"],
                          "plan": fallback, "fallback_used": True,
                          "resolved_by": "fallback-plan:%s" % fallback}
        return assignment

    def storage_snapshot(self):
        active = [s for s in self.sessions.values() if s["active"]]
        fallback = self.deployment.get("fallback_plan", "unified")
        loaded = sorted({s["plan"] for s in active} | {fallback})
        shared_bits = sum(self.deployment["storage"]
                          .get("plan_bits", {}).get(p, 0) for p in loaded)
        state_bits = sum(plan_state_bits(
            self.deployment["plans"].get(s["plan"], {}), self.policy)
            for s in active)
        budget = self.deployment["storage"].get("runtime_budget_bits",
                    self.deployment["storage"].get("budget_bits"))
        return {"loaded_shared_plan_bits": shared_bits,
                "active_session_state_bits": state_bits,
                "total_bits": shared_bits + state_bits,
                "loaded_plans": loaded,
                "active_sessions": len(active),
                "budget_bits": budget,
                "budget_satisfied": budget is None or
                shared_bits + state_bits <= budget}

    def can_load_for_switch(self, session, desired_plan, position):
        snapshot = self.storage_snapshot()
        plan_bits = self.deployment["storage"].get("plan_bits", {})
        loaded = {s["plan"] for s in self.sessions.values() if s["active"]}
        fallback = self.deployment.get("fallback_plan", "unified")
        loaded.add(fallback)
        extra_plan = 0 if desired_plan in loaded else plan_bits.get(
            desired_plan, 0)
        other_old_users = any(
            other is not session and other["active"] and
            other["plan"] == session["plan"]
            for other in self.sessions.values())
        releasable_plan = 0 if other_old_users else plan_bits.get(
            session["plan"], 0)
        old_state_bits = plan_state_bits(
            self.deployment["plans"].get(session["plan"], {}), self.policy)
        new_state_bits = plan_state_bits(
            self.deployment["plans"].get(desired_plan, {}), self.policy)
        projected = snapshot["total_bits"] + extra_plan - \
            releasable_plan + new_state_bits - old_state_bits
        budget = snapshot["budget_bits"]
        if budget is not None and projected > budget:
            return False, {"reason": "storage-budget-exceeded",
                          "projected_total_bits": projected,
                          "budget_bits": budget,
                          "additional_plan_bits": extra_plan,
                          "releasable_old_plan_bits": releasable_plan,
                          "additional_state_bits": new_state_bits}
        return True, None

    def process(self, x, session_id="default", position=None):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        if len(x) != 1:
            raise ValueError("会话接口一次只能处理一条输入")
        if position is None:
            position = self.sessions.get(session_id, {}).get(
                "last_seen_at", -1) + 1
        assignment = self._assignment(x)
        desired_plan = assignment["plan"]
        if session_id not in self.sessions:
            can_use, blocked = self._can_use_initial(desired_plan)
            if not can_use:
                fallback = self.deployment.get("fallback_plan", "unified")
                session = self._fresh_session(session_id, fallback, position)
                self.sessions[session_id] = session
                session["rejections"].append({"position": position,
                    "requested_plan": desired_plan, "plan": fallback,
                    **blocked})
                desired_plan = fallback
                session["plan"] = fallback
                session["state"] = self._initial_state(fallback, position)
                session["state_lineage"].append({"plan": fallback,
                    "position": position, "action": "storage-fallback"})
                assignment = {**assignment,
                              "status": "storage-blocked-fallback",
                              "requested_plan": assignment["plan"],
                              "plan": fallback, "fallback_used": True,
                              "resolved_by": blocked["reason"]}
            else:
                session = self._fresh_session(session_id, desired_plan,
                                              position)
                self.sessions[session_id] = session
        else:
            session = self.sessions[session_id]

        transition = preview_switch(session, desired_plan, position,
                                    self.policy, self.migrations)
        if desired_plan != session["plan"] and \
                transition["action"] != "retain-min-hold":
            ok, blocked = self.can_load_for_switch(
                session, desired_plan, position)
            if not ok:
                fallback = self.deployment.get("fallback_plan", "unified")
                session["rejections"].append({
                    "position": position, "requested_plan": desired_plan,
                    "retained_plan": session["plan"], **blocked})
                desired_plan = session["plan"]
                rejection = {**blocked, "position": position,
                             "requested_plan": assignment["plan"],
                             "retained_plan": session["plan"]}
                assignment = {**assignment, "plan": desired_plan,
                              "status": "storage-blocked-retain",
                              "fallback_used": desired_plan == fallback,
                              "resolved_by": blocked["reason"]}
                transition = {"from_plan": session["plan"],
                              "to_plan": assignment["plan"],
                              "action": "storage-blocked-retain",
                              **rejection}
            else:
                self._apply_switch(session, desired_plan, position,
                                   transition)
        elif transition["action"] == "retain-min-hold":
            desired_plan = session["plan"]
            assignment = {**assignment, "plan": desired_plan,
                          "status": "retained-min-hold",
                          "fallback_used": False,
                          "resolved_by": "min-hold"}

        qmodel = self.deployment["plans"][desired_plan]
        y, _, risk = run_quant(qmodel, x, return_risk=True)
        session["last_seen_at"] = position
        session.setdefault("output_positions", {}).setdefault(
            desired_plan, []).append(position)
        session["state"]["samples_seen"] = \
            session["state"].get("samples_seen", 0) + 1
        session["state"]["last_input_sha256_16"] = hashlib.sha256(
            np.ascontiguousarray(x, np.float64).tobytes()
        ).hexdigest()[:16]
        switched = transition["from_plan"] != transition["to_plan"] or \
            transition["action"] != "continue"
        return {"output": y[0], "assignment": assignment,
                "plan": desired_plan, "session_id": session_id,
                "position": position,
                "switch": transition if switched else None,
                "storage": self.storage_snapshot(), "risk": risk}

    def _can_use_initial(self, plan):
        snapshot = self.storage_snapshot()
        plan_bits = self.deployment["storage"].get("plan_bits", {})
        projected = snapshot["total_bits"] + plan_bits.get(plan, 0) + \
            plan_state_bits(self.deployment["plans"].get(plan, {}),
                            self.policy)
        budget = snapshot["budget_bits"]
        if budget is not None and projected > budget:
            return False, {"reason": "storage-budget-exceeded",
                           "projected_total_bits": projected,
                           "budget_bits": budget}
        return True, None

    def _apply_switch(self, session, desired_plan, position, transition):
        old_plan = session["plan"]
        event = {"position": position, "from_plan": old_plan,
                 "to_plan": desired_plan,
                 **{k: v for k, v in transition.items()
                    if k != "migration"}}
        if transition["action"] == "migrate":
            migrator = transition["migration"]
            session["state"] = migrator(copy.deepcopy(session["state"]),
                                        old_plan, desired_plan, position)
        else:
            session["state"] = self._initial_state(desired_plan, position)
        session["plan"] = desired_plan
        session["held_until"] = position + self.policy["min_hold_inputs"]
        session.setdefault("output_positions", {}).setdefault(
            desired_plan, [])
        session["switches"].append(event)
        session["state_lineage"].append({"plan": desired_plan,
            "position": position, "action": transition["action"]})

    def run_stream(self, stream, session_id="default"):
        outputs = []
        decisions = []
        for position, row in enumerate(stream):
            result = self.process(row, session_id=session_id,
                                  position=position)
            outputs.append(result["output"])
            decisions.append({k: result[k] for k in
                              ("assignment", "plan", "position", "switch",
                               "storage")})
        return np.asarray(outputs, dtype=np.float64), decisions


def run_independent_selection(deployment, x, policy=None):
    from .routing import route_inputs
    x = np.asarray(x, dtype=np.float64)
    fallback = deployment.get("fallback_plan", "unified")
    assignments = route_inputs(x, deployment["route_policy"]["rules"],
                               fallback)
    outputs = []
    for row, assignment in zip(x, assignments):
        plan = assignment["plan"]
        if plan not in deployment["plans"]:
            plan = fallback
        y, _, risk = run_quant(deployment["plans"][plan], row[None],
                               return_risk=True)
        outputs.append(y[0])
        assignment["state_handling"] = \
            "no-persistent-state; initialized and released per input"
        assignment["risk"] = risk
    return np.asarray(outputs, dtype=np.float64), assignments


def compare_session_selection_strategies(deployment, streams,
                                         session_policy=None,
                                         migrations=None):
    """比较统一方案、每次独立选组与保留会话状态三种方式。

    streams 为 {session_id: 二维输入}；结果只包含按当时规则得到的路径，
    不根据后续误差重写历史选择。
    """
    unified = deployment["plans"][deployment.get("fallback_plan", "unified")]
    report = {"unified": {}, "independent": {}, "stateful": {}}
    stateful = SessionDeploymentRuntime(
        deployment, policy=session_policy, migrations=migrations)
    for session_id, stream in streams.items():
        stream = np.asarray(stream, dtype=np.float64)
        y_unified, _, unified_risk = run_quant(
            unified, stream, return_risk=True)
        y_independent, independent_assignments = run_independent_selection(
            deployment, stream)
        y_stateful, stateful_decisions = stateful.run_stream(
            stream, session_id=str(session_id))
        report["unified"][str(session_id)] = {
            "plan": deployment.get("fallback_plan", "unified"),
            "output": y_unified.tolist(), "risk": unified_risk}
        report["independent"][str(session_id)] = {
            "output": y_independent.tolist(),
            "assignments": independent_assignments}
        report["stateful"][str(session_id)] = {
            "output": y_stateful.tolist(),
            "decisions": stateful_decisions}
    report["stateful_storage_snapshot"] = stateful.storage_snapshot()
    return report
