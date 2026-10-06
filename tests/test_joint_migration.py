from precswitch import (
    Budget,
    Engine,
    GroupingRule,
    InputEvent,
    JointMigrationCoordinator,
    MigrationRequest,
    Mode,
    Outcome,
    PrecisionScheme,
    StateKind,
    SwitchPolicy,
    compare_migration_strategies,
    load_artifact,
    plan_joint_migration,
    save_artifact,
)


def build_schemes(b_state_cost=30):
    a = PrecisionScheme("a", StateKind.COMPATIBLE, load_cost=20, state_cost=10)
    b = PrecisionScheme(
        "b", StateKind.COMPATIBLE, load_cost=10, state_cost=b_state_cost,
        migrators={"a": lambda st: {"scheme": "a", "history": list(st["history"])}},
    )
    a.migrators["b"] = lambda st: {"scheme": "b", "history": list(st["history"])}
    return a, b


def make_engine(limit, b_state_cost=30, interrupt_before_commit=False):
    a, b = build_schemes(b_state_cost)
    rule = GroupingRule(lambda p: "cold" if p >= 0 else "hot", version="v1")
    engine = Engine(
        Mode.STATEFUL,
        rule,
        SwitchPolicy(min_hold=1, version="p1"),
        {"cold": a, "hot": b},
        budget=Budget(limit),
        interrupt_before_commit=interrupt_before_commit,
    )
    engine.feed(InputEvent("s1", 0, 1))
    engine.feed(InputEvent("s2", 0, 2))
    return engine, a, b


def test_planning_compares_serial_and_batch_peaks_without_reserving():
    engine, _a, _b = make_engine(limit=100)
    requests = [MigrationRequest("s1", "b"), MigrationRequest("s2", "b")]
    before = engine.budget.report().total

    comparison = compare_migration_strategies(engine, requests)

    assert comparison.serial.feasible
    assert comparison.serial.peak_total == 100
    assert comparison.serial.max_waiting == 0
    assert comparison.serial.waiting_units == 0
    assert not comparison.batch.feasible
    assert comparison.batch.peak_total == 110
    assert comparison.batch.max_waiting == 1
    assert comparison.batch.waiting_units == 1
    assert "peak 110" in comparison.batch.reason
    # A plan is research only: unexecuted releases cannot be taken early.
    assert engine.budget.report().total == before


def test_no_feasible_order_reports_blocked_and_leaves_ledger_intact():
    engine, _a, _b = make_engine(limit=70)
    requests = [MigrationRequest("s1", "b"), MigrationRequest("s2", "b")]
    before = engine.budget.report()

    comparison = compare_migration_strategies(engine, requests)

    assert comparison.feasible_strategies == ()
    assert not comparison.serial.feasible and not comparison.batch.feasible
    assert engine.budget.report() == before


def test_serial_execution_shared_target_reserved_once_and_old_kept_until_last_user_commits():
    engine, _a, b = make_engine(limit=100)
    requests = [MigrationRequest("s1", "b"), MigrationRequest("s2", "b")]
    coordinator = JointMigrationCoordinator(engine, requests)

    result = coordinator.execute_serial()

    assert result.successful
    assert result.committed == ("s1", "s2")
    assert engine.sessions["s1"].scheme.name == "b"
    assert engine.sessions["s2"].scheme.name == "b"
    # Shared b load is one copy; both old a states are gone and a is unloaded.
    assert engine.budget.report().total == b.load_cost + 2 * b.state_cost


def test_batch_failure_releases_failed_candidate_and_other_migration_continues():
    engine, a, b = make_engine(limit=200)

    def fail_marked(state):
        if state.get("history") == ["broken"]:
            raise RuntimeError("checksum mismatch")
        return {"scheme": "b", "history": list(state["history"])}

    a.migrators["b"] = fail_marked
    engine.sessions["s1"].state = {"scheme": "a", "history": ["broken"]}
    before = engine.budget.report().total

    result = JointMigrationCoordinator(
        engine,
        [MigrationRequest("s1", "b"), MigrationRequest("s2", "b")],
    ).execute_batch()

    assert result.failed == ("s1",)
    assert result.committed == ("s2",)
    assert "checksum mismatch" in result.details["s1"]
    assert engine.sessions["s1"].scheme is a
    assert engine.sessions["s1"].state == {"scheme": "a", "history": ["broken"]}
    assert engine.sessions["s2"].scheme is b
    # Failed candidate's exclusive target reservation was released before s2
    # committed; steady ledger now has shared a+b and one old and one new state.
    assert engine.budget.report().total == (
        a.load_cost + b.load_cost + a.state_cost + b.state_cost)
    continued = engine.feed(InputEvent("s1", 1, 1))
    assert continued.outcome is Outcome.EMITTED and continued.scheme == "a"
    assert before == a.load_cost + 2 * a.state_cost


def test_half_finished_batch_survives_reopen_and_completion_is_idempotent(tmp_path):
    engine, _a, _b = make_engine(limit=200)
    requests = [MigrationRequest("s1", "b"), MigrationRequest("s2", "b")]
    coordinator = JointMigrationCoordinator(engine, requests)
    for request in requests:
        assert coordinator._begin(request, {})
    assert engine.budget.report().total == 110
    path = str(tmp_path / "joint.json")
    save_artifact(engine, path)

    reopened = load_artifact(path, engine.rule, engine.group_schemes)
    reopened_coordinator = JointMigrationCoordinator(reopened, requests)
    assert reopened.budget.report().total == 110
    assert reopened_coordinator.commit("s1") is not None
    assert reopened_coordinator.rollback("s2") is not None

    history_depth = len(reopened.history)
    assert reopened_coordinator.commit("s1") is None
    assert reopened_coordinator.rollback("s1") is None
    assert reopened_coordinator.commit("s2") is None
    assert reopened_coordinator.rollback("s2") is None
    assert len(reopened.history) == history_depth
    assert reopened.sessions["s1"].scheme.name == "b"
    assert reopened.sessions["s2"].scheme.name == "a"
    assert reopened.budget.report().total == 20 + 10 + 30 + 10


def test_explicit_order_is_simulated_without_claiming_other_releases():
    engine, _a, _b = make_engine(limit=100)
    plan = plan_joint_migration(
        engine,
        [MigrationRequest("s1", "b"), MigrationRequest("s2", "b")],
        strategy="batch",
        order=["s2", "s1"],
    )
    assert plan.order == ("s2", "s1")
    assert not plan.feasible and plan.peak_total == 110
