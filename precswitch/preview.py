from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .schemes import PrecisionScheme, StateKind


class UnexplainableStateError(Exception):
    """Raised when a session holds state that cannot be accounted for
    (no scheme claims it, or it was silently dropped). Dropping such
    state and claiming continuous inference is not allowed."""


@dataclass(frozen=True)
class SwitchPreview:
    """Explains what a scheme switch would mean for a live session.

    can_continue:        state carries over (possibly via migration).
    restart_at:          explicit position from which the session must
                         restart when state cannot be carried; None if
                         continuation is possible.
    old_scheme_outputs:  positions of prior outputs still based on the
                         old scheme (they are NOT retroactively upgraded).
    """

    from_scheme: str
    to_scheme: str
    can_continue: bool
    restart_at: Optional[int]
    old_scheme_outputs: List[int] = field(default_factory=list)
    reason: str = ""


def preview_switch(
    current: PrecisionScheme,
    target: PrecisionScheme,
    state,
    next_position: int,
    old_output_positions: List[int],
) -> SwitchPreview:
    if state is not None and not _state_belongs_to(state, current):
        raise UnexplainableStateError(
            "session state %r is not explainable by scheme %s" % (state, current.name)
        )
    old_outputs = list(old_output_positions)
    if current.name == target.name:
        return SwitchPreview(current.name, target.name, True, None, old_outputs, "no switch")
    if current.can_migrate_to(target):
        return SwitchPreview(
            current.name, target.name, True, None, old_outputs,
            "state migrates from %s to %s" % (current.name, target.name),
        )
    # State cannot be carried: must restart from an explicit position.
    return SwitchPreview(
        current.name, target.name, False, next_position, old_outputs,
        "state of %s is %s and cannot migrate to %s; session must restart at position %d"
        % (current.name, current.kind.value, target.name, next_position),
    )


def _state_belongs_to(state, scheme: PrecisionScheme) -> bool:
    if scheme.kind is StateKind.STATELESS:
        return state is None
    return isinstance(state, dict) and state.get("scheme") == scheme.name

