from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class InputEvent:
    """One input in a session's continuous stream."""

    session_id: str
    position: int
    payload: Any


class UncoveredInputError(Exception):
    """Raised when no grouping rule covers an input."""


class GroupingRule:
    """Maps an input payload to a group label (e.g. normal / high_risk).

    The rule is versioned: changing the grouping condition produces a new
    version, which invalidates calibrations and comparisons made under
    the old version.
    """

    def __init__(self, classifier: Callable[[Any], Optional[str]], version: str = "v1"):
        self._classifier = classifier
        self.version = version

    def group_of(self, payload: Any) -> str:
        group = self._classifier(payload)
        if group is None:
            raise UncoveredInputError("no group covers payload %r" % (payload,))
        return group

