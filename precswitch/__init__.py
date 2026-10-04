"""Precision-scheme switching over stateful session input streams."""

from .schemes import PrecisionScheme, StateKind
from .stream import InputEvent, GroupingRule, UncoveredInputError
from .policy import SwitchPolicy
from .engine import Engine, Mode, StepResult, Outcome, MigrationFailed
from .preview import SwitchPreview, UnexplainableStateError
from .budget import Budget, BudgetReport, SwitchProjection, BudgetExceeded
from .calibration import Calibration
from .persistence import save_artifact, load_artifact
