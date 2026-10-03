from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


class CalibrationError(Exception):
    pass


@dataclass
class Calibration:
    """Error/quality measurements backing a comparison of selection modes.

    A calibration is bound to the exact grouping-rule version and switch-
    policy version it was measured under. Changing the grouping condition
    or the hold rules produces a new comparison basis: the old calibration
    range does NOT endorse the new switching paths.
    """

    rule_version: str
    policy_version: str
    # metrics[mode][scheme_or_group] = observed error range (lo, hi)
    metrics: Dict[str, Dict[str, tuple]] = field(default_factory=dict)

    def record(self, mode: str, key: str, lo: float, hi: float) -> None:
        self.metrics.setdefault(mode, {})[key] = (lo, hi)

    def error_range(self, mode: str, key: str, rule_version: str, policy_version: str) -> tuple:
        if (rule_version, policy_version) != (self.rule_version, self.policy_version):
            raise CalibrationError(
                "calibration under rule=%s policy=%s does not cover rule=%s policy=%s; "
                "recalibrate for the new switching paths"
                % (self.rule_version, self.policy_version, rule_version, policy_version)
            )
        try:
            return self.metrics[mode][key]
        except KeyError:
            raise CalibrationError("no calibration for mode=%s key=%s" % (mode, key))

