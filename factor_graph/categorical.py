"""The only belief projection exposed to an LLM or preference model."""

from __future__ import annotations

from enum import Enum
import math


class BeliefLabel(str, Enum):
    REJECTED = "rejected"
    UNCERTAIN = "uncertain"
    ACCEPTED = "accepted"


def project_probability(
    probability: float,
    *,
    reject_below: float = 1.0 / 3.0,
    accept_at: float = 2.0 / 3.0,
) -> BeliefLabel:
    """Project an internal marginal to a non-numeric public belief label."""

    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be finite and in [0, 1]")
    if not 0.0 <= reject_below < accept_at <= 1.0:
        raise ValueError("categorical thresholds must be ordered inside [0, 1]")
    if probability < reject_below:
        return BeliefLabel.REJECTED
    if probability >= accept_at:
        return BeliefLabel.ACCEPTED
    return BeliefLabel.UNCERTAIN
