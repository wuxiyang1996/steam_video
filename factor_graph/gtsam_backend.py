"""Small, explicit GTSAM discrete graph used by the correction pilot.

This module never falls back to the repository's Python message-passing
backend. Importing it is safe without GTSAM, but constructing the backend
fails with an actionable error.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import math
from typing import Sequence

from .categorical import BeliefLabel, project_probability


GTSAM_AVAILABLE = importlib.util.find_spec("gtsam") is not None


@dataclass(frozen=True)
class VariableSpec:
    name: str
    key: int
    cardinality: int


@dataclass(frozen=True)
class FactorSpec:
    factor_id: str
    variable_names: tuple[str, ...]
    table: tuple[float, ...]
    source: str


@dataclass(frozen=True)
class InferenceResult:
    map_assignment: dict[str, int]
    marginals: dict[str, tuple[float, ...]]
    categorical_true: dict[str, BeliefLabel]
    variable_count: int
    factor_count: int
    normalizer: float


class GTSAMDiscreteBeliefGraph:
    """Validated wrapper around ``gtsam.DiscreteFactorGraph``.

    Factors may be appended after real observations. GTSAM then performs exact
    discrete optimization/elimination on the updated graph. This is not iSAM2:
    GTSAM 4.2.1 exposes the discrete graph in Python, but its iSAM2 API is for
    nonlinear/Gaussian state.
    """

    name = "gtsam_discrete_factor_graph/v0.1"

    def __init__(self, *, max_exact_states: int = 1_000_000) -> None:
        if not GTSAM_AVAILABLE:
            raise RuntimeError(
                "GTSAM is required. Use Python 3.12 and install "
                "factor_graph/requirements-gtsam.txt; no fallback was used."
            )
        if max_exact_states <= 0:
            raise ValueError("max_exact_states must be positive")
        import gtsam

        self._gtsam = gtsam
        self._graph = gtsam.DiscreteFactorGraph()
        self._variables: dict[str, VariableSpec] = {}
        self._factors: list[FactorSpec] = []
        self._max_exact_states = max_exact_states

    @property
    def variables(self) -> tuple[VariableSpec, ...]:
        return tuple(self._variables.values())

    @property
    def factors(self) -> tuple[FactorSpec, ...]:
        return tuple(self._factors)

    def add_variable(self, name: str, cardinality: int = 2) -> VariableSpec:
        if not name or name in self._variables:
            raise ValueError(f"variable name must be non-empty and unique: {name!r}")
        if cardinality < 2:
            raise ValueError("variable cardinality must be at least two")
        spec = VariableSpec(name=name, key=len(self._variables), cardinality=cardinality)
        self._variables[name] = spec
        self._ensure_exact_state_budget()
        return spec

    def add_factor(
        self,
        factor_id: str,
        variable_names: Sequence[str],
        table: Sequence[float],
        *,
        source: str,
    ) -> FactorSpec:
        if not factor_id or any(item.factor_id == factor_id for item in self._factors):
            raise ValueError(f"factor id must be non-empty and unique: {factor_id!r}")
        names = tuple(variable_names)
        if not names or len(names) != len(set(names)):
            raise ValueError("a factor requires unique variables")
        missing = set(names) - set(self._variables)
        if missing:
            raise ValueError(f"factor references unknown variables: {sorted(missing)}")
        expected = math.prod(self._variables[name].cardinality for name in names)
        values = tuple(float(value) for value in table)
        if len(values) != expected:
            raise ValueError(f"factor table has {len(values)} values; expected {expected}")
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("factor potentials must be finite and non-negative")
        if not any(value > 0.0 for value in values):
            raise ValueError("factor table cannot be all zero")
        keys = [
            (self._variables[name].key, self._variables[name].cardinality)
            for name in names
        ]
        self._graph.push_back(self._gtsam.DecisionTreeFactor(keys, list(values)))
        spec = FactorSpec(factor_id, names, values, source)
        self._factors.append(spec)
        return spec

    def add_binary_prior(self, name: str, probability_true: float, *, source: str) -> None:
        if not math.isfinite(probability_true) or not 0.0 < probability_true < 1.0:
            raise ValueError("binary prior must be strictly inside (0, 1)")
        self.add_factor(
            f"prior:{name}",
            (name,),
            (1.0 - probability_true, probability_true),
            source=source,
        )

    def add_binary_measurement(
        self,
        factor_id: str,
        name: str,
        *,
        likelihood_false: float,
        likelihood_true: float,
        source: str,
    ) -> None:
        """Append a calibrated factor only after its real evidence was observed."""

        self.add_factor(
            factor_id,
            (name,),
            (likelihood_false, likelihood_true),
            source=source,
        )

    def infer(self) -> InferenceResult:
        if not self._variables:
            raise ValueError("cannot infer an empty graph")
        uncovered = {
            name
            for name in self._variables
            if not any(name in factor.variable_names for factor in self._factors)
        }
        if uncovered:
            raise ValueError(f"variables without factors: {sorted(uncovered)}")

        map_values = self._graph.optimize()
        assignments = self._graph.product().enumerate()
        normalizer = sum(float(weight) for _, weight in assignments)
        if not math.isfinite(normalizer) or normalizer <= 0.0:
            raise ValueError("factor graph has zero or invalid total mass")

        marginal_mass = {
            name: [0.0] * spec.cardinality for name, spec in self._variables.items()
        }
        for values, weight in assignments:
            for name, spec in self._variables.items():
                marginal_mass[name][values[spec.key]] += float(weight)
        marginals = {
            name: tuple(value / normalizer for value in masses)
            for name, masses in marginal_mass.items()
        }
        return InferenceResult(
            map_assignment={
                name: int(map_values[spec.key]) for name, spec in self._variables.items()
            },
            marginals=marginals,
            categorical_true={
                name: project_probability(values[1])
                for name, values in marginals.items()
                if len(values) == 2
            },
            variable_count=len(self._variables),
            factor_count=len(self._factors),
            normalizer=normalizer,
        )

    def _ensure_exact_state_budget(self) -> None:
        states = math.prod(spec.cardinality for spec in self._variables.values())
        if states > self._max_exact_states:
            removed_name = next(reversed(self._variables))
            del self._variables[removed_name]
            raise ValueError(
                f"exact marginal state space {states} exceeds {self._max_exact_states}"
            )
