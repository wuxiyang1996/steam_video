"""Compile real CausalTemporalOverlay relation hypotheses into GTSAM graphs."""

from __future__ import annotations

from dataclasses import dataclass
import time
from memory_graph.types import CausalTemporalOverlay
from steam_video_new.implicit_world_model.l15_graph_navigator.factor_graph import (
    BinaryFactorGraph,
    PairwiseFactor,
    UnaryFactor,
    _build_graph,
)

from .categorical import BeliefLabel, project_probability
from .gtsam_backend import GTSAMDiscreteBeliefGraph
from .measurement import MeasurementCalibrationRegistry, MeasurementJournal


@dataclass(frozen=True)
class OverlayInferenceResult:
    posteriors: dict[str, float]
    categorical: dict[str, BeliefLabel]
    map_assignment: dict[str, int]
    variable_count: int
    factor_count: int
    component_count: int
    max_component_variables: int
    verified_measurement_targets: tuple[str, ...]
    elapsed_seconds: float


@dataclass(frozen=True)
class OverlayParityResult:
    gtsam: OverlayInferenceResult
    python_bp_posteriors: dict[str, float]
    max_absolute_error: float
    mean_absolute_error: float
    categorical_mismatches: tuple[str, ...]


class GTSAMOverlayAdapter:
    """Run exact GTSAM inference independently on factor connected components.

    The existing compatibility backend currently produces only unary and
    pairwise binary factors. Splitting disconnected components is exact and
    prevents accidental construction of a 2**N joint table for a real overlay.
    """

    name = "gtsam_overlay_components/v0.1"

    def __init__(self, *, max_component_variables: int = 20) -> None:
        if max_component_variables <= 0:
            raise ValueError("max_component_variables must be positive")
        self.max_component_variables = max_component_variables

    def infer(
        self,
        overlay: CausalTemporalOverlay,
        *,
        activate_verified_measurements: bool,
        include_pairwise_factors: bool = True,
        shuffle_verified_measurements: bool = False,
        measurement_journal: MeasurementJournal | None = None,
        calibration: MeasurementCalibrationRegistry | None = None,
    ) -> OverlayInferenceResult:
        started = time.perf_counter()
        graph, measurement_targets = self.prepare_binary_graph(
            overlay,
            activate_verified_measurements=activate_verified_measurements,
            include_pairwise_factors=include_pairwise_factors,
            shuffle_verified_measurements=shuffle_verified_measurements,
            measurement_journal=measurement_journal,
            calibration=calibration,
        )
        components = _connected_components(graph)
        posteriors: dict[str, float] = {}
        assignments: dict[str, int] = {}
        for component in components:
            if len(component) > self.max_component_variables:
                raise ValueError(
                    f"factor component has {len(component)} variables; limit is "
                    f"{self.max_component_variables}"
                )
            solver = GTSAMDiscreteBeliefGraph(
                max_exact_states=2 ** self.max_component_variables
            )
            for variable_id in component:
                solver.add_variable(variable_id)
                solver.add_binary_prior(
                    variable_id,
                    graph.variables[variable_id].prior_true,
                    source="overlay_relation_prior",
                )
            for factor in graph.factors:
                if not set(factor.variables) <= component:
                    continue
                if isinstance(factor, UnaryFactor):
                    solver.add_factor(
                        factor.factor_id,
                        factor.variables,
                        (factor.likelihood_false, factor.likelihood_true),
                        source=factor.source,
                    )
                else:
                    solver.add_factor(
                        factor.factor_id,
                        factor.variables,
                        tuple(value for row in factor.potential for value in row),
                        source=factor.source,
                    )
            result = solver.infer()
            posteriors.update({name: values[1] for name, values in result.marginals.items()})
            assignments.update(result.map_assignment)
        return OverlayInferenceResult(
            posteriors=posteriors,
            categorical={name: project_probability(value) for name, value in posteriors.items()},
            map_assignment=assignments,
            variable_count=len(graph.variables),
            factor_count=len(graph.factors) + len(graph.variables),
            component_count=len(components),
            max_component_variables=max((len(component) for component in components), default=0),
            verified_measurement_targets=tuple(sorted(measurement_targets)),
            elapsed_seconds=time.perf_counter() - started,
        )

    def parity(
        self,
        overlay: CausalTemporalOverlay,
        *,
        activate_verified_measurements: bool = True,
        inference_iterations: int = 8,
        measurement_journal: MeasurementJournal | None = None,
        calibration: MeasurementCalibrationRegistry | None = None,
    ) -> OverlayParityResult:
        graph, _ = self.prepare_binary_graph(
            overlay,
            activate_verified_measurements=activate_verified_measurements,
            measurement_journal=measurement_journal,
            calibration=calibration,
        )
        python_result = graph.infer(iterations=inference_iterations)
        gtsam_result = self.infer(
            overlay,
            activate_verified_measurements=activate_verified_measurements,
            measurement_journal=measurement_journal,
            calibration=calibration,
        )
        errors = {
            name: abs(value - python_result.posteriors[name])
            for name, value in gtsam_result.posteriors.items()
        }
        mismatches = tuple(
            sorted(
                name
                for name, value in python_result.posteriors.items()
                if project_probability(value) is not gtsam_result.categorical[name]
            )
        )
        return OverlayParityResult(
            gtsam=gtsam_result,
            python_bp_posteriors=python_result.posteriors,
            max_absolute_error=max(errors.values(), default=0.0),
            mean_absolute_error=(sum(errors.values()) / len(errors) if errors else 0.0),
            categorical_mismatches=mismatches,
        )

    def prepare_binary_graph(
        self,
        overlay: CausalTemporalOverlay,
        *,
        activate_verified_measurements: bool,
        include_pairwise_factors: bool = True,
        shuffle_verified_measurements: bool = False,
        measurement_journal: MeasurementJournal | None = None,
        calibration: MeasurementCalibrationRegistry | None = None,
    ) -> tuple[BinaryFactorGraph, set[str]]:
        source = _build_graph(overlay, set())
        graph = BinaryFactorGraph()
        for variable in source.variables.values():
            graph.add_variable(variable.variable_id, variable.prior_true)

        measurements = [
            factor
            for factor in source.factors
            if isinstance(factor, UnaryFactor)
            and factor.source == "hard_verified_relation"
        ]
        non_measurements = [
            factor
            for factor in source.factors
            if not (
                isinstance(factor, UnaryFactor)
                and factor.source == "hard_verified_relation"
            )
        ]
        for factor in non_measurements:
            if isinstance(factor, PairwiseFactor) and not include_pairwise_factors:
                continue
            _copy_factor(graph, factor)

        targets: set[str] = set()
        if activate_verified_measurements:
            variables = tuple(sorted(graph.variables))
            for index, factor in enumerate(measurements):
                target = factor.variable_id
                if shuffle_verified_measurements and len(variables) > 1:
                    target_index = (variables.index(target) + 1 + index) % len(variables)
                    target = variables[target_index]
                graph.add_unary(
                    f"{factor.factor_id}:replay:{index}",
                    target,
                    factor.likelihood_false,
                    factor.likelihood_true,
                    "persisted_hard_verifier_measurement"
                    if not shuffle_verified_measurements
                    else "shuffled_hard_verifier_measurement",
                )
                targets.add(target)
        if measurement_journal is not None:
            if calibration is None:
                raise ValueError("measurement journal requires an internal calibration")
            for measurement in measurement_journal.measurements:
                if measurement.overlay_id != overlay.overlay_id:
                    raise ValueError("measurement belongs to a different overlay")
                if measurement.calibration_version != calibration.version:
                    raise ValueError("measurement calibration version mismatch")
                if measurement.variable_id not in graph.variables:
                    raise ValueError(
                        f"measurement references unknown relation variable: "
                        f"{measurement.variable_id}"
                    )
                likelihood = calibration.lookup(measurement.outcome)
                if likelihood is None:
                    continue
                graph.add_unary(
                    f"factor:{measurement.measurement_id}",
                    measurement.variable_id,
                    likelihood.likelihood_false,
                    likelihood.likelihood_true,
                    f"executed_graph_read:{measurement.outcome.value}",
                )
                targets.add(measurement.variable_id)
        return graph, targets


def changed_variables(
    before: OverlayInferenceResult,
    after: OverlayInferenceResult,
) -> tuple[str, ...]:
    return tuple(
        sorted(
            name
            for name in before.categorical
            if before.categorical[name] is not after.categorical[name]
        )
    )


def _copy_factor(graph: BinaryFactorGraph, factor: UnaryFactor | PairwiseFactor) -> None:
    if isinstance(factor, UnaryFactor):
        graph.add_unary(
            factor.factor_id,
            factor.variable_id,
            factor.likelihood_false,
            factor.likelihood_true,
            factor.source,
        )
    else:
        graph.add_pairwise(
            factor.factor_id,
            factor.left_variable_id,
            factor.right_variable_id,
            factor.potential,
            factor.source,
        )


def _connected_components(graph: BinaryFactorGraph) -> tuple[frozenset[str], ...]:
    adjacency = {name: set() for name in graph.variables}
    for factor in graph.factors:
        variables = factor.variables
        for left in variables:
            adjacency[left].update(right for right in variables if right != left)
    components: list[frozenset[str]] = []
    visited: set[str] = set()
    for start in sorted(adjacency):
        if start in visited:
            continue
        pending = [start]
        visited.add(start)
        component: set[str] = set()
        while pending:
            current = pending.pop()
            component.add(current)
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    pending.append(neighbor)
        components.append(frozenset(component))
    return tuple(components)
