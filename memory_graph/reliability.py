"""Deterministic and human-audited gates for the Video_Skills L1 substrate."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal

from .contracts import L1HumanAudit
from .types import MemoryNode


@dataclass(frozen=True)
class ReliabilityThresholds:
    provenance_completeness: float = 1.0
    hidden_supervision_leakage: float = 0.0
    timestamp_validity: float = 0.95
    grounded_event_precision: float = 0.90
    key_event_coverage: float = 0.90
    entity_link_precision: float = 0.90
    compound_event_rate: float = 0.10


@dataclass(frozen=True)
class ReliabilityMetric:
    value: float | None
    numerator: int
    denominator: int
    threshold: float
    comparison: Literal["at_least", "at_most"]
    source: Literal["deterministic", "human"]
    passed: bool | None
    note: str | None = None


@dataclass(frozen=True)
class L1ReliabilityReport:
    status: Literal["pass", "fail", "incomplete"]
    metrics: dict[str, ReliabilityMetric]
    issues: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "metrics": {
                name: asdict(metric)
                for name, metric in self.metrics.items()
            },
            "issues": list(self.issues),
        }


def audit_l1_nodes(
    nodes: list[MemoryNode],
    *,
    human_audit: L1HumanAudit | None = None,
    observation_end_s: float | None = None,
    thresholds: ReliabilityThresholds | None = None,
) -> L1ReliabilityReport:
    """Evaluate hard structural gates and optional independent semantic labels."""
    thresholds = thresholds or ReliabilityThresholds()
    issues: list[str] = []

    provenance_valid = sum(_has_complete_provenance(node) for node in nodes)
    timestamp_valid = sum(
        _has_valid_timestamp(node, observation_end_s=observation_end_s)
        for node in nodes
    )
    leaked_nodes = [node.node_id for node in nodes if _uses_hidden_supervision(node)]
    if leaked_nodes:
        issues.append(f"hidden supervision detected in nodes: {sorted(leaked_nodes)}")

    metrics = {
        "provenance_completeness": _metric(
            provenance_valid,
            len(nodes),
            thresholds.provenance_completeness,
            "at_least",
            "deterministic",
        ),
        "hidden_supervision_leakage": _metric(
            len(leaked_nodes),
            len(nodes),
            thresholds.hidden_supervision_leakage,
            "at_most",
            "deterministic",
        ),
        "timestamp_validity": _metric(
            timestamp_valid,
            len(nodes),
            thresholds.timestamp_validity,
            "at_least",
            "deterministic",
        ),
    }
    metrics.update(_human_metrics(nodes, human_audit, thresholds, issues))

    states = [metric.passed for metric in metrics.values()]
    if any(state is False for state in states):
        status: Literal["pass", "fail", "incomplete"] = "fail"
    elif any(state is None for state in states):
        status = "incomplete"
    else:
        status = "pass"
    return L1ReliabilityReport(status=status, metrics=metrics, issues=tuple(issues))


def _human_metrics(
    nodes: list[MemoryNode],
    audit: L1HumanAudit | None,
    thresholds: ReliabilityThresholds,
    issues: list[str],
) -> dict[str, ReliabilityMetric]:
    if audit is None:
        note = "independent human labels are required"
        return {
            "grounded_event_precision": _missing_metric(
                thresholds.grounded_event_precision, "at_least", note
            ),
            "key_event_coverage": _missing_metric(
                thresholds.key_event_coverage, "at_least", note
            ),
            "entity_link_precision": _missing_metric(
                thresholds.entity_link_precision, "at_least", note
            ),
            "compound_event_rate": _missing_metric(
                thresholds.compound_event_rate, "at_most", note
            ),
        }

    node_ids = {node.node_id for node in nodes}
    grounded_ids = set(audit.grounded_event_correct)
    compound_ids = set(audit.compound_event)
    node_labels_complete = grounded_ids == node_ids and compound_ids == node_ids
    if not node_labels_complete:
        issues.append("human node labels must cover exactly all audited L1 nodes")

    grounded = _metric(
        sum(audit.grounded_event_correct.values()),
        len(audit.grounded_event_correct),
        thresholds.grounded_event_precision,
        "at_least",
        "human",
        complete=node_labels_complete,
    )
    compound = _metric(
        sum(audit.compound_event.values()),
        len(audit.compound_event),
        thresholds.compound_event_rate,
        "at_most",
        "human",
        complete=node_labels_complete,
    )
    key_events_complete = bool(audit.gold_key_event_ids)
    if not key_events_complete:
        issues.append("human audit must define at least one gold key event")
    key_coverage = _metric(
        len(set(audit.covered_key_event_ids)),
        len(set(audit.gold_key_event_ids)),
        thresholds.key_event_coverage,
        "at_least",
        "human",
        complete=key_events_complete,
    )
    entity_link_endpoints = {
        endpoint
        for judgment in audit.entity_link_judgments
        for endpoint in (judgment.src, judgment.dst)
    }
    entity_links_complete = bool(audit.entity_link_judgments) and entity_link_endpoints <= node_ids
    if not entity_links_complete:
        issues.append(
            "human audit must include independently judged entity links between audited nodes"
        )
    entity_precision = _metric(
        sum(judgment.correct for judgment in audit.entity_link_judgments),
        len(audit.entity_link_judgments),
        thresholds.entity_link_precision,
        "at_least",
        "human",
        complete=entity_links_complete,
    )
    return {
        "grounded_event_precision": grounded,
        "key_event_coverage": key_coverage,
        "entity_link_precision": entity_precision,
        "compound_event_rate": compound,
    }


def _metric(
    numerator: int,
    denominator: int,
    threshold: float,
    comparison: Literal["at_least", "at_most"],
    source: Literal["deterministic", "human"],
    *,
    complete: bool = True,
) -> ReliabilityMetric:
    value = numerator / denominator if denominator else None
    if not complete or value is None:
        passed = None
    elif comparison == "at_least":
        passed = value >= threshold
    else:
        passed = value <= threshold
    return ReliabilityMetric(
        value=value,
        numerator=numerator,
        denominator=denominator,
        threshold=threshold,
        comparison=comparison,
        source=source,
        passed=passed,
        note=None if complete else "audit labels are incomplete",
    )


def _missing_metric(
    threshold: float,
    comparison: Literal["at_least", "at_most"],
    note: str,
) -> ReliabilityMetric:
    return ReliabilityMetric(
        value=None,
        numerator=0,
        denominator=0,
        threshold=threshold,
        comparison=comparison,
        source="human",
        passed=None,
        note=note,
    )


def _has_complete_provenance(node: MemoryNode) -> bool:
    required_provenance = {"source_graph_id", "source_node_type", "adapter"}
    return bool(
        node.source_node_id
        and node.source_segments
        and all(node.provenance.get(key) for key in required_provenance)
    )


def _has_valid_timestamp(node: MemoryNode, *, observation_end_s: float | None) -> bool:
    start = node.time_span.start_s
    end = node.time_span.end_s
    if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
        return False
    return observation_end_s is None or end <= observation_end_s + 1e-6


def _uses_hidden_supervision(node: MemoryNode) -> bool:
    visibility_payload = node.metadata.get("visibility")
    if isinstance(visibility_payload, dict):
        if visibility_payload.get("hidden_supervision") is True:
            return True
        if visibility_payload.get("visible_to_agent") is False:
            return True
        visibility = str(visibility_payload.get("mode") or "").strip().lower()
    else:
        visibility = str(visibility_payload or "").strip().lower()
    source_type = str(
        node.provenance.get("source_type")
        or node.metadata.get("source_type")
        or ""
    ).strip().lower()
    if visibility in {"hidden", "answer_only", "supervision_only", "gold"}:
        return True
    if source_type in {"gold_annotation", "qa_answer", "hidden_supervision"}:
        return True
    return bool(node.provenance.get("uses_hidden_supervision"))
