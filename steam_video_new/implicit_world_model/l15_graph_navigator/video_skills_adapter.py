"""Execute graph-read actions through Video_Skills and emit L2-compatible logs."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys
from typing import Any

from memory_graph.navigation import GraphReadAction, NavigationActionType
from memory_graph.types import CausalTemporalOverlay, MemoryNode

from .contracts import BeliefSnapshot, GraphReadExecution, NavigationRun


_RELATION_ACTIONS = {
    NavigationActionType.TEMPORAL_BACK,
    NavigationActionType.TEMPORAL_FORWARD,
    NavigationActionType.TRACK_ENTITY,
    NavigationActionType.INSPECT_STATE_CHANGE,
    NavigationActionType.FOLLOW_DEPENDENCY,
    NavigationActionType.CANDIDATE_CAUSE,
    NavigationActionType.EFFECT,
}


class VideoSkillsL2Adapter:
    """Bridge L1.5 actions to real Video_Skills retrieval skill calls."""

    def __init__(
        self,
        video_skills_root: str | Path | None = None,
        *,
        use_video_skills_runtime: bool = True,
    ) -> None:
        default_root = Path(__file__).resolve().parents[4] / "Video_Skills"
        self.video_skills_root = Path(video_skills_root or default_root).resolve()
        self.use_video_skills_runtime = use_video_skills_runtime
        self._runtime: dict[str, Any] | None = None
        self._graphs: dict[str, dict[str, Any]] = {}

    def execute(
        self,
        belief: BeliefSnapshot,
        action: GraphReadAction,
        overlay: CausalTemporalOverlay,
    ) -> GraphReadExecution:
        by_id = {
            node.node_id: node
            for node in overlay.atomic_events + overlay.l1_observations
        }
        if self.use_video_skills_runtime:
            result = self._execute_video_skills(belief, action, overlay)
            returned_refs = tuple(str(value) for value in result.evidence_refs or [])
            skill_id = str(result.skill_id)
            skill_ok = bool(result.ok)
            failure_code = result.failure_code
            outputs = dict(result.outputs or {})
        else:
            returned_refs = action.target_ids
            skill_id = _skill_id_for_action(action)
            skill_ok = bool(returned_refs)
            failure_code = None if skill_ok else "no_graph_target"
            outputs = {"evidence_refs": list(returned_refs)}

        returned = set(returned_refs)
        if action.action_type in {
            NavigationActionType.FIND_BRIDGE,
            NavigationActionType.SEARCH_COUNTEREVIDENCE,
        }:
            observed_ids = tuple(
                node_id
                for node_id in returned_refs
                if node_id in by_id
                and node_id not in set(belief.acquired_evidence)
            )
        else:
            observed_ids = tuple(
                target
                for target in action.target_ids
                if target in returned and target in by_id
            )
        observations = tuple(by_id[node_id] for node_id in observed_ids)
        evidence_refs = _grounded_l1_refs(observations, overlay)
        empty_search_observed = (
            action.action_type is NavigationActionType.SEARCH_COUNTEREVIDENCE
            and not observations
            and self.use_video_skills_runtime
            and (failure_code == "no_counterevidence" or not returned_refs)
        )
        if empty_search_observed and action.source_id in by_id:
            evidence_refs = _grounded_l1_refs((by_id[action.source_id],), overlay)
        verifier_result = _categorical_verifier_result(
            belief=belief,
            action=action,
            observations=observations,
            skill_outputs=outputs,
            evidence_refs=evidence_refs,
            runtime_used=self.use_video_skills_runtime,
        )
        invocation = {
            "node_id": _stable_id(
                "skill",
                belief.belief_id,
                action.action_type.value,
                action.source_id,
                action.target_ids,
            ),
            "skill_id": skill_id,
            "args": {
                "action_type": action.action_type.value,
                "source_id": action.source_id,
                "target_ids": list(action.target_ids),
                "relation": action.relation,
            },
            "outputs": {
                "skill_outputs": outputs,
                "skill_returned_refs": list(returned_refs),
                "real_observation_ids": list(observed_ids),
            },
            "evidence_refs": list(evidence_refs),
            "evidence_visibility": "discovered_runtime",
            "status": (
                "executed"
                if observations or empty_search_observed
                else "insufficient" if skill_ok else "failed"
            ),
            "failure_code": failure_code,
            "search_completed": empty_search_observed or bool(observations),
            "grounded_empty_result": empty_search_observed and bool(evidence_refs),
            "verifier_result": {
                "skill_ok": skill_ok,
                "planned_target_returned": bool(observations),
                "relation_verified_before_read": _relation_verified(
                    belief, action
                ),
                **verifier_result,
            },
        }
        return GraphReadExecution(observations, invocation)

    def execute_review_anchored(
        self,
        belief: BeliefSnapshot,
        action: GraphReadAction,
        overlay: CausalTemporalOverlay,
    ) -> GraphReadExecution:
        """Execute a reviewed endpoint read without adding a relation to the graph.

        Targetless searches still use the live Video_Skills runtime.  A reviewed
        relation action whose edge was not admitted may read its explicitly
        reviewed endpoints, but the invocation records that this was an endpoint
        read rather than a graph traversal.
        """

        if action.action_type is NavigationActionType.SEARCH_COUNTEREVIDENCE:
            return self.execute(belief, action, overlay)
        if action.action_type is NavigationActionType.STOP:
            return GraphReadExecution((), {})
        by_id = {
            node.node_id: node
            for node in overlay.atomic_events + overlay.l1_observations
        }
        observed_ids = tuple(
            node_id for node_id in action.target_ids if node_id in by_id
        )
        observations = tuple(by_id[node_id] for node_id in observed_ids)
        verification_nodes = (
            ((by_id[action.source_id],) if action.source_id in by_id else ())
            + observations
        )
        evidence_refs = _grounded_l1_refs(verification_nodes, overlay)
        verifier = _review_anchored_verifier(
            action, verification_nodes, evidence_refs
        )
        invocation = {
            "node_id": _stable_id(
                "skill", belief.belief_id, "review_anchored", _action_signature(action)
            ),
            "skill_id": "review_anchored_endpoint_read",
            "args": {
                "action_type": action.action_type.value,
                "source_id": action.source_id,
                "target_ids": list(action.target_ids),
                "relation": action.relation,
            },
            "outputs": {
                "real_observation_ids": list(observed_ids),
                "graph_edge_inserted": False,
                "execution_semantics": "direct_persisted_endpoint_read",
            },
            "evidence_refs": list(evidence_refs),
            "evidence_visibility": "review_restored_endpoint",
            "status": "executed" if observations and evidence_refs else "failed",
            "failure_code": None if observations and evidence_refs else "unknown_endpoint",
            "search_completed": False,
            "grounded_empty_result": False,
            "verifier_result": verifier,
        }
        return GraphReadExecution(observations, invocation)


    def _execute_video_skills(
        self,
        belief: BeliefSnapshot,
        action: GraphReadAction,
        overlay: CausalTemporalOverlay,
    ) -> Any:
        runtime = self._load_runtime()
        graph = self._graphs.setdefault(
            overlay.overlay_id,
            overlay_to_video_skills_graph(overlay),
        )
        if action.action_type is NavigationActionType.SEMANTIC:
            return runtime["retrieve_by_event"](
                graph,
                event_description=belief.question,
            )
        if action.action_type in _RELATION_ACTIONS:
            relation = action.relation or ""
            if relation == "event_projection":
                relation = "grounded_by"
            return runtime["retrieve_by_relation"](
                graph,
                source_node=action.source_id or "missing",
                relation_type=relation,
                hop_limit=1,
            )
        if action.action_type is NavigationActionType.FIND_BRIDGE:
            return runtime["bridge_evidence_hops"](
                graph,
                source_evidence=action.source_id or list(belief.frontier),
                target_hypothesis={"claim_text": belief.question},
                max_hops=2,
            )
        if action.action_type is NavigationActionType.SEARCH_COUNTEREVIDENCE:
            return runtime["search_counterevidence"](
                graph,
                claim={"claim_text": belief.question},
                supporting_evidence=list(belief.acquired_evidence),
                search_scope=belief.question,
            )
        if action.action_type is NavigationActionType.VERIFY:
            return runtime["verify_claim_support"](
                {"claim_text": belief.question},
                evidence_chain={"evidence_refs": list(action.target_ids)},
                evidence_graph=graph,
                question_text=belief.question,
            )
        raise ValueError(f"Video_Skills cannot execute action: {action.action_type.value}")

    def _load_runtime(self) -> dict[str, Any]:
        if self._runtime is not None:
            return self._runtime
        if not self.video_skills_root.is_dir():
            raise RuntimeError(
                f"Video_Skills repository not found: {self.video_skills_root}"
            )
        root = str(self.video_skills_root)
        if root not in sys.path:
            sys.path.insert(0, root)
        from atomic_skills.reasoning_graph_assembly import (  # noqa: PLC0415
            bridge_evidence_hops,
            retrieve_by_event,
            retrieve_by_relation,
            search_counterevidence,
            verify_claim_support,
        )

        self._runtime = {
            "bridge_evidence_hops": bridge_evidence_hops,
            "retrieve_by_event": retrieve_by_event,
            "retrieve_by_relation": retrieve_by_relation,
            "search_counterevidence": search_counterevidence,
            "verify_claim_support": verify_claim_support,
        }
        return self._runtime


def _categorical_verifier_result(
    *,
    belief: BeliefSnapshot,
    action: GraphReadAction,
    observations: tuple[MemoryNode, ...],
    skill_outputs: dict[str, Any],
    evidence_refs: tuple[str, ...],
    runtime_used: bool,
) -> dict[str, Any]:
    """Project verifier behavior without exposing scores or inventing rejects."""

    if action.action_type is NavigationActionType.VERIFY:
        if not runtime_used:
            return {
                "applicability": "applicable",
                "categorical_outcome": "inconclusive",
                "source": "persisted_replay_no_post_read_verifier",
                "post_read": False,
                "evidence_refs": list(evidence_refs),
                "reasons": ["Video_Skills runtime verifier was not executed"],
                "numeric_output_exposed": False,
            }
        explicit = skill_outputs.get("categorical_outcome")
        if explicit in {"supports", "rejects", "inconclusive"}:
            outcome = str(explicit)
            reason = "runtime returned an explicit categorical verifier outcome"
        elif skill_outputs.get("passed") is True and observations and evidence_refs:
            outcome = "supports"
            reason = "runtime claim-support verifier passed on grounded evidence"
        else:
            outcome = "inconclusive"
            reason = "runtime verifier did not establish grounded support"
        return {
            "applicability": "applicable",
            "categorical_outcome": outcome,
            "source": "video_skills_post_read_claim_verifier",
            "post_read": True,
            "evidence_refs": list(evidence_refs),
            "reasons": [reason],
            "numeric_output_exposed": False,
        }
    persisted = _persisted_relation_verifier_outcome(belief, action)
    if persisted is not None:
        return {
            "applicability": "applicable",
            "categorical_outcome": persisted,
            "source": "persisted_relation_verifier",
            "post_read": False,
            "evidence_refs": list(evidence_refs),
            "reasons": ["categorical outcome existed before the read"],
            "numeric_output_exposed": False,
        }
    if action.action_type in _RELATION_ACTIONS:
        return {
            "applicability": "applicable",
            "categorical_outcome": "inconclusive",
            "source": "no_post_read_relation_verifier",
            "post_read": False,
            "evidence_refs": list(evidence_refs),
            "reasons": ["relation read has no post-read categorical verifier"],
            "numeric_output_exposed": False,
        }
    return {
        "applicability": "not_applicable",
        "categorical_outcome": "not_applicable",
        "source": "not_applicable",
        "post_read": False,
        "evidence_refs": list(evidence_refs),
        "reasons": [],
        "numeric_output_exposed": False,
    }


def _review_anchored_verifier(
    action: GraphReadAction,
    observations: tuple[MemoryNode, ...],
    evidence_refs: tuple[str, ...],
) -> dict[str, Any]:
    """Run only deterministic categorical checks on newly read endpoints."""

    state_related = action.action_type is NavigationActionType.INSPECT_STATE_CHANGE or (
        action.action_type is NavigationActionType.VERIFY
        and action.relation == "state_transition"
    )
    if state_related:
        supported = _has_explicit_state_delta(observations)
        return {
            "applicability": "applicable",
            "categorical_outcome": "supports" if supported else "inconclusive",
            "source": "review_anchored_post_read_state_verifier",
            "post_read": True,
            "evidence_refs": list(evidence_refs),
            "reasons": [
                "aligned grounded participant has an explicit attribute value change"
                if supported
                else "endpoint read did not establish a strict grounded state delta"
            ],
            "numeric_output_exposed": False,
        }
    return {
        "applicability": (
            "applicable" if action.action_type in _RELATION_ACTIONS else "not_applicable"
        ),
        "categorical_outcome": (
            "inconclusive" if action.action_type in _RELATION_ACTIONS else "not_applicable"
        ),
        "source": "review_anchored_endpoint_read_no_relation_verdict",
        "post_read": False,
        "evidence_refs": list(evidence_refs),
        "reasons": ["endpoint visibility does not itself verify the proposed relation"],
        "numeric_output_exposed": False,
    }


def _has_explicit_state_delta(observations: tuple[MemoryNode, ...]) -> bool:
    if len(observations) < 2:
        return False
    left, right = observations[0], observations[-1]
    left_meta = dict(left.metadata or {})
    right_meta = dict(right.metadata or {})
    left_people = _participant_state_index(left_meta)
    right_people = _participant_state_index(right_meta)
    for identity in set(left_people) & set(right_people):
        left_states = left_people[identity]
        right_states = right_people[identity]
        for attribute in set(left_states) & set(right_states):
            if left_states[attribute] != right_states[attribute]:
                return True
    return False


def _participant_state_index(metadata: dict[str, Any]) -> dict[tuple[str, str], dict[str, str]]:
    identities: dict[str, tuple[str, str]] = {}
    for participant in metadata.get("participants") or []:
        mention_id = str(participant.get("mention_id") or "")
        identity = (
            str(participant.get("entity_type") or "").strip().casefold(),
            str(participant.get("surface") or "").strip().casefold(),
        )
        if mention_id and all(identity):
            identities[mention_id] = identity
    indexed: dict[tuple[str, str], dict[str, str]] = {}
    for state in metadata.get("states") or []:
        identity = identities.get(str(state.get("mention_id") or ""))
        attribute = str(state.get("attribute") or "").strip().casefold()
        value = str(state.get("value") or "").strip().casefold()
        if identity and attribute and value and state.get("polarity", "positive") == "positive":
            indexed.setdefault(identity, {})[attribute] = value
    return indexed


def _persisted_relation_verifier_outcome(
    belief: BeliefSnapshot,
    action: GraphReadAction,
) -> str | None:
    for state in belief.relation_states:
        if action.source_id not in {state.src, state.dst}:
            continue
        if not any(target in {state.src, state.dst} for target in action.target_ids):
            continue
        if action.relation and action.relation in state.verified_relations:
            return "supports"
        if state.grounding.value == "contradicted":
            return "rejects"
    return None


def overlay_to_video_skills_graph(
    overlay: CausalTemporalOverlay,
) -> dict[str, Any]:
    nodes = []
    for node in overlay.l1_observations + overlay.atomic_events:
        payload = node.to_dict()
        if node.node_type == "atomic_event":
            payload["node_type"] = "event"
        nodes.append(payload)
    edges: list[dict[str, Any]] = []
    for edge in overlay.relations + overlay.l1_structural_relations:
        for relation, probability in edge.relation_probabilities.items():
            edges.append(
                {
                    "edge_id": f"{edge.edge_id}:{relation}",
                    "src": edge.src,
                    "dst": edge.dst,
                    "edge_type": relation,
                    "payload": {
                        "probability": probability,
                        "correlation_features": dict(edge.features),
                        "calibration_status": edge.status.value,
                        "direction_confidence": edge.direction_confidence,
                        "provenance": edge.provenance,
                    },
                }
            )
    l1_ids = {node.node_id for node in overlay.l1_observations}
    for event in overlay.atomic_events:
        for l1_id in event.source_segments:
            if l1_id not in l1_ids:
                continue
            edges.append(
                {
                    "edge_id": _stable_id("grounded", event.node_id, l1_id),
                    "src": event.node_id,
                    "dst": l1_id,
                    "edge_type": "grounded_by",
                    "payload": {"provenance_only": True},
                }
            )
    return {
        "schema_version": "video-skills-relaunch/v0.1",
        "graph_id": f"l15_navigation:{overlay.overlay_id}",
        "example_id": overlay.example_id,
        "video_id": overlay.video_id,
        "layer": "evidence",
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "source_overlay_id": overlay.overlay_id,
            "embedding_model": _embedding_model(overlay),
        },
    }


def build_video_skills_l2_rollout(
    run: NavigationRun,
    overlay: CausalTemporalOverlay,
    *,
    question: str,
) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    for index, step in enumerate(run.steps):
        invocation = dict(step.skill_invocation or {})
        if not invocation:
            invocation = {
                "node_id": _stable_id("skill", step.belief_before_id, "stop"),
                "skill_id": "preference_stop",
                "args": {"action_type": "stop"},
                "outputs": {"real_observation_ids": []},
                "evidence_refs": [],
                "evidence_visibility": "none",
                "status": "executed",
                "failure_code": None,
            }
        invocation.setdefault("node_id", _stable_id("skill", index))
        invocation.setdefault("evidence_refs", [])
        invocation.setdefault("status", "executed")
        nodes.append(invocation)
    edges = [
        {
            "edge_id": _stable_id("edge", left["node_id"], right["node_id"]),
            "src": left["node_id"],
            "dst": right["node_id"],
            "edge_type": "control",
            "payload": {"replanned_after_real_observation": True},
        }
        for left, right in zip(nodes, nodes[1:])
    ]
    input_mode = str(overlay.metadata.get("input_mode") or "video_only")
    if input_mode not in {"video_only", "expert_demo"}:
        input_mode = "video_only"
    return {
        "schema_version": "video-skills-relaunch/v0.1",
        "rollout_id": f"preference_rollout:{overlay.example_id}",
        "example_id": overlay.example_id,
        "rollout_source": "preference_only_l15_world_model",
        "input_mode": input_mode,
        "layer": "reasoning",
        "clue_memory_ref": {
            "graph_id": overlay.metadata.get("source_l1_graph_id")
            or overlay.overlay_id,
        },
        "question": {"question_text": question},
        "nodes": nodes,
        "edges": edges,
        "claims": [],
        "answer_support_chain": [],
        "final_answer": {"label": None, "text": None, "confidence": None},
        "verifier_summary": {
            "schema_valid": True,
            "all_commits_have_evidence": True,
            "answer_chain_valid": False,
            "timestamp_valid": True,
            "no_old_video_fact_leakage": True,
            "no_hidden_supervision_leakage": input_mode == "video_only",
        },
        "acceptance_status": "pending",
        "failure_reasons": ["answer_generation_outside_navigation_scope"],
        "preference_navigation": {
            "output_contract": "ordinal_only",
            "belief_backend": run.final_belief.backend_name,
            "trace": run.to_l2_rollout(),
        },
    }


def _skill_id_for_action(action: GraphReadAction) -> str:
    if action.action_type is NavigationActionType.SEMANTIC:
        return "retrieve_by_event"
    if action.action_type in _RELATION_ACTIONS:
        return "retrieve_by_relation"
    if action.action_type is NavigationActionType.FIND_BRIDGE:
        return "bridge_evidence_hops"
    if action.action_type is NavigationActionType.SEARCH_COUNTEREVIDENCE:
        return "search_counterevidence"
    if action.action_type is NavigationActionType.VERIFY:
        return "verify_claim_support"
    return "preference_stop"


def _action_signature(action: GraphReadAction) -> str:
    return "|".join(
        (
            action.action_type.value,
            action.source_id or "",
            ",".join(action.target_ids),
            action.relation or "",
        )
    )


def _grounded_l1_refs(
    observations: tuple[MemoryNode, ...],
    overlay: CausalTemporalOverlay,
) -> tuple[str, ...]:
    l1_ids = {node.node_id for node in overlay.l1_observations}
    refs: list[str] = []
    for node in observations:
        if node.node_id in l1_ids:
            refs.append(node.node_id)
        refs.extend(ref for ref in node.source_segments if ref in l1_ids)
    return tuple(dict.fromkeys(refs))


def _relation_verified(
    belief: BeliefSnapshot,
    action: GraphReadAction,
) -> bool:
    for state in belief.relation_states:
        if action.source_id not in {state.src, state.dst}:
            continue
        if not any(target in {state.src, state.dst} for target in action.target_ids):
            continue
        if action.relation in state.verified_relations:
            return True
        if state.calibration_status == "deterministic":
            return True
    return False


def _embedding_model(overlay: CausalTemporalOverlay) -> str | None:
    return next(
        (
            node.embedding_ref.model
            for node in overlay.atomic_events + overlay.l1_observations
            if node.embedding_ref is not None
        ),
        None,
    )


def _stable_id(prefix: str, *parts: object) -> str:
    raw = "|".join(str(part) for part in parts if part is not None)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}:{digest}"
