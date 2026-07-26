"""Strict replay adapter for independently grounded raw-clip rereads."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Callable, Sequence

from memory_graph.types import MemoryNode
from memory_graph.visual_verifier import VLMClient, _sample_frame_data_uris

from .contracts import LegalGraphAction, RetainedEvidenceGraph


REREAD_SCHEMA = "steam-grounded-real-evidence-reread/v0.1"


def build_node_reread_artifact(
    *,
    graph: RetainedEvidenceGraph,
    node_ids: Sequence[str],
    video_path: str | Path,
    client: VLMClient,
    frames_per_window: int = 6,
    frame_sampler: Callable[..., tuple[list[str], list[dict[str, Any]]]] = (
        _sample_frame_data_uris
    ),
) -> dict[str, Any]:
    """Create blinded Qwen/VLM clip descriptors for executed L1 addresses."""

    if not node_ids or len(node_ids) != len(set(node_ids)):
        raise ValueError("reread node IDs must be non-empty and unique")
    path = Path(video_path).expanduser().resolve()
    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for node_id in node_ids:
        node = graph.node_by_id.get(node_id)
        if node is None:
            raise ValueError(f"reread node is absent from graph: {node_id}")
        frames, frame_records = frame_sampler(
            path,
            windows=[
                {
                    "start_s": node.time_span.start_s,
                    "end_s": node.time_span.end_s,
                    "purpose": "executed_l1_read",
                }
            ],
            frames_per_window=frames_per_window,
        )
        if not frames:
            failures.append({"node_id": node_id, "failure": "no_frames_decoded"})
            continue
        response = client.perceive(
            _node_reread_prompt(node, frame_records),
            image_urls=frames,
            system=(
                "Describe only directly visible evidence in the sampled frames. "
                "Do not infer the question, answer, identity, intent, or causality. "
                "Return strict JSON without confidence, probability, score, or utility."
            ),
        )
        try:
            descriptor, structured = _validate_node_reread(response)
        except ValueError as exc:
            failures.append({"node_id": node_id, "failure": str(exc)})
            continue
        records.append(
            {
                "node_id": node.node_id,
                "video_id": node.video_id,
                "time_span": {
                    "start_s": node.time_span.start_s,
                    "end_s": node.time_span.end_s,
                },
                "descriptor": descriptor,
                "structured_observation": structured,
                "frame_records": frame_records,
                "model": str(client.model),
                "review_status": "unreviewed",
                "hidden_question_or_answer_used": False,
            }
        )
    return {
        "schema_version": REREAD_SCHEMA,
        "source_video_path": str(path),
        "records": records,
        "failures": failures,
        "review_status": "unreviewed",
        "training_allowed": False,
        "hidden_question_or_answer_used": False,
    }


def _node_reread_prompt(
    node: MemoryNode, frame_records: Sequence[dict[str, Any]]
) -> str:
    return json.dumps(
        {
            "task": "Describe this executed L1 interval from visible frames only.",
            "existing_l1_address": {
                "node_id": node.node_id,
                "caption": node.text,
                "predicate": node.metadata.get("predicate"),
            },
            "available_frame_labels": [
                f"F{row['frame_index']}" for row in frame_records
            ],
            "required_output": {
                "only_keys": [
                    "descriptor",
                    "action_kind",
                    "participants",
                    "visible_states",
                    "state_change",
                ],
                "descriptor": "one grounded sentence",
                "action_kind": "short categorical phrase or unknown",
                "participants": "list of visible role descriptions",
                "visible_states": "list of visible state descriptions",
                "state_change": "visible change phrase or none",
            },
        },
        ensure_ascii=False,
    )


def _validate_node_reread(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    required = {
        "descriptor",
        "action_kind",
        "participants",
        "visible_states",
        "state_change",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("node reread response schema is invalid")
    descriptor = str(payload["descriptor"] or "").strip()
    if not descriptor:
        raise ValueError("node reread descriptor is empty")
    participants = payload["participants"]
    states = payload["visible_states"]
    if not isinstance(participants, list) or not isinstance(states, list):
        raise ValueError("node reread participants/states must be lists")
    structured = {
        "action_kind": str(payload["action_kind"] or "unknown").strip(),
        "participants": [str(row).strip() for row in participants if str(row).strip()],
        "states": [str(row).strip() for row in states if str(row).strip()],
        "state_change": str(payload["state_change"] or "none").strip(),
    }
    return descriptor, structured


class ArtifactBackedRealEvidenceReader:
    """Attach reviewed clip evidence only after its graph action is executed.

    The artifact is deliberately separate from L1/L1.5 and cannot add an address,
    alter a timestamp, or influence imagined rollouts and action preference.
    """

    def __init__(self, path: str | Path, *, require_reviewed: bool = True) -> None:
        self.path = Path(path).expanduser().resolve()
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != REREAD_SCHEMA:
            raise ValueError("unsupported real-evidence reread schema")
        rows = payload.get("records")
        if not isinstance(rows, list):
            raise ValueError("real-evidence reread records must be a list")
        self.records: dict[str, dict[str, Any]] = {}
        for row in rows:
            if not isinstance(row, dict) or not str(row.get("node_id") or ""):
                raise ValueError("each reread record requires a node_id")
            if require_reviewed and row.get("review_status") not in {
                "human_locked",
                "ground_truth_locked",
            }:
                raise ValueError("real-evidence reread is not independently locked")
            node_id = str(row["node_id"])
            if node_id in self.records:
                raise ValueError(f"duplicate real-evidence reread: {node_id}")
            self.records[node_id] = row
        self.audit_records: list[dict[str, Any]] = []

    def read(
        self,
        observation: MemoryNode,
        action: LegalGraphAction,
        graph: RetainedEvidenceGraph,
    ) -> MemoryNode:
        row = self.records.get(observation.node_id)
        if row is None:
            self.audit_records.append(
                {"node_id": observation.node_id, "status": "original_l1_used"}
            )
            return observation
        expected_span = row.get("time_span") or {}
        if str(row.get("video_id") or "") != observation.video_id:
            raise ValueError("reread video provenance does not match executed node")
        if (
            float(expected_span.get("start_s", -1)) != observation.time_span.start_s
            or float(expected_span.get("end_s", -1)) != observation.time_span.end_s
        ):
            raise ValueError("reread time span does not match executed node")
        if action.target_id != observation.node_id:
            raise ValueError("reread target does not match executed graph action")
        descriptor = str(row.get("descriptor") or "").strip()
        if not descriptor:
            raise ValueError("locked reread descriptor must be non-empty")
        structured = row.get("structured_observation") or {}
        if not isinstance(structured, dict):
            raise ValueError("reread structured_observation must be an object")
        metadata = dict(observation.metadata)
        metadata.update(structured)
        metadata["reread_descriptor"] = descriptor
        metadata["visual_reread"] = {
            "source_artifact": str(self.path),
            "review_status": str(row["review_status"]),
            "model": str(row.get("model") or "unknown"),
            "hidden_answer_used": False,
        }
        self.audit_records.append(
            {"node_id": observation.node_id, "status": "locked_reread_used"}
        )
        return replace(observation, text=descriptor, metadata=metadata)
