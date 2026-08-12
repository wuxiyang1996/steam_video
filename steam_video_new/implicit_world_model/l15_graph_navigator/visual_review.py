"""Build outcome-blind visual evidence bindings for transition review packets."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from .transition_review import validate_transition_review_packet


PUBLIC_SCHEMA_VERSION = "steam-visual-transition-review/v0.1"
HIDDEN_SCHEMA_VERSION = "steam-visual-transition-review-assets/v0.1"
_PUBLIC_SCHEMA = Path(__file__).resolve().parent / "visual_review_index.schema.json"


def build_visual_review_bundle(
    packet: dict[str, Any],
    hidden_key: dict[str, Any],
    *,
    video_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Bind public review nodes to video time windows without exposing outcomes.

    ``hidden_key`` is used only to locate the source overlay. Its verifier, target,
    sampling stratum, and provenance fields are never copied to the public index.
    """

    errors = validate_transition_review_packet(packet)
    if errors or packet.get("annotation_status") != "unreviewed":
        raise ValueError(
            "visual review requires a valid unreviewed packet: "
            + "; ".join(errors[:8])
        )
    packet_sha256 = _checksum(packet)
    if hidden_key.get("packet_sha256") != packet_sha256:
        raise ValueError("hidden key does not bind to this unreviewed packet")

    public_ids = [str(item["item_id"]) for item in packet.get("items") or []]
    hidden_items = list(hidden_key.get("items") or [])
    hidden_rows = {
        str(item["item_id"]): item for item in hidden_items
    }
    if len(hidden_rows) != len(hidden_items) or set(public_ids) != set(hidden_rows):
        raise ValueError("hidden key item ids do not exactly match the packet")

    root = video_root.expanduser().resolve()
    overlay_cache: dict[Path, tuple[dict[str, Any], dict[str, dict[str, Any]]]] = {}
    duration_cache: dict[Path, float | None] = {}
    assets: dict[str, dict[str, Any]] = {}
    public_items: list[dict[str, Any]] = []
    missing_reasons: Counter[str] = Counter()
    partial_reasons: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    video_counts: Counter[str] = Counter()
    fully_covered = 0
    partially_covered = 0
    uncovered = 0

    for item in packet.get("items") or []:
        item_id = str(item["item_id"])
        hidden = hidden_rows[item_id]
        overlay_path = Path(str(hidden.get("overlay_path") or "")).expanduser().resolve()
        if not overlay_path.is_file():
            raise ValueError(f"source overlay is missing for {item_id}")
        overlay_sha256 = hashlib.sha256(overlay_path.read_bytes()).hexdigest()
        if hidden.get("overlay_sha256") != overlay_sha256:
            raise ValueError(f"source overlay checksum mismatch for {item_id}")
        if overlay_path not in overlay_cache:
            overlay = json.loads(overlay_path.read_text(encoding="utf-8"))
            nodes = {
                str(node["node_id"]): node
                for group in ("atomic_events", "l1_observations")
                for node in overlay.get(group) or []
            }
            overlay_cache[overlay_path] = (overlay, nodes)
        overlay, nodes = overlay_cache[overlay_path]
        video_id = str(overlay.get("video_id") or "")
        video_path = (root / f"{video_id}.mp4").resolve()
        if root not in video_path.parents:
            raise ValueError(f"unsafe video id in overlay: {video_id}")
        if video_path not in duration_cache:
            duration_cache[video_path] = _probe_video_duration(video_path)
        video_duration_s = duration_cache[video_path]

        action = item.get("action") or {}
        executed = item.get("executed_result") or {}
        source_id = str(action.get("source_id") or "")
        targets = {str(value) for value in action.get("target_ids") or []}
        observations = {
            str(value) for value in executed.get("real_observation_ids") or []
        }
        requested_ids: list[str] = []
        for node in (item.get("visible_context") or {}).get("nodes") or []:
            node_id = str(node.get("node_id") or "")
            if node_id and node_id not in requested_ids:
                requested_ids.append(node_id)

        visual_evidence: list[dict[str, Any]] = []
        for node_id in requested_ids:
            roles: list[str] = []
            if node_id == source_id:
                roles.append("source")
            if node_id in targets:
                roles.append("target")
            if node_id in observations:
                roles.append("observation")
            if not roles:
                roles.append("context")
            role_counts.update(roles)

            node = nodes.get(node_id)
            missing_reason: str | None = None
            start_s: float | None = None
            end_s: float | None = None
            if node is None:
                missing_reason = "node_absent_from_overlay"
            else:
                time_span = node.get("time_span") or {}
                try:
                    start_s = float(time_span["start_s"])
                    end_s = float(time_span["end_s"])
                except (KeyError, TypeError, ValueError):
                    missing_reason = "time_span_missing_or_invalid"
                if (
                    missing_reason is None
                    and (start_s is None or end_s is None or start_s < 0 or end_s <= start_s)
                ):
                    missing_reason = "time_span_missing_or_invalid"
            if missing_reason is None and not video_path.is_file():
                missing_reason = "video_file_missing"
            partial_reason: str | None = None
            if (
                missing_reason is None
                and video_duration_s is not None
                and start_s is not None
                and end_s is not None
            ):
                if start_s >= video_duration_s:
                    missing_reason = "video_window_out_of_range"
                elif end_s > video_duration_s:
                    partial_reason = "video_window_exceeds_duration"

            digest_input = f"{video_id}\0{node_id}\0{start_s}\0{end_s}"
            asset_id = "visual:" + hashlib.sha256(digest_input.encode()).hexdigest()[:24]
            if missing_reason is not None:
                availability = "missing"
                issue_reason = missing_reason
            elif partial_reason is not None:
                availability = "partial"
                issue_reason = partial_reason
            else:
                availability = "available"
                issue_reason = None
            public_asset = {
                "asset_id": asset_id,
                "node_id": node_id,
                "roles": roles,
                "video_id": video_id,
                "start_s": start_s,
                "end_s": end_s,
                "media_url": f"/api/media/{asset_id}" if availability != "missing" else None,
                "availability": availability,
                "missing_reason": issue_reason,
            }
            visual_evidence.append(public_asset)
            if missing_reason is not None:
                missing_reasons[missing_reason] += 1
            else:
                if partial_reason is not None:
                    partial_reasons[partial_reason] += 1
                video_counts[video_id] += 1
                existing = assets.get(asset_id)
                binding = {
                    "asset_id": asset_id,
                    "video_path": str(video_path),
                    "video_id": video_id,
                    "node_id": node_id,
                    "start_s": start_s,
                    "end_s": end_s,
                }
                if existing is not None and existing != binding:
                    raise ValueError(f"visual asset id collision: {asset_id}")
                assets[asset_id] = binding

        available = sum(row["availability"] != "missing" for row in visual_evidence)
        exact = sum(row["availability"] == "available" for row in visual_evidence)
        if visual_evidence and exact == len(visual_evidence):
            coverage = "full"
            fully_covered += 1
        elif available:
            coverage = "partial"
            partially_covered += 1
        else:
            coverage = "none"
            uncovered += 1
        public_items.append(
            {
                "item_id": item_id,
                "coverage": coverage,
                "visual_evidence": visual_evidence,
            }
        )

    public_index = {
        "schema_version": PUBLIC_SCHEMA_VERSION,
        "packet_id": packet["packet_id"],
        "packet_sha256": packet_sha256,
        "output_contract": "outcome_blind_visual_evidence_only",
        "items": public_items,
    }
    _assert_public_index_safe(public_index)
    public_sha256 = _checksum(public_index)
    hidden_manifest = {
        "schema_version": HIDDEN_SCHEMA_VERSION,
        "packet_id": packet["packet_id"],
        "packet_sha256": packet_sha256,
        "public_index_sha256": public_sha256,
        "warning": "Private local media bindings; never expose this file to reviewers.",
        "assets": [assets[key] for key in sorted(assets)],
    }
    total_nodes = sum(len(row["visual_evidence"]) for row in public_items)
    report = {
        "schema_version": "steam-visual-transition-review-coverage/v0.1",
        "packet_id": packet["packet_id"],
        "packet_sha256": packet_sha256,
        "public_index_sha256": public_sha256,
        "item_count": len(public_items),
        "fully_covered_item_count": fully_covered,
        "partially_covered_item_count": partially_covered,
        "uncovered_item_count": uncovered,
        "requested_node_count": total_nodes,
        "available_node_count": sum(
            asset["availability"] == "available"
            for row in public_items
            for asset in row["visual_evidence"]
        ),
        "partially_available_node_count": sum(
            asset["availability"] == "partial"
            for row in public_items
            for asset in row["visual_evidence"]
        ),
        "missing_node_count": sum(missing_reasons.values()),
        "unique_media_binding_count": len(assets),
        "role_counts": dict(sorted(role_counts.items())),
        "available_windows_by_video": dict(sorted(video_counts.items())),
        "missing_reasons": dict(sorted(missing_reasons.items())),
        "partial_reasons": dict(sorted(partial_reasons.items())),
        "video_durations_s": {
            path.stem: duration
            for path, duration in sorted(duration_cache.items(), key=lambda row: row[0].name)
            if duration is not None
        },
        "delivery": "original_mp4_http_range_with_browser_time_window",
        "pre_generated_clip_count": 0,
        "pre_generated_keyframe_count": 0,
        "formal_eligible": False,
        "training_ready": False,
        "training_performed": False,
    }
    return public_index, hidden_manifest, report


def validate_visual_review_bundle(
    packet: dict[str, Any],
    public_index: dict[str, Any],
    hidden_manifest: dict[str, Any],
) -> list[str]:
    try:
        import jsonschema
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("jsonschema is required for visual review") from exc
    schema = json.loads(_PUBLIC_SCHEMA.read_text(encoding="utf-8"))
    errors = [
        f"{'.'.join(str(part) for part in error.absolute_path) or '<root>'}: {error.message}"
        for error in sorted(
            jsonschema.Draft202012Validator(schema).iter_errors(public_index),
            key=lambda error: list(error.path),
        )
    ]
    errors.extend(validate_transition_review_packet(packet))
    packet_sha256 = _checksum(packet)
    if public_index.get("schema_version") != PUBLIC_SCHEMA_VERSION:
        errors.append("unsupported public visual index schema")
    if hidden_manifest.get("schema_version") != HIDDEN_SCHEMA_VERSION:
        errors.append("unsupported hidden visual asset schema")
    for payload, name in ((public_index, "public index"), (hidden_manifest, "hidden manifest")):
        if payload.get("packet_id") != packet.get("packet_id"):
            errors.append(f"{name} packet_id mismatch")
        if payload.get("packet_sha256") != packet_sha256:
            errors.append(f"{name} packet checksum mismatch")
    if hidden_manifest.get("public_index_sha256") != _checksum(public_index):
        errors.append("hidden manifest public index checksum mismatch")
    public_ids = [str(row.get("item_id")) for row in public_index.get("items") or []]
    packet_ids = [str(row.get("item_id")) for row in packet.get("items") or []]
    if public_ids != packet_ids:
        errors.append("public visual items do not preserve packet order and ids")
    try:
        _assert_public_index_safe(public_index)
    except ValueError as exc:
        errors.append(str(exc))
    binding_rows = list(hidden_manifest.get("assets") or [])
    bindings = {str(row.get("asset_id")): row for row in binding_rows}
    if len(bindings) != len(binding_rows):
        errors.append("hidden manifest contains duplicate asset ids")
    referenced_assets: set[str] = set()
    public_assets: dict[str, dict[str, Any]] = {}
    for item in public_index.get("items") or []:
        for asset in item.get("visual_evidence") or []:
            if asset.get("availability") in {"available", "partial"}:
                asset_id = str(asset.get("asset_id"))
                referenced_assets.add(asset_id)
                existing = public_assets.get(asset_id)
                if existing is not None and any(
                    existing.get(key) != asset.get(key)
                    for key in ("video_id", "node_id", "start_s", "end_s")
                ):
                    errors.append(f"inconsistent repeated public asset: {asset_id}")
                public_assets[asset_id] = asset
                if asset_id not in bindings:
                    errors.append(f"missing private media binding: {asset_id}")
    if set(bindings) != referenced_assets:
        errors.append("hidden manifest assets do not exactly match public available assets")
    for asset_id in sorted(set(bindings) & set(public_assets)):
        if any(
            bindings[asset_id].get(key) != public_assets[asset_id].get(key)
            for key in ("video_id", "node_id", "start_s", "end_s")
        ):
            errors.append(f"private media binding metadata mismatch: {asset_id}")
    return errors


def _assert_public_index_safe(value: Any, path: str = "public_index") -> None:
    forbidden_keys = {
        "video_path", "overlay_path", "sampling_stratum", "stored_target",
        "stored_verifier_measurement", "verifier_measurement", "target_source",
        "action_provenance", "review_status", "reward", "score", "probability",
    }
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in forbidden_keys:
                raise ValueError(f"forbidden key in public visual index: {path}.{key}")
            _assert_public_index_safe(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_public_index_safe(child, f"{path}[{index}]")
    elif isinstance(value, str):
        if value.startswith(("/fs/", "/home/", "/nfshomes/", "file://")):
            raise ValueError(f"local path leaked in public visual index: {path}")


def _checksum(payload: dict[str, Any]) -> str:
    normalized = json.loads(json.dumps(payload))
    if isinstance(normalized, dict) and "locked_sha256" in normalized:
        normalized["locked_sha256"] = None
    data = json.dumps(
        normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _probe_video_duration(video_path: Path) -> float | None:
    """Return container duration when ffprobe can read it; otherwise stay conservative."""

    if not video_path.is_file():
        return None
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(video_path),
            ],
            capture_output=True,
            check=False,
            text=True,
            timeout=15,
        )
        duration = float(result.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        duration = 0.0
    if duration > 0:
        return duration
    try:
        import cv2  # type: ignore[import-not-found]

        capture = cv2.VideoCapture(str(video_path))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        duration = frames / fps if fps > 0 and frames > 0 else 0.0
    except (ImportError, ValueError):
        return None
    return duration if duration > 0 else None
