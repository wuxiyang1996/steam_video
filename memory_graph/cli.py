"""Build an L1-grounded causal-temporal overlay over atomic events."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .adapter import load_json
from .atomic_events import StructuredAtomicEventExtractor
from .contracts import L1HumanAudit
from .embedding import Qwen3VLEmbeddingProvider
from .graph_builder import LinearRelationScorer
from .openrouter_validation import GPTOSSGraphValidator
from .pipeline import PayloadAtomicEventExtractor, build_causal_temporal_overlay
from .schema_validation import require_valid_overlay_artifact
from .visual_verifier import QwenVisualRereadProvider
from .video_skills_l1 import VideoSkillsL1AtomicEventExtractor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", required=True, help="Video_Skills CanonicalVideoExample JSON")
    parser.add_argument("--output", required=True, help="Output memory graph JSON")
    parser.add_argument(
        "--embedding-output",
        help="Optional .npy destination. Enables Qwen3-VL-Embedding-2B node encoding.",
    )
    parser.add_argument("--device", default=None, help="SentenceTransformers device, e.g. cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--relation-head",
        help="Optional JSON file containing learned relation-head weights and biases",
    )
    parser.add_argument(
        "--l1-human-audit",
        help="Independent L1 audit JSON required for trusted video-only relations",
    )
    parser.add_argument(
        "--input-mode",
        choices=["expert_demo", "video_only"],
        default="expert_demo",
    )
    event_source = parser.add_mutually_exclusive_group(required=True)
    event_source.add_argument(
        "--atomic-events",
        help="Precomputed atomic-event JSON with an events list",
    )
    event_source.add_argument(
        "--extract-events",
        action="store_true",
        help="Extract atomic events with GPT-OSS through OpenRouter",
    )
    event_source.add_argument(
        "--extract-video-l1",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    event_source.add_argument(
        "--use-video-skills-l1-events",
        action="store_true",
        help="Promote accepted semantic ClueMemoryGraph nodes without regenerating L1",
    )
    parser.add_argument(
        "--relation-teacher",
        action="store_true",
        help="Use GPT-OSS to propose structured probabilistic relations",
    )
    parser.add_argument("--keys-py", default="/fs/gamma-projects/vlm-robot/keys.py")
    parser.add_argument(
        "--video-skills-root",
        default="/fs/gamma-projects/vlm-robot/Video_Skills",
    )
    parser.add_argument(
        "--allow-provisional-expert-demo",
        action="store_true",
        help="Allow untrusted relation proposals from expert text for development only",
    )
    parser.add_argument(
        "--disable-hard-verifiers",
        action="store_true",
        help="Development ablation; never use for trusted output",
    )
    parser.add_argument(
        "--calibration",
        help="Independent-human relation calibration report from calibration.py",
    )
    parser.add_argument("--top-k-candidates", type=int, default=4)
    parser.add_argument("--max-before-neighbors", type=int, default=4)
    parser.add_argument(
        "--memory-capacity",
        type=int,
        help="Optional bounded L1.5 memory budget for SelectStream action planning",
    )
    parser.add_argument(
        "--visual-reread",
        action="store_true",
        help="Reread causal witness windows with a Qwen3.5-9B-compatible VLM",
    )
    parser.add_argument("--visual-model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--visual-frames-per-window", type=int, default=3)
    parser.add_argument(
        "--visual-api-base",
        default="http://127.0.0.1:8000/v1/chat/completions",
        help="OpenAI-compatible local VLM endpoint",
    )
    parser.add_argument(
        "--require-visual-verification",
        action="store_true",
        help="Reject explains/enables unless raw-video reread passes",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.extract_video_l1:
        raise ValueError(
            "--extract-video-l1 replacement is disabled; provide a staged "
            "Video_Skills clue_memory_graph"
        )
    canonical = load_json(args.canonical)
    human_audit = (
        L1HumanAudit.from_dict(load_json(args.l1_human_audit))
        if args.l1_human_audit
        else None
    )
    validator = None
    if args.extract_events or args.relation_teacher:
        validator = GPTOSSGraphValidator(
            keys_py_path=args.keys_py,
            video_skills_root=args.video_skills_root,
        )
    if args.atomic_events:
        event_extractor = PayloadAtomicEventExtractor(load_json(args.atomic_events))
    elif args.use_video_skills_l1_events:
        if args.input_mode != "video_only":
            raise ValueError(
                "--use-video-skills-l1-events requires --input-mode video_only"
            )
        event_extractor = VideoSkillsL1AtomicEventExtractor()
    else:
        assert validator is not None
        event_extractor = StructuredAtomicEventExtractor(
            validator.client,
            model=validator.model,
        )

    provider = None
    if args.embedding_output:
        provider = Qwen3VLEmbeddingProvider(device=args.device)

    scorer = None
    if args.relation_head:
        if provider is None:
            raise ValueError("--relation-head requires --embedding-output")
        relation_head = load_json(args.relation_head)
        scorer = _load_relation_scorer(relation_head)

    if args.relation_teacher and provider is None:
        raise ValueError("--relation-teacher requires --embedding-output")
    if args.require_visual_verification and not args.visual_reread:
        raise ValueError("--require-visual-verification requires --visual-reread")
    visual_provider = None
    if args.visual_reread:
        client_class = _load_skill_model_client(Path(args.video_skills_root))
        visual_client = client_class.from_local(
            model=args.visual_model,
            base_url=args.visual_api_base,
            max_tokens=1400,
            timeout_s=180,
        )
        if args.visual_reread:
            visual_provider = QwenVisualRereadProvider(
                client=visual_client,
                frames_per_window=args.visual_frames_per_window,
            )
    relation_thresholds: dict[str, float] = {}
    if args.calibration:
        calibration = load_json(args.calibration)
        if (
            calibration.get("labels_source") != "independent_human"
            or not calibration.get("trusted_labels")
        ):
            raise ValueError("calibration must use independent_human labels")
        for relation, report in (calibration.get("relations") or {}).items():
            if (
                isinstance(report, dict)
                and report.get("calibrated") is True
                and report.get("threshold") is not None
            ):
                relation_thresholds[str(relation)] = float(report["threshold"])
    result = build_causal_temporal_overlay(
        canonical,
        input_mode=args.input_mode,
        event_extractor=event_extractor,
        human_audit=human_audit,
        embedding_provider=provider,
        embedding_output_path=args.embedding_output,
        relation_teacher=validator if args.relation_teacher else None,
        relation_scorer=scorer,
        batch_size=args.batch_size,
        top_k_candidates=args.top_k_candidates,
        max_before_neighbors=args.max_before_neighbors,
        relation_thresholds=relation_thresholds,
        apply_hard_verifiers=not args.disable_hard_verifiers,
        visual_reread_provider=visual_provider,
        require_visual_verification=args.require_visual_verification,
        memory_capacity=args.memory_capacity,
        allow_provisional_expert_demo=args.allow_provisional_expert_demo,
    )

    payload = result.to_dict()
    require_valid_overlay_artifact(payload)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


def _load_skill_model_client(video_skills_root: Path) -> Any:
    root = str(video_skills_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from atomic_skills.skill_model_client import SkillModelClient

    return SkillModelClient


def _load_relation_scorer(payload: dict[str, Any]) -> LinearRelationScorer:
    weights = payload.get("weights")
    biases = payload.get("biases")
    if not isinstance(weights, dict) or not isinstance(biases, dict):
        raise ValueError("relation-head JSON requires object-valued weights and biases")
    return LinearRelationScorer(
        weights={
            str(relation): {str(feature): float(value) for feature, value in coefficients.items()}
            for relation, coefficients in weights.items()
            if isinstance(coefficients, dict)
        },
        biases={str(relation): float(value) for relation, value in biases.items()},
        calibrated=bool(payload.get("calibrated", False)),
        producer=str(payload.get("producer") or "memory_graph.linear_relation_scorer"),
    )


if __name__ == "__main__":
    raise SystemExit(main())
