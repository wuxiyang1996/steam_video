# Dataset Clip Wrapper File Purposes

Last updated: 2026-07-05

This document explains why each `dataset_clip_wrapper` folder/file exists.
Use it when adding new modules or deciding where a change belongs.

## Package Root

| Path | Purpose |
|------|---------|
| `__init__.py` | Public package exports and temporary compatibility aliases for old import paths. New code should import from bundle subpackages directly. |
| `README.md` | How to run the wrapper, staged API pipeline, repair protocol, and benchmark checks. |
| `FILE_PURPOSES.md` | This file-purpose map. |
| `module_bundles.py` | Machine-readable bundle registry used by `tests/smoke_test_module_bundles.py`. |
| `schemas.py` | Dataclasses/enums for wrapper config, clip policies, model configs, regimes, and canonical wrapper settings. |
| `dataset_graph_presets.py` | Dataset/regime defaults for clip policy, retrieval, hidden-source policy, and benchmark profiles. |
| `pipeline.py` | Lightweight canonical-example builder that does not own API/VLM orchestration. |
| `cli.py` | Simple CLI wrapper around canonical-example export. |
| `build_training_manifests.py` | Compatibility entrypoint for `manifests/build_training_manifests.py`. |
| `run_llm_pipeline.py` | Compatibility entrypoint for `runners/run_llm_pipeline.py`. |
| `run_staged_llm_pipeline.py` | Compatibility entrypoint for `runners/run_staged_llm_pipeline.py`. |
| `run_repair_protocol.py` | Compatibility entrypoint for `verification/run_repair_protocol.py`. |
| `report_l1_l2_quality.py` | Compatibility entrypoint for `verification/report_l1_l2_quality.py`. |
| `report_final_acceptance.py` | Compatibility entrypoint for `verification/report_final_acceptance.py`. |
| `report_failure_taxonomy.py` | Compatibility entrypoint for `verification/report_failure_taxonomy.py`. |
| `report_evidence_audit.py` | Compatibility entrypoint for `verification/report_evidence_audit.py`. |
| `export_expert_demos.py` | Compatibility entrypoint for `expert_demos/export_expert_demos.py`. |
| `retrofit_l2_trajectory.py` | Compatibility entrypoint for `verification/retrofit_l2_trajectory.py`. |
| `output/` | Generated API outputs, staged caches, repair artifacts, and reports. Only `.gitkeep` should be tracked. |

## `adapters/`

Dataset readers. These modules convert benchmark-specific files into
`RawDatasetItem` records. They should not build L1/L2 graphs or call models.

| Path | Purpose |
|------|---------|
| `adapters/base.py` | Adapter base class and `RawDatasetItem` contract. |
| `adapters/video_holmes.py` | Video-Holmes reader. |
| `adapters/cg_bench.py` | CG-Bench reader. |
| `adapters/vrbench.py` | VRBench reader. |
| `adapters/siv_bench.py` | SIV-Bench reader. |
| `adapters/streaming_video.py` | StreamBridge-style OVO-Bench and VideoMME readers. |
| `adapters/__init__.py` | Adapter registry and `get_adapter()`. |

## `perception/`

Video/clip input tools. These modules produce structured clip schemas or clip
references. They should never commit answers.

| Path | Purpose |
|------|---------|
| `perception/backbone.py` | Generic perception-backbone helpers and API-key loading for perception calls. |
| `perception/clip_policy.py` | Short/long/streaming clip segmentation and `ClipSpan` utilities. |
| `perception/clip_schema.py` | Qwen clip-schema producer and clip-level VLM schema contract. |
| `perception/openrouter_client.py` | OpenRouter/OpenAI-compatible JSON chat client and total timeout handling. |
| `perception/subtitles.py` | Subtitle parsing helpers. Audio/ASR is not part of the current visual-only scope unless explicitly enabled. |
| `perception/video_probe.py` | Lightweight video duration probing. |
| `perception/video_tool_backend.py` | Local `video_tools` perception backend for smoke/offline checks. |

## `l1_clue_graph/`

Layer-1 clue memory. L1 stores visible, question-agnostic video evidence and
retrieval structure. It may mark missing visual clues, but it must not turn
commonsense/background facts into visual evidence.

| Path | Purpose |
|------|---------|
| `l1_clue_graph/clue_memory.py` | Extracts `ClueMemoryGraph` from canonical examples and creates L2 rollout shells. |
| `l1_clue_graph/graph_composer.py` | Builds semantic L1 nodes/edges from clip schemas, including neighbor-local GPT-OSS composition. |
| `l1_clue_graph/graph_plan_validator.py` | Validates/coerces graph-composition skill plans. |
| `l1_clue_graph/clip_retrieval.py` | Coarse/fine clip retrieval over visual summaries and time anchors. |
| `l1_clue_graph/gate_l1_for_l2.py` | L1 quality/answerability gate before expensive L2 calls. |
| `l1_clue_graph/skill_graph_bridge.py` | Converts canonical examples into skill-graph-compatible structures. |

## `l2_reasoning_graph/`

Layer-2 reasoning. L2 consumes `question + L1 graph`, builds reasoning rollouts,
records bounded recursive trajectories, and asks verification/repair to resolve
weak evidence.

| Path | Purpose |
|------|---------|
| `l2_reasoning_graph/reasoning_planner.py` | GPT-OSS question-conditioned reasoning planner and execution over atomic skills. |
| `l2_reasoning_graph/reasoning_rollout.py` | Deterministic/base reasoning rollout builder. |
| `l2_reasoning_graph/l2_recursive_trace.py` | POMDP/Semi-MDP-compatible trajectory and repair-subgraph encoding. |
| `l2_reasoning_graph/fault_repair.py` | Local trace-level fault localization/repair for failed skill steps. |

## `verification/`

Repair, verification, and reporting. This bundle decides whether L2 can commit,
bridge, or abstain.

| Path | Purpose |
|------|---------|
| `verification/run_repair_protocol.py` | Bounded repair protocol: gap diagnosis, repair clip selection, L1 patch, audit-guided semantic patch nodes, GPT-OSS option evidence selector, verifier, bridge verifier. |
| `verification/runtime_verifier.py` | Runtime schema/evidence/leakage invariants. |
| `verification/evaluate_l1_query_memory.py` | L1 query-memory quality and answerability diagnostics. |
| `verification/evaluate_vrbench_video_only_graph.py` | VRBench-specific video-only graph evaluation helper. |
| `verification/report_l1_l2_quality.py` | Batch L1/L2 quality report with trajectory completeness fields. |
| `verification/report_final_acceptance.py` | Merges base quality and repair reports into final acceptance status. |
| `verification/report_failure_taxonomy.py` | Classifies non-accepted examples by failure stage, missing evidence type, repairability, and dataset-fit risk. |
| `verification/report_evidence_audit.py` | Uses GPT-OSS to audit non-accepted examples from packed L1/L2/repair evidence instead of heuristic labels. |
| `verification/retrofit_l2_trajectory.py` | Adds latest L2 trajectory metadata to older JSONL artifacts without re-running perception. |

## `expert_demos/`

Training-data export. This bundle turns verified video-only L1/L2/repair traces
into expert-demo candidates while keeping hidden supervision out of visible
inputs.

| Path | Purpose |
|------|---------|
| `expert_demos/export_expert_demos.py` | Exports direct, repaired, bridge, and abstain trajectories plus a demo quality report. |

## `manifests/`

Split-aware dataset manifests. This bundle decides which examples may be used
for demo gathering, prompt/dev tuning, or held-out evaluation.

| Path | Purpose |
|------|---------|
| `manifests/build_training_manifests.py` | Builds deterministic train/dev/test manifests grouped by video id to prevent cross-split video leakage. |

## `runners/`

End-to-end orchestration. Runners may connect adapters, perception, L1, L2, and
verification, but implementation details should remain in the bundle modules.

| Path | Purpose |
|------|---------|
| `runners/llm_pipeline.py` | Non-staged API/VLM pipeline for building enriched examples. |
| `runners/run_llm_pipeline.py` | CLI for `llm_pipeline.py`. |
| `runners/run_staged_llm_pipeline.py` | Resumable staged Qwen + GPT-OSS pipeline with per-stage artifacts/cache. |

## `tests/`

Executable smoke tests. These should be fast enough for local validation and
target specific contracts rather than full benchmark accuracy.

| Path | Purpose |
|------|---------|
| `tests/smoke_test_module_bundles.py` | Ensures every wrapper module is classified in `module_bundles.py`. |
| `tests/smoke_test_l2_recursive_trace.py` | Checks trajectory and repair-subgraph encoding. |
| `tests/smoke_test_two_layer_schema.py` | Validates L1/L2 schemas across datasets/regimes. |
| `tests/smoke_test_graph_compose.py` | Checks deterministic graph composition. |
| `tests/smoke_test_neighbor_vlm_l1_graph_compose.py` | Checks neighbor-local L1 graph composition contract. |
| `tests/smoke_test_vlm_l1_graph_compose.py` | Checks VLM L1 graph composition contract. |
| `tests/smoke_test_retrieval.py` | Checks coarse/fine retrieval and memory routing. |
| `tests/smoke_test_video_tools.py` | Checks local `video_tools` perception backend. |
| `tests/smoke_test_video_only_takein.py` | Checks all-dataset video-only canonicalization. |
| `tests/smoke_test_coarse_fine_graph_crafting.py` | Checks hierarchical long-video graph crafting. |
| `tests/smoke_test_reasoning_rollout.py` | Checks L2 rollout shell/contract. |
| `tests/smoke_test_multi_hop_reasoning_skills.py` | Checks multi-hop reasoning-skill execution. |
| `tests/smoke_test_skill_executor.py` | Checks atomic skill executor backend behavior. |
| `tests/smoke_test_fault_repair.py` | Checks local L2 fault repair. |
| `tests/smoke_test_export_expert_demos.py` | Checks expert-demo export and gold-field boundary. |
| `tests/smoke_test_training_manifests.py` | Checks split-aware manifest helpers and gold-field stripping. |
| `tests/smoke_test_graph_plan_validator.py` | Checks graph-composition plan validation. |
| `tests/smoke_test_long_coarse_fine_profile.py` | Checks long coarse/fine profile defaults. |
| `tests/smoke_test_short_multi_hop_profile.py` | Checks short multi-hop profile defaults. |
| `tests/smoke_test_long_retrieval_repair.py` | Checks long-video repair span selection. |
| `tests/smoke_test.py` | General canonical wrapper smoke test. |

## Placement Rules

- New video/clip input code goes in `perception/`.
- New question-blind evidence-memory code goes in `l1_clue_graph/`.
- New reasoning rollout or trajectory code goes in `l2_reasoning_graph/`.
- New answer verification, evidence repair, or acceptance reporting goes in
  `verification/`.
- New training-data/demo export code goes in `expert_demos/`.
- New split or dataset manifest code goes in `manifests/`.
- New orchestration commands go in `runners/` plus a thin compatibility
  entrypoint at the package root only when an existing public command needs it.
- New smoke tests go in `tests/` and should be added to `module_bundles.py`.
