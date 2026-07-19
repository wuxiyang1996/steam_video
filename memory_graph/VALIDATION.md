# Causal-Temporal Overlay Validation

## Current protocol

The trusted pipeline now operates on two layers:

```text
video-only L1 observations
  -> independent L1 reliability gate
  -> grounded atomic L1.5 events
  -> event embeddings and candidate pairs
  -> structured relation teacher
  -> deterministic hard verifiers
  -> independently calibrated decisions
```

Three conclusion scopes must remain separate:

- **provisional expert-text**: useful only for testing extraction, graph, and verifier wiring; it cannot pass the L1 gate;
- **trusted video-only**: requires a passing independent L1 audit before any candidate-causal relation is generated;
- **benchmark scope**: Video-Holmes independent edge labels measure candidate-causal precision, while VRBench measures temporal/multi-hop transfer and does not provide complete typed causal-edge gold.

The updated runner writes `causal_temporal_overlay.json`, records L1 status and verifier rejections, and refuses to reuse the old segment-level API retry path. A 10-video expert-text development rerun and the 47-video locked rerun require new API calls; a trusted acceptance decision additionally requires independent human edge labels.

## Historical coarse-segment Video-Holmes run

## Setup

Validation used:

- Video_Skills `VideoHolmesAdapter` and canonical/L1 pipeline;
- `Qwen/Qwen3-VL-Embedding-2B` on an RTX A5000;
- 2048-dimensional normalized embeddings;
- `openai/gpt-oss-120b` through OpenRouter as the relation teacher and auditor;
- Video-Holmes inference scenes, key relationships, QA answers, and QA explanations as held-out audit evidence.

The historical relation teacher saw only timestamped segment descriptions. It did not see the question, answer, key relationships, inference scenes, or QA explanations. It did not use atomic events or hard verifiers, so these numbers are a baseline rather than evidence for the new overlay.

### 50-video parallel run

| Item | Value |
|---|---|
| Requested videos | 50 |
| Succeeded | 47 |
| Failed (malformed teacher JSON after retries) | 3 |
| Runner | GPU embedding overlapped with OpenRouter process workers |
| Outputs | `memory_graph/outputs/video_holmes_50/` |
| Jobs | `7106954` (main, cancelled after hang), `7106960` (API retry) |

Failed IDs: `033fKtGdpPc`, `8zdIX4VeBfA`, `9OhUJyF1bYo`.

## Results (47 videos)

| Metric | Value | Gate |
|---|---:|---|
| Temporal pass rate | 100% | pass |
| Audited predictions at ≥0.5 | 386 | — |
| Strictly supported | 210 | — |
| Plausible | 158 | — |
| Unsupported / contradicted | 18 | — |
| Strict precision | **54.4%** | fail (<70%) |
| Supported-or-plausible rate | 95.3% | — |
| Videos with strict ≥70% | 13 / 45 | — |
| Videos with strict ≥50% | 28 / 45 | — |
| Median per-video strict precision | 60.0% | — |

Earlier two-video pilot strict precision was 43.8%. The larger run improved absolute precision but still fails the causal gate.

### Relation mix among audited edges

| Relation | Count | Strictly supported |
|---|---:|---:|
| `same_entity` | 155 | 144 |
| `state_transition` | 91 | 41 |
| `explains` | 68 | 11 |
| `enables` | 69 | 11 |
| `contradicts` | 3 | 3 |

`same_entity` is reliable. `explains` / `enables` remain the main precision sink.

## Verdict

| Component | Status |
|---|---|
| Canonical → memory-node conversion | GO |
| Qwen3-VL-Embedding-2B encoding | GO |
| Deterministic temporal skeleton | GO |
| Candidate-causal overlay (`explains` / `enables` / loose `state_transition`) | **NO-GO** |

Do not start belief-transition or MPC training until the new post-verifier overlay reaches at least 70% strict precision on independent human labels and the video-only L1 gate passes.

## Runtime notes

- All 50 embeddings finished in about 2–3 minutes on one A5000.
- API labeling/audit dominated wall time (~15–20 minutes with retries).
- The historical runner used process workers and a disk-backed retry path. That segment-level retry path is now disabled because it bypasses atomic extraction and hard verification.

## Important Limitations

1. Teacher and auditor are the same model family (`gpt-oss-120b`); this is not independent human gold.
2. Inputs are gold segment descriptions, not video-only perception.
3. Three videos failed after JSON parse errors; they are excluded from precision.
4. Soft precision (~95%) is not an acceptance metric; only strict support counts toward the 70% gate.

## VRBench staged validation

`validate_vrbench.py` selects Event Attribution, Logical Linkage, and Implicit Inference by default. `--phase smoke` evaluates 20 examples; `--phase locked` evaluates 100 and requires `--mode video_only`. Hidden timestamped `reasoning_process` steps are used only after graph construction to score event-span coverage, temporal order consistency, and interior bridge coverage.

An expert-demo run may be used to smoke-test the evaluator, but its `trusted_gate` is always false. VRBench results must never be described as causal-edge precision.

## Atomic expert-text 10-video slice (current)

See [`outputs/video_holmes_atomic_dev_10/RESULTS_DEV_SLICE.md`](outputs/video_holmes_atomic_dev_10/RESULTS_DEV_SLICE.md).

Key provisional findings on 8 completed overlays:

- atomic extraction works (~6.6 events/L1; participant/state grounding 100%);
- hard verifiers currently accept **only** `same_entity` edges;
- all proposed `explains` / `enables` / `state_transition` / `contradicts` were rejected (quote grounding, entity mismatch, or L1-untrusted intra-span timing);
- GPT self-audit precision on surviving `same_entity` edges is high (~93%), but this is not an independent causal gate and must not unlock belief/MPC;
- a blinded 208-item independent annotation packet is ready under `independent_edge_audit/`.

## Independent edge audit

GPT-OSS self-audits are diagnostics only. Generate a blinded packet after a run:

```bash
python -m memory_graph.prepare_independent_edge_audit \
  --run-dir memory_graph/outputs/video_holmes_atomic_dev_10 \
  --output-dir memory_graph/outputs/video_holmes_atomic_dev_10/independent_edge_audit
```

Give only `annotation_packet.json` to an annotator. Keep `model_key.json`, model probabilities, verifier decisions, and GPT audits hidden until labels are locked. After annotation:

```bash
python -m memory_graph.materialize_independent_edge_audit \
  --annotation-packet /path/to/locked_annotation_packet.json \
  --model-key /path/to/model_key.json \
  --run-dir memory_graph/outputs/video_holmes_atomic_dev_10
```

Then run `evaluate_validation_outcomes --labels-source independent_human` and `calibration --labels-source independent_human`. `unclear` labels are excluded rather than treated as positive.

The historical 45 native-L1 relations use a separate blinded packet because
their endpoints are L1 observations rather than L1.5 events:

```bash
python -m memory_graph.l1_relation_audit prepare \
  --overlay memory_graph/outputs/video_holmes_video_skills_l1_graph_smoke/TNYeYwYiAag/causal_temporal_overlay.json \
  --output-dir memory_graph/outputs/video_holmes_video_skills_l1_graph_smoke/independent_l1_relation_audit

# An independent reviewer fills annotation_packet.json without opening model_key.json.
python -m memory_graph.l1_relation_audit evaluate \
  --packet /path/to/locked_annotation_packet.json \
  --output /path/to/l1_relation_precision.json
```

The gate passes only when both admitted identity and admitted
`state_transition` groups contain decided labels and reach at least 90% strict
precision. `unclear` is reported but excluded from the precision denominator.

## Atomic-event grounding audit

`audit_atomic_overlay.py` reports both containment and trusted temporal grounding. An event span lying inside a coarse L1 span is not enough to trust its finer ordering: LLM-refined timestamps are provisional unless frame-level evidence verifies them. Candidate-causal verifiers therefore reject causal direction that exists only between two events split from the same coarse L1 interval.
