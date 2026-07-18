# Memory Graph: Temporal and Candidate-Causal Structure with Belief

## 1. Goal

This directory will implement a temporal and candidate-causal graph grounded in video memory nodes:

```text
streaming video
  → bounded memory nodes
  → temporal + candidate-causal graph
  → question-conditioned belief
  → evidence action
  → real memory read
  → posterior update and replanning
```

The design is inspired by SelectStream's fixed-capacity latent evidence graph, but it does not depend on SelectStream code. No official implementation is currently available, so this project must implement its own minimal memory-graph substrate and clearly describe it as our implementation.

Video_Skills can provide video segmentation, clip schemas, L1/L2 graphs, typed skills, provenance, verifiers, and trajectory logging. It is the data and execution interface, not the belief model proposed here.

### Current prototype status

The first Phase 1/2 implementation now includes:

- `adapter.py`: converts a Video_Skills canonical example or L1 clue graph into timestamped `MemoryNode` objects;
- `embedding.py`: lazily loads the official `Qwen/Qwen3-VL-Embedding-2B` checkpoint and stores 2048-dimensional normalized node embeddings by reference;
- `graph_builder.py`: creates deterministic interval relations and sparse Top-K embedding candidate pairs;
- `contracts.py` and `atomic_events.py`: define and verify L1-grounded atomic event, entity mention, and visible-state hypotheses;
- `event_adapter.py` and `pipeline.py`: implement the only supported L1 → gate → L1.5 → relation → verifier → calibration path;
- `verifiers/`: deterministically reject ungrounded entity, state, contradiction, explanation, and enablement claims;
- `calibration.py`: fits relation-wise decision thresholds only from explicitly identified independent labels;
- `validate_vrbench.py`: measures long-video temporal order and bridge coverage without treating reasoning steps as causal-edge gold;
- `reliability.py` and `audit_l1.py`: compute hard L1 checks and require independent labels for semantic reliability gates;
- `types.py`, `memory_graph.schema.json`, and `causal_temporal_overlay.schema.json`: define separate L1 observations and L1.5 event endpoints;
- `cli.py`: builds an atomic-event overlay from one canonical example;
- `tests/test_memory_graph.py`: covers adapter selection, interval relations, relation scoring, and the embedding contract.

The repository does **not** include trained relation-head weights. Consequently, `same_entity`, `state_transition`, `explains`, `enables`, and `contradicts` must not be reported as calibrated posteriors until a teacher-labeled or human-audited relation dataset has been fitted and calibrated.

See [`VALIDATION.md`](VALIDATION.md) for the historical coarse-segment run and the new staged protocol. The historical 47/50 result remains **54.4% strict precision** and must not be presented as validation of the atomic-event overlay. The trusted video-only + independent-audit rerun is still required.

See [`L1_RELIABILITY.md`](L1_RELIABILITY.md) for the updated architecture decision: L1 remains the grounded observation graph, while atomic events and causal-temporal hypotheses live in a separate L1.5 belief overlay.

## 2. Core Distinctions

### 2.1 Memory nodes are grounded observation anchors

```text
m_i = (h_i, τ_i, p_i)
```

- `h_i`: latent embedding of a video segment;
- `τ_i`: temporal span;
- `p_i`: provenance and source-segment references;
- optional metadata: surprise, read count, merge history, and confidence.

A memory node is what the system actually stores and reads. It is not necessarily identical to one event: one node may cover multiple events, and multiple nodes may describe the same event.

### 2.2 Memory nodes are not event nodes

```text
L1 memory node = grounded observation container
L1.5 event node = atomic, revisable hypothesis grounded in one or more L1 nodes
```

The earlier `1 memory node ≈ 1 event node` prototype assumption is retired because Video-Holmes segments contain multiple actions and state changes. The overlay must allow:

- one memory node to split into multiple event hypotheses;
- multiple memory nodes to map to the same event;
- new evidence to split, merge, or remove posterior hypotheses.

### 2.3 A typed graph is not the belief itself

The explicit graph is used to:

- store grounded nodes;
- execute temporal, causal, bridge, and verification actions;
- retain provenance;
- produce training trajectories and support verifier audits.

The belief is a joint posterior over event assignments, relations, competing explanations, contradictions, and missing evidence.

### 2.4 Causality is an L1.5 hypothesis, not another temporal relation

The causal-temporal overlay keeps observation, time, and candidate causality separate:

- **L1 observations** record only what is directly visible and temporally grounded in the video. L1 must not expose `explains` or `enables` as observed facts; an unverified `causal_hint` is only a proposal for later checking.
- **Temporal relations** such as `before`, `overlaps`, and `during` answer when events occur. Temporal precedence is required for a directed candidate-causal edge, but is not evidence of causality by itself.
- **L1.5 candidate-causal relations** are revisable hypotheses between grounded atomic events. `explains` means that an earlier event helps answer why a later event occurs; `enables` means that it creates a necessary or materially facilitating condition.

For example, `a person opens a door → the person enters the room` may support `enables`, because opening the door changes a relevant precondition. By contrast, `a light turns on → the person enters the room` is only temporal unless grounded evidence establishes an explanatory mechanism or precondition.

Candidate-causal relations therefore require more than ordering or semantic similarity: they need grounded evidence plus an explicit state bridge, mechanism, or precondition. They remain probabilistic and revisable, and must not be reported as proof that event A truly caused event B.

## 3. Initial Schemas

### 3.1 MemoryNode

```yaml
MemoryNode:
  node_id: string
  video_id: string
  embedding_ref: string
  embedding_metadata:
    encoder: string
    encoder_version: string
    shape: [integer]
    dtype: string
    checksum: string
  time_span:
    start_s: float
    end_s: float
  provenance:
    source_type: string
    source_ids: [string]
    created_by: string
  source_segments: [string]
  surprise: float | null
  consolidated_from: [string]
```

Large embedding arrays should not be serialized into every graph JSON file. JSON records only `embedding_ref` and the required metadata. A runtime tensor store loads vectors using `node_id` or `embedding_ref`.

### 3.2 RelationBelief

```yaml
RelationBelief:
  edge_id: string
  src: string
  dst: string
  relation_probabilities:
    temporal_next: float
    before: float
    overlaps: float
    during: float
    same_entity: float
    state_transition: float
    explains: float
    enables: float
    contradicts: float
  direction_confidence: float
  warrant: string | null
  evidence_refs: [string]
  provenance:
    producer: string
    model: string | null
```

A causal relation should not be stored only as a hard label such as `A causes B`. Use a relation posterior:

```text
q(r_ij) = P(r_ij | h_i, h_j, Δt_ij, metadata, evidence)
```

### 3.3 BeliefState

```yaml
BeliefState:
  belief_id: string
  question_id: string
  acquired_evidence: [string]
  event_assignments: object
  relation_posteriors: [RelationBelief]
  active_hypotheses: [object]
  contradictions: [object]
  missing_roles: [string]
  frontier: [string]
  residual_uncertainty: float
  remaining_budget:
    graph_reads: integer
    planning_steps: integer
```

## 4. Embedding Representation

### 4.1 Node embedding

```text
h_i = Proj(
  visual_latent
  ⊕ text_semantics
  ⊕ entity_state
  ⊕ temporal_position
)
```

The first version should reuse clip/event representations and provenance produced by Video_Skills. If no embedding is available, it can be constructed from the corresponding video segment, caption, and state description.

### 4.2 Relation embedding

```text
e_ij = MLP(
  h_i
  ⊕ h_j
  ⊕ h_i ⊙ h_j
  ⊕ (h_j - h_i)
  ⊕ Time2Vec(Δt_ij)
  ⊕ relation_type_embedding
  ⊕ evidence_features
)
```

The structural relation prior is:

```text
q_0(r_ij) = softmax(W e_ij)
```

`q_0` remains question-independent at L1. After a question arrives and grounded evidence has been acquired, the system forms:

```text
q(r_ij | q, E_1:t)
```

Temporal order must be constrained by explicit `time_span` values. Embedding similarity cannot replace interval reasoning.

## 5. Relation Vocabulary

| Relation | Meaning | Status |
|---|---|---|
| `temporal_next` | Temporally adjacent events | Deterministic or derived |
| `before` / `overlaps` / `during` | Interval relations | Explicit temporal constraints |
| `same_entity` | Entity continuity across nodes | Probabilistic |
| `state_transition` | State change of the same entity | Probabilistic |
| `explains` / `enables` | Candidate explanatory or enabling support | Probabilistic |
| `contradicts` | Conflicting states, events, or claims | Probabilistic |
| `bridge` | Evidence role that completes a multi-hop path | Derived from structure and task |

`explains` and `enables` represent candidate causal support only. They do not identify true causal effects from observational video.

## 6. Belief Update and Navigation

Initial action vocabulary:

```text
semantic
temporal_back
temporal_forward
same_entity
candidate_cause
effect
bridge
counter
verify
stop
```

Before executing an action, the world model predicts a belief transition:

```text
b^-_(t+1) = T_phi(b_t, a_t)
```

The system executes only the first planned action and reads a real memory node:

```text
o_(t+1) = ReadGraph(G_t, a_t)
b_(t+1) = Update(b^-_(t+1), o_(t+1))
```

Imagined evidence must never enter the final answer directly. The posterior must be corrected and planning repeated after every real graph read.

## 7. Are We Performing Causal Inference?

The current method does not perform classical causal inference:

- it does not estimate average treatment effects;
- it does not use do-calculus to identify causal effects;
- it does not generate physical counterfactual videos;
- it does not treat temporal precedence as sufficient evidence of causality.

Here, a counterfactual means epistemic-action branching: execute different evidence-acquisition actions from the same belief checkpoint and compare their effects on observations, belief, and future answerability.

## 8. Are We Using a Factor Graph?

The first version will not implement a complete GTSAM-style factor graph.

It will use:

- a sparse typed graph;
- relation probabilities;
- a question-conditioned factorized belief;
- contradiction and missing-role updates;
- one- or two-step model-predictive planning.

A complete factor graph may be considered later, but it is outside the first version and its novelty claim.

## 9. Integration with Video_Skills

Reuse or adapt:

- `CanonicalVideoExample` for video, question, segment, and evidence provenance;
- `ClueMemoryGraph` for question-independent L1 nodes and edges;
- `SkillGraphRollout` for skill invocations, evidence references, verifier outputs, and execution costs.

Do not copy the existing schemas unchanged:

- dataset enums are closed;
- the L1 schema does not cover all causal edge types produced at runtime;
- retrieval reserves an `embedding` mode, but the current implementation is primarily lexical or sequential;
- the L2 rollout is an execution trace, not an uncertainty-aware belief.

The legacy graph remains `steam-causal-graph/v0.1`; new two-layer outputs use:

```text
steam-causal-overlay/v0.2
```

## 10. Running the Prototype

Build an overlay from precomputed atomic events without loading the embedding model:

```bash
python -m memory_graph.cli \
  --canonical /path/to/canonical_example.json \
  --atomic-events /path/to/atomic_events.json \
  --input-mode video_only \
  --output /path/to/causal_temporal_overlay.json
```

Build node embeddings with the official Qwen 2B multimodal embedding checkpoint:

```bash
python -m memory_graph.cli \
  --canonical /path/to/canonical_example.json \
  --extract-events \
  --embedding-output /path/to/event_embeddings.npy \
  --output /path/to/causal_temporal_overlay.json \
  --input-mode expert_demo \
  --allow-provisional-expert-demo \
  --device cuda:0
```

The embedding path requires:

```text
sentence-transformers
torch
transformers>=4.57.0
qwen-vl-utils>=0.0.14
numpy
```

Add `--relation-head /path/to/relation_head.json` only after relation weights have been trained. The file contains per-relation linear weights and biases:

```json
{
  "producer": "relation-head-v1",
  "calibrated": false,
  "weights": {
    "same_entity": {
      "cosine_similarity": 2.0,
      "temporal_proximity": 0.5
    }
  },
  "biases": {
    "same_entity": -1.0
  }
}
```

An uncalibrated head is stored as `uncalibrated_prior`, never as a posterior.

Run the L1 gate independently:

```bash
python -m memory_graph.audit_l1 \
  --canonical /path/to/canonical_example.json \
  --human-audit /path/to/independent_l1_labels.json \
  --output /path/to/l1_reliability_report.json
```

Without `--human-audit`, deterministic checks still run, but the result is `incomplete`. In `video_only` mode, probabilistic relations are blocked unless `--l1-human-audit` passes. `expert_demo` can produce relations only with `--allow-provisional-expert-demo`, and the output is permanently marked provisional.

Run VRBench in two stages after Video_Skills has generated video-only canonical files:

```bash
python -m memory_graph.validate_vrbench \
  --mode video_only \
  --phase smoke \
  --canonical-dir /path/to/vrbench_video_only_canonical \
  --output /path/to/vrbench_smoke_20.json
```

Use `--phase locked` for the fixed 100-example run. VRBench evaluates timestamp coverage, temporal order, and interior bridge recall only; Video-Holmes independent edge audits remain responsible for candidate-causal precision.

## 11. First Implementation Phase

1. Audit video-only L1 observations for factuality, coverage, atomicity, provenance, and leakage.
2. Split coarse L1 observations into atomic L1.5 event hypotheses while preserving L1 references.
3. Build a deterministic temporal skeleton from explicit time spans.
4. Evaluate sparse candidate-pair recall separately from relation precision.
5. Predict constrained `same_entity`, `state_transition`, `explains`, `enables`, and `contradicts` hypotheses.
6. Reject edges that fail temporal, entity, state, grounding, or minimal-support checks.
7. Calibrate relation scores against independent human labels.
8. Implement question-conditioned belief and real graph-read updates.
9. Record sibling action trajectories from the same checkpoint.
10. Train transition, observation, and delayed-utility heads.
11. Test whether one- or two-step MPC outperforms greedy retrieval and direct action ranking.

## 12. Go / No-Go Criteria

Go:

- the L1 reliability gates in [`L1_RELIABILITY.md`](L1_RELIABILITY.md) pass on an independently audited video-only set;
- manually audited candidate-relation precision is at least 70%;
- at least 20% of valid questions require a bridge or delayed-utility hop;
- under the same graph-read budget, compared with greedy and direct-ranking baselines:
  - supporting-event Recall@K improves by at least 10 absolute points, or
  - answer accuracy improves by at least 5 absolute points;
- shuffling temporal/causal edges or transition targets materially reduces the gain.

No-Go:

- most questions can be solved by direct semantic retrieval;
- candidate causal edges are too noisy;
- direct ranking or ordinary beam search matches world-model planning;
- gains disappear under equal retrieval budgets;
- edge shuffling has no effect.
