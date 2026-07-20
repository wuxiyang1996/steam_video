# Memory Graph: Temporal and Candidate-Causal Structure with Belief

## 1. Goal

> **Current architecture role:** this directory implements grounded evidence
> memory plus the explicit graph artifact used by graph-based baselines,
> diagnostics, teachers, and optional GTSAM correction. The target main method
> uses this fixed L1/L1.5 graph as a shared evidence substrate. Existing graph
> construction is necessary infrastructure but is not the primary IWM
> contribution; final hop selection must depend on predicted future belief.

This directory implements a temporal and predictive-dependency graph grounded
in video memory nodes, with mechanism-grounded candidate causality retained as
a smaller verified subset:

```text
streaming video
  → bounded memory nodes
  → temporal + predictive-dependency graph
  → optional verified causal witnesses
  → question-conditioned belief
  → evidence action
  → real memory read
  → posterior update and replanning
```

The design is inspired by SelectStream's fixed-capacity latent evidence graph, but it does not depend on SelectStream code. No official implementation is currently available, so this project must implement its own minimal memory-graph substrate and clearly describe it as our implementation.

Video_Skills can provide video segmentation, clip schemas, L1/L2 graphs, typed skills, provenance, verifiers, and trajectory logging. It is the data and execution interface, not the belief model proposed here.

### Graph-baseline decision: dependency-first navigation

The graph-based baseline does not require every useful event relation to be
causal. It separates three levels:

```text
observed structure:
  temporal relations, same_entity, visible state_transition

predictive navigation:
  transition_support, response_candidate, derived question bridge

verified candidate causality:
  explains / enables only when a mechanism-grounded visual witness passes
```

`transition_support` means that an earlier event supplies grounded entity or
state context useful for retrieving or anticipating a later transition. It
does not mean that the earlier event caused the later one.

`response_candidate` is a temporally local visible action-response hypothesis.
It is useful for deciding what to inspect next, but does not infer hidden
intent or causal effect.

Question bridges are derived during navigation and are never written as
question-independent observed facts. Correlation/dependency edges may choose a
real graph read, but they may not enter the final answer as evidence.

The graph-baseline navigation world model is therefore epistemic:

```text
P(next observation, belief change, answerability
  | current belief, graph-read action, dependency graph)
```

It predicts the value of reading memory, not the physical effect of
intervening on the video world. Every planned action executes a real persisted
graph/video read before belief is updated.

#### Identity, correlation, and causality are separate layers

The graph must not jump directly from text similarity or a Video_Skills edge
label to `same_entity`, `transition_support`, or causality. The required order
is:

```text
L1 observed facts
  → identity association
  → correlation / predictive dependency
  → verified candidate causality
```

**Observed structure** contains timestamps, temporal order, entity mentions,
attributes, locations, actions, and visible states. This is the immutable
evidence substrate.

**Identity association** is a data-association problem. Temporal proximity and
semantic embeddings generate candidates; they do not prove identity.
Candidates are checked using compatible entity type, stable and distinctive
attributes, spatial/trajectory continuity, interaction context, and hard
negative evidence. Missing attributes are unknown rather than contradictory.
Type conflicts, simultaneous distinct instances, impossible motion, or
incompatible stable attributes reject an association. The output progresses
from `same_instance_candidate` or `reappears_candidate` to verified
`same_object` / `same_entity`. Identity components are not transitively merged
unless the entire component remains temporally and attributively consistent.

**Correlation and predictive dependency** include co-occurrence, state
continuity, action-response candidates, and transition support. They guide
SelectStream retrieval and memory utility, but are neither answer evidence nor
explanations. `supports_observation` is evidence support and must not be
mechanically renamed `transition_support`; `reappears` and `same_object` must
not be mechanically renamed `same_entity`.

**Verified candidate causality** is a sparse overlay. An `explains` or
`enables` edge requires reliable affected-entity association, temporal order,
an observable before/after state delta, an explicit mechanism, a minimal
support set, raw-video verification, and no direct counterevidence. Without
reliable identity and state change, no causal claim is admitted.

The current implementation has the dual-layer storage contract and a strict
relation-admission gate, but it does not yet have enough grounded identity
evidence for useful coverage. A blinded GPT-5.6 provisional audit of the 45
old-flow L1 navigation priors produced 19/34 supported identity labels and 3/7
supported state-transition labels. Decided precision was 70.4% and 50.0%; the
more conservative precision that counts `unclear` as not established was 55.9%
and 42.9%. These are model-provisional diagnostics, not human ground truth.

The fresh high-grade smoke artifact contains 85 native identity candidates,
but its 80 distinct candidate endpoints carry no structured `entity_type`,
`mention_id`, observable attributes, or evidence references; only one native
edge connects two explicit `entity_mention` nodes. The correct next step is to
preserve grounded clip-schema entity references through L1 composition, not to
lower the identity threshold. Until independent human labels accept non-empty
admitted identity and state-transition sets, candidate edges are not called
reliable and the formal navigation comparison remains gated.

The isolated entity-reference rerun under
`outputs/video_holmes_entity_refs_smoke` closes that data-contract failure. It
completed 55/55 clip schemas and 55/55 neighbor-composer cache targets after
resumable retries, with zero final compose, schema, or integrity errors and an
intrinsic `high` grade. All 93 entity mentions have a mention ID, entity type,
non-empty observable attributes, and evidence references. All 82 native
identity edges connect two such entity mentions; the old smoke was 1/85.

The strict verifier automatically admits none of those model proposals: eight
carry explicit type or normalized stable-attribute conflicts and 74 remain
raw-video reread candidates (63 unique endpoint pairs). A GPT-5.6 visual
provisional reread accepted 69 edges, of which the component-wide consistency
pass merged 64; the simultaneously visible look-alike men near the end remain
separate tracks. This is an engineering smoke, not the independent-human gate.
No accepted track has a sufficiently grounded same-attribute before/after
delta, so zero `state_transition` edges are generated. A one-case provisional
navigation ablation is negative: semantic-only evidence recall is 0.5 at two
reads, while event-only, native-L1, and verified-dependency recall are 0.0.
Formal navigation conclusions therefore remain blocked on independent labels,
non-empty verified state/dependency edges, and a larger fixed gold case set.

The full isolated end-to-end run also completed with 32 atomic events, 172 L1
observations, 175 event relations, and 37 native L1 navigation relations. Its
relation teacher finished without errors. The first held-out graph-audit reply
was truncated at the shared 3000-token limit; audit retries now use an
independent 8000-token budget and compact retry instructions. Re-auditing the
persisted overlay completed all 27/27 requested rows with temporal consistency
passing. That result is a GPT-OSS engineering audit, not independent-human
precision evidence. The failed response is preserved as `audit.failed.json`.
An existing overlay can be retried without repeating video extraction:

```bash
python -m memory_graph.retry_overlay_audit \
  --overlay memory_graph/outputs/video_holmes_entity_refs_smoke/memory_graph/mKqiGQrHtW8/causal_temporal_overlay.json \
  --summary memory_graph/outputs/video_holmes_entity_refs_smoke/memory_graph/summary.json
```

#### Implementation plan: evidence-first relation admission

The implementation is ordered by safety dependency. Later phases may not
bypass an incomplete earlier phase.

**Phase 0 — close integrity and causal-admission holes**

1. In `pipeline.py`, remove `explains` and `enables` whenever
   `causal_allowed` is false. Require a passed visual verification for every
   causal edge when video-only L1 has no independent human acceptance.
2. Align the Video_Skills composer vocabulary, clue-memory JSON Schema, and
   memory-graph relation vocabulary. Validate actual artifacts, not only Python
   dataclasses.
3. Strengthen `audit_video_skills_l1` with duplicate node/edge IDs, self-edges,
   endpoint existence, timestamp validity, probability ranges, edge
   vocabulary, failed compose steps, and full schema validation.
4. Add `build_report` to the overlay schema or serialize it as a separate
   artifact.
5. Reject unknown explicit local node references and duplicate
   `(clip_id, local_node_id)` values instead of silently falling back to a
   clip-primary node.

**Phase 1 — restore relation semantics**

1. Stop the broad mappings:
   `same_object/reappears → same_entity`,
   `supports_observation → transition_support`, and unconditional
   `state_change → state_transition`.
2. Add distinct L1 relation contracts for `observation_support`,
   `same_instance_candidate`, `reappears_candidate`, verified `same_object`,
   and verified `same_entity`.
3. Remove clip-only endpoint substitution from admitted relations. An
   unresolved endpoint becomes a repair request, not an edge.
4. Preserve source labels, source confidence, endpoint-resolution method, and
   evidence references in every candidate.

**Phase 2 — build entity observations and identity candidates**

1. Project explicit `entity_mention`, `state_of`, and `derived_from` references
   first. Lexical same-clip association is retained only as a provisional
   candidate.
2. Normalize entity type and observable attributes such as color, clothing,
   material, shape, size, role, carried-by, located-in, and motion.
3. Generate candidate pairs from temporal neighborhoods, semantic/embedding
   top-K, native Video_Skills identity hints, and long-gap recurrence retrieval.
4. Store decomposed support:
   temporal feasibility, semantic compatibility, attribute compatibility,
   trajectory continuity, context continuity, and contradiction evidence.

**Phase 3 — verify identity and construct conflict-aware tracks**

1. Hard-reject type conflict, simultaneous incompatible instances, impossible
   displacement, and incompatible stable attributes.
2. Treat absent attributes as unknown, never as disagreement.
3. Admit `same_object` / `same_entity` only with grounded endpoint mentions and
   sufficient positive identity evidence. Retain ambiguous pairs as candidate
   edges.
4. Before merging two tracks, test every member against component-level time,
   type, location, and stable-attribute constraints. Do not use unconstrained
   transitive union.
5. Send only unresolved high-value candidates to targeted raw-video reread.

**Phase 4 — derive state and dependency relations**

1. Build a state transition only after its subject maps to one accepted
   identity track.
2. Require the same normalized attribute, different grounded before/after
   values, correct temporal direction, and explicit evidence for both states.
3. Keep repeated or consistent state as `state_continuity`, not
   `state_transition`.
4. Derive `transition_support` from accepted identity/state continuity plus
   temporal relevance. Keep `observation_support` separate.
5. Treat action-response as `response_candidate` until an observable response
   verifier passes.

**Phase 5 — re-enable sparse candidate causality**

1. Generate causal candidates only from accepted event identities and verified
   state deltas.
2. Require affected entity, before state, after state, mechanism kind,
   temporally ordered evidence, and minimal support set.
3. Apply raw-video witness verification and counterevidence checks.
4. Store rejected causal candidates and reasons; never downgrade them into an
   accepted correlation edge automatically.

**Phase 6 — navigation and SelectStream evaluation**

1. Enforce graph-read budgets and relation-specific confidence thresholds.
2. Candidate identity/correlation edges may propose one-hop reads but may not
   support answers, transitive clustering, or causal traversal.
3. Compare semantic-only retrieval, event-only graph navigation, native L1
   candidate navigation, and verified dependency navigation.
4. Measure answer accuracy, graph reads, evidence recall, identity precision,
   state-transition precision, dependency precision, and memory
   keep/merge/evict utility.

The first validation set is the existing 45-edge artifact with independent
human labels added to the GPT5.6 provisional audit. Acceptance targets are at
least 90% strict precision for admitted identity and state-transition edges,
zero schema/integrity errors, zero causal-gate bypasses, and measurable
navigation improvement over semantic-only retrieval. A newly generated
Video_Skills artifact must then confirm that preserved local IDs and explicit
reference edges improve coverage without reducing precision.

### Video-only L1 before relation prediction

The authoritative substrate is the accepted Video_Skills L1 pipeline:

```text
raw video
  → Qwen clip schemas
  → neighbor-aware Video_Skills L1 graph compose
  → video_only hidden-supervision filter
  → intrinsic high-grade L1 acceptance gate
  → immutable ClueMemoryGraph adapter
  ├→ native observation/entity/state navigation graph
  └→ event-only L1.5 passthrough → verified relation overlay
```

`pipeline.py` rejects video-only inputs without a materialized
`metadata.clue_memory_graph`. `video_skills_l1.py` requires the Video_Skills
intrinsic grade to be `high`, no hidden nodes or invalid edges, successful Qwen
clip schemas, no deterministic graph fallback, and semantic nodes produced by
the L1 composer. This acceptance is structural/perceptual; it is not mislabeled
as independent-human semantic correctness.

`adapter.py` preserves the L1 node provenance, clip anchors, modality,
confidence, and incident L1 edges. `l1_structural_edges.py` separately
materializes accepted native `same_entity`/`same_object`/`reappears`,
`state_change`, and `supports_observation` edges as L1 navigation relations.
Model-composed edges retain their original confidence and remain uncalibrated
priors. Deterministic reference edges are marked deterministic.

This is a dual-track contract:

```text
l1_structural_relations
  endpoints: immutable L1 observations
  purpose: entity/state/dependency navigation
  forbidden: explains, enables

relations
  endpoints: L1.5 atomic events
  purpose: temporal, verified semantic, and candidate-causal overlay
  causal admission: hard verifier + optional visual witness
```

Navigation may enter the native L1 graph through an event's evidence reference,
follow a structural relation, and return through an event projection. The
native relation chooses what persisted evidence to read; it is not itself
answer evidence. `causal_hint` remains recall-only and never becomes an
accepted L1 or event causal edge.

The standalone full-video `video_l1.py` extractor is not an authoritative L1
replacement. Its replacement path is disabled. Qwen frame rereads remain
available only as targeted repair/verification for a specific accepted node,
state transition, or candidate-causal witness.

GPU execution remains staged: run Video_Skills
`dataset_clip_wrapper.run_staged_llm_pipeline --skip-l2-planner`, unload the
9B server, then pass its `examples.jsonl` to
`validate_video_holmes.py --video-skills-l1-jsonl ...`.

#### Historical result and required structuralization

The experiments established a consistent failure sequence:

1. Coarse expert segments completed 47/50 videos but achieved only 54.4%
   strict relation precision. Narrative succession was over-labeled as
   explanation or enablement.
2. Targeted visual rereads found five candidate-causal witnesses. Only one
   initially passed, and the hard verifier rejected it as a perception-only
   restatement. No trusted causal edge remained.
3. A separate full-video Qwen L1 extractor produced frame-grounded events but
   also duplicated long actions, admitted camera/edit events, confused
   participant surfaces, and produced almost no state changes. It is not an
   authoritative L1 source.
4. An accepted Video_Skills artifact passed the intrinsic high-grade gate with
   301 semantic nodes, 190 semantic edges, and complete clip coverage. The
   memory overlay retained 269 L1 observations and 27 event endpoints, but all
   20 semantic/dependency labels were rejected: 14 `same_entity`, three
   `state_transition`, and three `transition_support`.

The last rejection is a schema mismatch, not evidence that the accepted L1 is
poor. Video_Skills stores entities and states as graph nodes and edges, whereas
the memory verifier expects event-local `participants` and `states` arrays.
Passing event text through while dropping that neighborhood correctly leaves
the hard verifier with nothing it can ground.

The required bridge is deterministic:

```text
accepted Video_Skills ClueMemoryGraph
  → event/observation node
  → traverse entity_mention / same_entity / same_object edges
  → event-local EntityMention with source L1 node ID as mention_id
  → traverse state_change / supports_observation / located_in edges
  → event-local StateAssertion with source evidence refs
  → preserve unresolved links explicitly
  → L1.5 event endpoints
  → dependency proposals and hard verification
```

This bridge must not invent an entity or state from prose. Direct graph
structure is projected first; unresolved identity/state is escalated to a
targeted visual reread. Existing `causal_hint` edges remain recall hints and
never bypass visual witness verification.

The first accepted-artifact smoke after implementing this bridge projected
participants for 10/27 event nodes and typed states for 2/27. GPT-OSS proposed
two `same_entity` labels above threshold; both were still rejected because the
two endpoint mentions belonged to different L1 identity components. This is
the intended conservative result: structuralization removed the original
"empty participants" failure where evidence exists, but did not collapse
unlinked mentions merely because their prose both says "man" or "vehicle".
The remaining unresolved event identities and state subjects are targeted
visual-reread work, not grounds for weakening the verifier.

New Video_Skills L1 artifacts preserve each neighbor composer's
`local_node_id` together with its clip scope. The composer also emits
deterministic intra-clip `entity_mention`, `state_of`, and `derived_from`
reference edges. This makes future entity/state projection an exact graph
operation. Older accepted artifacts remain usable through conservative
endpoint resolution; on the current smoke artifact that recovers 45 native L1
navigation relations (34 identity, seven state-transition, four transition
support) while leaving causal admission unchanged.

### Current prototype status

The first Phase 1/2 implementation now includes:

- `adapter.py`: converts a Video_Skills canonical example or L1 clue graph into timestamped `MemoryNode` objects;
- `embedding.py`: lazily loads the official `Qwen/Qwen3-VL-Embedding-2B` checkpoint and stores 2048-dimensional normalized node embeddings by reference;
- `graph_builder.py`: creates deterministic interval relations and sparse Top-K embedding candidate pairs;
- `contracts.py` and `atomic_events.py`: define and verify L1-grounded atomic event, entity mention, and visible-state hypotheses;
- `event_adapter.py` and `pipeline.py`: implement the only supported L1 → gate → L1.5 → relation → verifier → calibration path;
- `verifiers/`: deterministically reject ungrounded entity, state, contradiction, explanation, and enablement claims;
- `navigation.py`: proposes temporal/entity/state/dependency reads, predicts
  read utility, executes real graph reads, and keeps imagined observations out
  of the acquired-evidence state;
- `selectstream_policy.py`: protects predictive bridges and verified causal
  witness sets under a bounded keep/merge/evict plan;
- `calibration.py`: fits relation-wise decision thresholds only from explicitly identified independent labels;
- `prepare_independent_edge_audit.py`: emits a blinded annotation packet and a separately held model key;
- `audit_atomic_overlay.py`: distinguishes L1-contained spans from genuinely trusted temporal order;
- `validate_vrbench.py`: measures long-video temporal order and bridge coverage without treating reasoning steps as causal-edge gold;
- `reliability.py` and `audit_l1.py`: compute hard L1 checks and require independent labels for semantic reliability gates;
- `types.py`, `memory_graph.schema.json`, and `causal_temporal_overlay.schema.json`: define separate L1 observations and L1.5 event endpoints;
- `schema_validation.py`: validates serialized overlays with local schema
  resolution before CLI and benchmark writers persist them; legacy v0.2
  reports may omit unavailable `candidate_relations`, while every new build
  writes the complete field;
- `identity_tracks.py`: admits only explicitly verified identity links after
  component-wide type, stable-attribute, simultaneous-instance, and motion
  consistency checks; unverified native labels remain candidates;
- `identity_reread.py`: prepares graph-hash-bound raw-video review packets,
  deduplicates repeated endpoint pairs, validates reviewer provenance and
  evidence, and applies decisions back to either a graph or nested canonical
  artifact;
- `state_relations.py`: derives a state-transition proposal only when both
  visible states belong to the same accepted conflict-aware track, use the
  same normalized attribute, and have different grounded values;
- `l1_relation_audit.py`: prepares the blinded 45-edge native-L1 human audit
  and enforces separate ≥90% identity and state-transition precision gates;
- `navigation_ablation.py`: compares semantic-only, event-only, native-L1
  candidate, and verified-dependency reads under the same persisted-read
  budget and independently supplied evidence targets. Explicit state-change
  questions route the verified strategy to the jointly most relevant
  hard-verified `state_transition` pair; no gold event ID is used for routing;
- `cli.py`: builds an atomic-event overlay from one canonical example;
- `tests/test_memory_graph.py`: covers adapter selection, interval relations, relation scoring, and the embedding contract.

#### Latest structured-state smoke assessment (2026-07-20)

The new structured-state Video-Holmes smoke supersedes the earlier negative
navigation diagnosis above, while retaining that run as historical evidence.
The resumable Video_Skills flow completed 55/55 clip schemas and 55/55 graph
composition targets with zero final schema or integrity errors. The resulting
canonical graph contains 177 structured state nodes and exactly 177 `state_of`
edges. Its L1.5 overlay contains 30 atomic events, all 30 with explicit
participant references, and 67 strict visible-state assertions.

Every atomic event retains an `embedding_ref` for the normalized,
2048-dimensional `Qwen/Qwen3-VL-Embedding-2B` representation. Embedding text
uses the `event+participants+states/v1` contract so future navigation can use
the event description, grounded participants, and visible states together.
Large vectors remain in `.npy` matrices; graph JSON stores the model,
dimension, dtype, normalization flag, row index, matrix path, and SHA-256
checksum. Navigation questions use the same model and persist equivalent query
embedding references and manifests.

After a GPT-5.6 visual **provisional** identity reread, the conflict-aware track
pass admits one strict state transition: the same tracked man's expression
changes from `serious` to `focused`. The transition connects the explicit
before/after events and passes the accepted-track hard verifier. The base
video-only artifact still admits no probabilistic identity or state edge
without an independent reviewer, as intended.

On the unchanged six-case, raw-video-reviewed provisional gold set, with two
persisted event reads per question, the final Qwen embedding ablation is:

| Strategy | Answer accuracy | Mean evidence recall |
| --- | ---: | ---: |
| Semantic only | 66.7% | 0.833 |
| Event only | 66.7% | 0.833 |
| Native L1 candidate | 66.7% | 0.833 |
| Verified dependency | **83.3%** | **0.917** |

For the held diagnostic state-change question, the first three strategies
acquire only one of two required events (`0%` answer accuracy and `0.5`
evidence recall). Verified dependency follows the strict state edge and
acquires both endpoints (`100%` and `1.0`) under the same read budget. This is
positive engineering evidence that a verified state dependency can improve
navigation; it is not yet a statistically reliable benchmark result.

The current assessment is therefore:

- the implementation is reliable enough for continued evaluation, and the
  central dependency-navigation hypothesis now has a positive signal;
- strict identity/state admission correctly favors precision over coverage;
- the embedding representation is a reusable artifact rather than a temporary
  evaluation input;
- formal validation is still blocked because the result covers one video, six
  fixed questions, and one admitted transition, and its identity decisions are
  GPT-5.6 provisional rather than independent-human labels;
- the `change` / `become` / `from ... to ...` query-intent route needs broader
  paraphrase and hard-negative evaluation before it is treated as general.

The next evidence milestones, in order, are:

1. independently review the existing identity-relation packet and require at
   least 90% strict precision for admitted identity and state-transition edges;
2. freeze 30--50 navigation questions across multiple videos before examining
   ablation results;
3. include diverse state-change paraphrases and negative questions that mention
   change even though no verified transition exists;
4. report coverage alongside precision: videos with accepted tracks, tracks
   with grounded state deltas, and questions that can use a verified dependency;
5. only after those gates pass, build the persistent embedding index and expand
   to learned or multi-hop navigation.

The repository does **not** include trained relation-head weights. Consequently, `same_entity`, `state_transition`, `explains`, `enables`, and `contradicts` must not be reported as calibrated posteriors until a teacher-labeled or human-audited relation dataset has been fitted and calibrated.

See [`VALIDATION.md`](VALIDATION.md) for the historical coarse-segment run and the new staged protocol. The historical 47/50 result remains **54.4% strict precision** and must not be presented as validation of the atomic-event overlay. The trusted video-only + independent-audit rerun is still required.

See [`L1_RELIABILITY.md`](L1_RELIABILITY.md) for the updated architecture decision: L1 remains the grounded observation graph, while atomic events and causal-temporal hypotheses live in a separate L1.5 belief overlay.

### Model routing for mechanism-grounded candidate causality

The causal overlay uses a routed model stack. The models must not all perform
the same proposal-and-approval task:

```text
raw video
  → local Qwen/Qwen3.5-9B VLM: visible events, entities, states, and timestamps
  → deterministic mechanism candidate generation
  → targeted Qwen/Qwen3.5-9B VLM reread: cause / bridge / effect windows
  → GPT-OSS-120B teacher/critic: causal interpretation and alternative causes
  → deterministic hard verifier
  → calibrated candidate-causal witness
  → SelectStream-style keep / merge / evict and graph reads
```

`Qwen/Qwen3-VL-Embedding-2B` remains the node encoder. It can supplement
candidate recall, but embedding similarity is not a causal mechanism.

The local 9B VLM is the visual evidence producer. It must attach frame or time
provenance to:

- directly visible cause and effect events;
- entity continuity;
- before/after state;
- contact, transfer, access, response, or another visible mechanism;
- any refined temporal order inside a coarse L1 span.

GPT-OSS-120B is a teacher and causal critic, not the visual source of truth. It
may classify a structured witness, identify missing mediators, challenge an
alternative explanation, and generate student training traces. A GPT-OSS
judgment over event text alone must never pass as visual verification.

The runtime target is a local 9B VLM plus deterministic verifiers and a small
calibrated relation/mechanism head. GPT-OSS is reserved for data generation,
hard-case escalation, and offline audit. Independent human labels remain
required because using GPT-OSS as both teacher and auditor is circular.

#### Causal witness, not pair classification

Candidate causality is represented by a structured witness:

```yaml
CausalWitness:
  cause_event_id: string
  effect_event_id: string
  mechanism_event_id: string | null
  affected_entity: object | null
  before_state: object | null
  after_state: object | null
  mechanism: state_bridge | observable_precondition | rule_response | contact_transfer
  mechanism_detail: string
  evidence_refs: [string]
  minimal_support_set: [string]
  evidence_quotes: object
  confidence_components:
    temporal_grounding: float
    entity_continuity: float
    state_delta_grounding: float
    mechanism_visibility: float
    effect_grounding: float
    alternative_cause_penalty: float
  alternative_explanations: [string]
```

The support set must contain the cause and effect and may contain an explicit
mediator. This replaces the earlier assumption that every causal claim can be
verified from exactly two event endpoints.

The first mechanism-grounded implementation keeps the public `explains` and
`enables` relations for compatibility. Internally, it distinguishes:

- direct visible state change;
- establishment of an observable precondition;
- visible action/response;
- unsupported temporal or narrative progression.

`explains` should eventually be derived at question time from one or more
verified primitive witnesses rather than treated as a free-form base fact.

#### SelectStream integration

Candidate-causal evidence affects bounded-memory value:

```text
memory_value =
    semantic_relevance
  + temporal_bridge_value
  + state_change_value
  + predictive_dependency_value
  + causal_witness_value
  + unresolved_hypothesis_value
  - redundancy
```

`keep` protects witness endpoints, mediators, visible state transitions, and
unresolved candidates. `merge` is permitted only when temporal order, entity
continuity, before/after states, mechanism, and provenance survive. `evict`
prefers redundant background observations and must not remove only one member
of a protected witness set.

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
| `transition_support` | Grounded entity/state trajectory useful for navigation | Probabilistic, non-causal |
| `response_candidate` | Temporally local visible action-response hypothesis | Probabilistic, non-causal |
| `explains` / `enables` | Mechanism-grounded candidate causal support | Verified subset |
| `contradicts` | Conflicting states, events, or claims | Probabilistic |
| `bridge` | Evidence role that completes a multi-hop path | Derived from structure and task |

`explains` and `enables` represent candidate causal support only. They do not identify true causal effects from observational video.

## 6. Belief Update and Navigation

Initial action vocabulary:

```text
semantic
temporal_back
temporal_forward
track_entity
inspect_state_change
follow_dependency
find_bridge
search_counterevidence
verify
stop
```

Before executing an action, the world model predicts a belief transition:

```text
b^-_(t+1) = T_phi(b_t, a_t)
```

The system executes only the first planned operation and reads real grounded
evidence memory:

```text
o_(t+1) = ReadMemory(M_t, a_t)
b_(t+1) = Update(b^-_(t+1), o_(t+1))
```

Imagined evidence must never enter the final answer directly. The belief must
be corrected and planning repeated after every real evidence-memory read.

For IWM supervision, every legal action can also be executed independently
from the same immutable L1/L1.5 checkpoint. The resulting dataset stores real
observation descriptors and recomputes categorical belief deltas only after
the persisted read and correction. It keeps Qwen embedding references for
future retrieval but excludes raw vectors and numeric targets. Generated
records remain unreviewed until independently accepted/rejected and locked;
they are not automatically training gold.

Verifier provenance is explicit in executed-transition records. A persisted
hard-verifier result is labeled pre-read, replay without a verifier is
inconclusive, and only an actually invoked Video_Skills verifier is labeled
post-read. Failure to establish support is not treated as contradiction.
Balanced mining reports missing correction-sensitive categories rather than
backfilling them with ordinary temporal examples.
Its independent review rows also remove the miner's expected outcome from the
ID, question, tags, and category, and use stable hash ordering to prevent label
and row-position leakage.

## 7. Are We Performing Causal Inference?

The current method does not perform classical causal inference:

- it does not estimate average treatment effects;
- it does not use do-calculus to identify causal effects;
- it does not generate physical counterfactual videos;
- it does not treat temporal precedence as sufficient evidence of causality.

Here, a counterfactual means epistemic-action branching: execute different evidence-acquisition actions from the same belief checkpoint and compare their effects on observations, belief, and future answerability.

## 8. Are We Using a Factor Graph?

The L1/L1.5 Memory Graph is required as the shared evidence substrate. The
factor graph is not required as the main planner or ranking mechanism. The IWM
uses the fixed memory graph to construct legal candidates and bounded context,
then predicts future-belief effects to choose the next reasoning hop.

Factor graph / GTSAM remains available as an optional real-evidence belief
correction backup and as a graph-based baseline, diagnostic oracle, teacher,
and visualization tool. It never consumes imagined evidence and does not rank
reasoning operations. The complete backup boundary, existing implementation,
pilot, destructive controls, and migration status are centralized in
[`factor_graph/README.md`](../factor_graph/README.md).

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

Run mechanism-first candidate generation, targeted local 9B visual rereads, and
a SelectStream-style bounded-memory plan:

```bash
# Start the local Qwen/Qwen3.5-9B OpenAI-compatible worker first.
python -m memory_graph.cli \
  --canonical /path/to/video_only_canonical.json \
  --atomic-events /path/to/atomic_events.json \
  --embedding-output /path/to/event_embeddings.npy \
  --relation-teacher \
  --visual-reread \
  --visual-model Qwen/Qwen3.5-9B \
  --visual-api-base http://127.0.0.1:8000/v1/chat/completions \
  --require-visual-verification \
  --memory-capacity 32 \
  --input-mode video_only \
  --l1-human-audit /path/to/independent_l1_labels.json \
  --output /path/to/causal_temporal_overlay.json
```

`--visual-reread` is opt-in because it requires a running VLM endpoint and a
valid raw-video path in the canonical example. `--require-visual-verification`
turns a failed or inconclusive reread into a hard rejection for `explains` and
`enables`. `--memory-capacity` plans keep/merge/evict actions without mutating
the immutable evidence graph; executing safe consolidation remains a separate
writer responsibility.

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
