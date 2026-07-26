# GPT-5-mini H2 scientific smoke v2

This is a no-training, three-case, five-arm infrastructure and failure-slice
smoke over frozen CG-Bench L1/L1.5 graphs. The model is
`openai/gpt-5-mini`, the real-read budget is two, and the model-backed IWM uses
horizon two. The complete run is in `run/cohort.final.json`.

## Integrity result

- all three cases and all five matched arms completed;
- runtime errors and method failures: zero;
- every IWM step retained complete hypothesis-conditioned joint-chain coverage;
- the categorical evidence scheduler selected one real read while preserving
  the full surviving path frontier;
- 297 no-read transition rows were generated locally as neutral protocol
  invariants and were not sent to the model;
- no numeric reward, Top-K pruning, training, or hidden clue feedback was used.

The shuffled-IWM intervention originally rotated no-read controls together with
model predictions. It now shuffles only real-read consequences and preserves
terminal/cursor-only invariants. The evaluator-only oracle now uses the same
model-localized entry frontier as the learned arms; it no longer starts from an
arbitrary node in the full graph.

## Results

| Arm | answer accuracy | mean clue recall | mean reads | transition outcome calibration | transition progress calibration |
|---|---:|---:|---:|---:|---:|
| world-model guided | 0.000 | 0.333 | 2.0 | 0.305 | 0.218 |
| no world model | 0.000 | 0.333 | 2.0 | n/a | n/a |
| shuffled IWM | 0.000 | 0.333 | 2.0 | 0.326 | 0.326 |
| immediate only | 0.000 | 0.667 | 2.0 | 0.311 | 0.660 |
| matched localized oracle | n/a | 0.833 | 1.33 | n/a | n/a |

The matched localized oracle reaches full clue coverage in two cases. In the
gluten case it reaches only one of two clues, because the frozen entry frontier
does not contain a start from which both clues are executable within two reads.
Consequently, the compile-time full-graph oracle is only a substrate ceiling;
the localized oracle is the valid per-model preflight.

On the two cases with a complete localized oracle ceiling, world-model clue
recall is 0.5, the same as no-WM and shuffled-IWM, while immediate-only reaches
1.0. The intact IWM therefore does not show a navigation advantage. Its action
sequence does change, all six WM decisions schedule real reads, and corrected
hypothesis beliefs diverge after reads, so the closed-loop mechanism works; the
predicted transitions and resulting choices are not calibrated well enough.

One representative failure predicts that the address `hand holds object`
resolves all question roles and makes the answer ready. The real observation
does not support that belief patch. This upstream over-crediting is then
faithfully propagated by the planner. The next data gate must therefore target
grounded action-conditioned transition and hypothesis-discrimination labels,
not add a heuristic tie-break or Top-K selector.

## Interpretation

This smoke validates the revised runtime contract, not the research claim. It
shows that the remaining bottlenecks are separable:

1. entry-localization recall under the actual read/hop budget;
2. IWM transition/belief-delta calibration;
3. path preference conditioned on those predicted transitions;
4. final answer evidence sufficiency.

`training_performed` is false. The cached responses and executed traces remain
evaluation artifacts and are not automatically eligible for SFT.
