# Counterfactual causal navigation — V3 result

Setup: HotpotQA distractor, 8 candidates, frozen
Qwen3-Embedding-0.6B, train 4096 / validation 512, four navigation steps,
32 latent memory slots, A100.

Results:

- pooled token probe: 63.48% accuracy
- token one-shot: 63.09%
- causal navigator: 65.04%
- `do(read=0)`: 0.39%
- Gaussian random direction: 64.65%
- batch-shuffled learned direction: 64.45%
- reject all reads: 14.06%

The causal navigator beats token one-shot by 1.95 points. Its gold-answer
margin rises from 1.951 to 2.089 across four steps; 66.1% of examples rise
monotonically. Removing reads causes a 64.65-point collapse, so recurrent reads
are causally necessary for this trained decision mechanism.

However, shuffling valid learned directions across examples drops only 0.59
points, below the preregistered 2-point gate. Gaussian random directions reduce
the final margin from 2.089 to 1.988 but change accuracy by only 0.39 points.
Thus most of the useful computation is in recurrent contextual reading, not in
semantic direction selection.

Verdict: **NO-GO for causal navigation**. The evidence supports
**counterfactually necessary recurrent reads**, but not a claim that the
learned semantic direction is causally necessary.

Artifacts:

- `causal_v2.py`: token memory, explicit cosine routing, interventions
- `train_causal_v2.py`: training, probes, and hard Go/No-Go gates
- `outputs/causal_v3_final_hotpot_4k.json`
- `outputs/run_causal_v3_final.log`

## MuSiQue 3–4 hop transfer

Setup: answerable MuSiQue 3–4 hop examples, all 20 paragraphs represented
without support-label ordering, 8 candidates including intermediate-hop hard
negatives, train 4096 / validation 512, 2048 encoder tokens.

Results:

- pooled token probe: 60.35%
- token one-shot: 63.28%
- causal navigator: 61.13%
- `do(read=0)`: 1.95%
- Gaussian random direction: 62.30%
- batch-shuffled learned direction: 61.72%

The four-step margin rises sharply from 1.425 to 3.363, and removing reads
causes a 59.18-point accuracy collapse. Nevertheless, the navigator is 2.15
points below token one-shot. Random and shuffled directions both outperform
the learned-direction result. Increasing answer margin therefore reflects
recurrent confidence amplification, not successful semantic navigation.

MuSiQue verdict: **NO-GO**. This harder benchmark strengthens the conclusion
that read dependence and rising margins are insufficient evidence for causal
direction selection.

Artifacts:

- `outputs/causal_v3_musique_3plus_4k.json`
- `outputs/run_causal_v3_musique_3plus_4k.log`
