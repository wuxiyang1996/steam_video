# Route A rescue — Go / No-Go judgment

**Setup:** controllable synthetic 2/3-hop possession chains, 8 candidates, Qwen3-Embedding-0.6B, action-LoRA discrete policy, weak `L_action` (warmup only) + entropy, progress loss.

**Hard Go criteria (all required):**
1. `discrete > one_shot`
2. temporal shuffle drop ≥ 2 pts
3. `m_k` rises (`m_gain>0` and `mono_rate>0.5`)

## Results (`outputs/rescue_synth_go_nogo.json`)

| Model | Acc | m_gain | mono_rate |
|---|---:|---:|---:|
| one_shot | **0.992** | 0.00 | 0.00 |
| plain_recurrent | ~(see json) | | |
| discrete_action | 0.291 | 0.005 | 0.516 |
| shuffle_temporal | 0.291 | — | — |
| random_actions | 0.314 | — | — |

## Verdict: **NO-GO**

| Criterion | Result |
|---|---|
| discrete > one_shot | ❌ 0.291 vs 0.992 |
| shuffle drop ≥ 2pts | ❌ drop = 0.0 |
| m_k rises | ✅ weak (mono≈0.52) |

## Honest read

1. Synthetic task as written is **too easy for one-shot** (99%): gold location phrase is linearly recoverable from frozen embeddings + one read.
2. Discrete navigator **collapsed / underperformed badly** — actions not helping, not causally necessary.
3. Therefore Route A **fails the death-sentence experiment**. Do **not** claim dynamic semantic navigation works from these runs.

## Recommended next move

**Switch to Route B (shrink claim):**  
fixed latent memory + support-progressive recurrent reads; demote actions to optional ablation.

Do not spend more A100 on Hotpot/Qwen-policy chasing this discrete-action claim until the task forces multi-step necessity (e.g. adversarial distractors that one-shot cannot solve but multi-step can — currently not the case).
