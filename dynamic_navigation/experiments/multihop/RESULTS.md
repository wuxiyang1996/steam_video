# Multi-hop navigator results — HotpotQA distractor (A100)

**Job:** `7091127` on `cml31` (1× A100 80GB, scavenger)  
**Encoder:** `Qwen/Qwen3-Embedding-0.6B` (VL-Embedding-2B AutoModel broken)  
**Metric:** candidate ranking acc (gold + 3 distractors), `m_k`, type breakdown

## Week 0.5 soft mixture (normalized)

| Model | Acc | m_gain | mono_rate |
|---|---:|---:|---:|
| one_shot | **0.602** | 0.00 | 0.00 |
| plain_recurrent | 0.594 | 0.16 | 0.60 |
| soft_mixture | 0.570 | 0.15 | 0.57 |

## Discrete action policy (main method)

`DiscreteActionNavigator`: `π(a|u,r,a_prev,q)` + Gumbel-Softmax + action FiLM + STOP.

**File:** `outputs/discrete_action_qwen06.json` (train512/val256, 8 epochs)

| Model | Acc | m_gain | mono_rate | bridge | comparison |
|---|---:|---:|---:|---:|---:|
| one_shot | **0.586** | 0.00 | 0.00 | 0.540 | 0.759 |
| plain_recurrent | 0.578 | 0.19 | 0.56 | 0.540 | 0.722 |
| soft_mixture | 0.570 | 0.20 | 0.57 | 0.530 | 0.722 |
| **discrete_action** | 0.582 | 0.12 | 0.38 | 0.550 | 0.704 |

Learned actions collapsed to teacher sketch: `GROUND → TRACE → COMPOSE → STOP` (freq 0.25 each).

| Intervention | Acc | m_gain |
|---|---:|---:|
| shuffle_actions | 0.582 | 0.12 |
| force_GROUND | 0.582 | 0.11 |
| force_TRACE | 0.582 | 0.43 |
| force_COMPOSE | 0.582 | 0.11 |
| force_teacher_sketch | 0.582 | 0.12 |

### Honest verdict
- Discrete action **code path works** (policy, FiLM, STOP, interventions runnable).
- Still **No-Go**: does not beat one_shot; interventions do not drop accuracy → actions not yet causally useful (policy collapsed + weak FiLM effect).
- Next: lower `λ_a`, entropy bonus, larger data, stronger action-specific LoRA.

## Artifacts
- Code: `experiments/multihop/model.py` (`DiscreteActionNavigator`)
- Log: `outputs/run_discrete.log`
- JSON: `outputs/discrete_action_qwen06.json`
