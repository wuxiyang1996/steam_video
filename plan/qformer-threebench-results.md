# Q-Former RAG 三 Benchmark 冻结评测结果

评测日期：2026-08-11。模型、检索预算与统计协议见
`plan/qformer-four-slot-baseline-plan.md`。本报告只使用修复 cardinality 和 mmap
normalization 后的 full run `7232817`；早期 `n=1` run 不属于有效实验。

## 实验完整性

- OVO-Bench：3035/3035；
- VideoMME：2700/2700；
- StreamingBench flattened：4500/4500；
- StreamingBench 主结果：4004 个 official-compatible timestamped single-turn 样本；
- 三个 arm 使用相同候选池、`K=3`、Qwen3.5-9B、prompt、帧预算与 deterministic decoding；
- frozen Q-Former checkpoint：原训练 selection split 选择的 seed 17；
- 10,000 次 paired bootstrap、exact McNemar、六项检验 Holm correction；
- 完整统计 artifact：
  `/fs/gamma-projects/vlm-robot/steam_video_runs/qformer_threebench_v01/eval/paired_statistics.json`。

## 最终准确率

| Benchmark | Uniform | Visual RAG | Q-Former RAG | QF - Visual |
| --- | ---: | ---: | ---: | ---: |
| OVO-Bench | 44.74% | 43.49% | 43.49% | 0.00 pp |
| VideoMME | 58.30% | 57.70% | 57.70% | 0.00 pp |
| StreamingBench 主 subset | 59.92% | 57.32% | 57.42% | +0.10 pp |

StreamingBench flattened 4500 条的准确率分别为：uniform 60.49%、visual 58.16%、
Q-Former 58.22%。

## 配对统计

| Comparison | Delta | Bootstrap 95% CI | McNemar p | Holm p |
| --- | ---: | ---: | ---: | ---: |
| OVO: QF - visual | 0.00 pp | [-0.20, +0.20] pp | 1.000 | 1.000 |
| VideoMME: QF - visual | 0.00 pp | [0.00, 0.00] pp | 1.000 | 1.000 |
| Streaming: QF - visual | +0.10 pp | [-0.05, +0.25] pp | 0.344 | 1.000 |
| OVO: visual - uniform | -1.25 pp | [-2.21, -0.30] pp | 0.014 | 0.070 |
| VideoMME: visual - uniform | -0.59 pp | [-1.30, +0.11] pp | 0.129 | 0.517 |
| Streaming: visual - uniform | -2.60 pp | [-3.65, -1.55] pp | 2.15e-6 | 1.29e-5 |

## 诊断与结论

Q-Former 的 residual 确实会改变少量排序，但很少改变 top-3 集合：

- OVO：3025/3035 个问题的 top-3 集合与 visual 相同；
- VideoMME：2700/2700 完全相同；
- StreamingBench：约 99% 相同。

因此当前 QF residual 主要重排相同证据，而没有带来新的有效 evidence。最终 QA 上，QF
相对 visual 在 OVO/VideoMME 完全持平，在 Streaming 只有不显著的 +0.10 pp。与此同时，
uniform 在三个 benchmark 均优于 semantic retrieval，Streaming 的差距在 Holm correction
后仍显著。

这版 baseline 不能支持“Q-Former 改善 RAG QA”的结论。下一轮不应继续增加 QA benchmark
或 loss term；应先检查训练分布迁移和 residual 强度，并用 retrieval-level oracle/evidence
labels 验证 QF 是否真的改变 top-k evidence。外部 benchmark 缺少独立 entity/state sidecar，
本次该槽按协议 validity=0，也是需要单独验证的分布差异。
