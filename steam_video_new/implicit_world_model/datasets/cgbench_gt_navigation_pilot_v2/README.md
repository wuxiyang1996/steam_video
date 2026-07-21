# CG-Bench GT-only Navigation Pilot v2

本目录是替代旧 matched-control pilot 的正式数据协议。唯一标签来源是 CG-Bench 已有的
人工 clue intervals 与 terminal answer；没有把 GT 外片段当作负例，也不要求重复人工审核。

## 当前规模

- 256 个 multi-clue cases，205 个视频；
- 672 条 GT clue acquisition transitions；
- 256 组 complete-clue vs leave-one-clue-out categorical preferences；
- train / validation / test：207 / 22 / 27，按 video ID 确定性隔离；
- numeric reward、probability、utility：无；
- negative transition supervision：无；
- human-review gate：无；
- model training：未执行。

## 文件

- `navigation_dataset.gt_only.json`：公开 planner inputs、GT-clue reads、categorical belief
  delta 与 coverage preference，不含正确答案；
- `terminal_targets.hidden_key.json`：terminal answer、answer key 与 clue identity，不得提供给
  planner；
- `build_report.json`：构建、切分、泄漏与状态报告；
- `gt_ablation.unexecuted.json`：五组无训练评测干预；
- `gt_ablation.hidden_key.json`：离线评分 key。

hidden key 文件受仓库的 `*.hidden_key.json` 规则保护，本地存在但不提交。

## 五组干预

1. `all_clues`：完整 GT evidence；
2. `leave_one_clue_out`：移除一个 GT clue；
3. `shuffled_clue_order`：完整 evidence、扰动读取顺序；
4. `transition_prediction_shuffle`：真实 reads 不变，只打乱 IWM imagined transition 对应；
5. `cross_video_distractor`：等读取次数的视频不相交结构性干预。

第 5 组只用于测试。它不是“语义无关”的人工标签，不能作为 negative supervision。

本 pilot 是 GT transition/controlled-ablation 数据，不把“候选仅含 clues”的设置包装成完整
navigation benchmark。真实 closed-loop 实验应由 L1.5 memory 提供候选 hops；CG-Bench 只为
命中的 clues 和 terminal answer 提供监督，其他候选保持 unlabeled。

## 状态

当前 observation descriptor 尚待 Qwen-VL 对这 672 个真实 interval 完整读取，embedding 字段
已预留给 `Qwen/Qwen3-VL-Embedding-2B`。按项目约束，本阶段停在数据构建与检查：
`formal_eligible=false`、`training_ready=false`、`training_performed=false`。

下一门禁仅包括 grounded-read 完整性、schema、GT coverage、hidden-answer leakage 和
video-disjoint split 检查。GPT-5.6/人工视觉检查可以作为抽样诊断，但不会改变 CG-Bench
标签，也不会阻塞后续实验。

## 真实候选空间准备

新增：

- `l15_candidates.public.json`：按视频共享、planner-visible 的真实 memory candidates；
- `l15_candidate_alignment.hidden_key.json`：candidate hop 与 GT clue 的 temporal overlap，仅供
  evaluator；
- `l15_candidate_report.json`：候选来源与缺失报告；
- `l15_graph_generation_manifest.json`：205 个视频的无 GT graph-generation 清单；
- `closed_loop_protocol.json`：WM/no-WM/prediction-shuffle/immediate-only/oracle-ceiling 协议；
- `modality_coverage_summary.json`：full grounding 验收后自动生成的模态覆盖报告。

当前 CG-Bench 尚无 persisted L1.5 graph。86/205 个视频存在 time-aligned SRT，可为 109/256
cases 提供 subtitle L1 fallback；其余 147 cases 暂无真实候选。subtitle fallback 不是 L1.5
完成状态，因此 generation manifest 仍包含全部 205 个视频。构建 graph 时明确禁止读取
question、choices、answer、clue intervals 或 hidden alignment。

候选文件约 19MB，但不会整体进入模型上下文。candidate sets 按视频去重；每一步只向 IWM/
planner 提供经过外部 embedding/structure retrieval 的 bounded top-K hops。
