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

Phase 1/2 新增 `iwm_supervision_phase2/`：

- `iwm_supervision.untrained.json`：672 条 transition 的 readiness 与 categorical
  preference，明确禁止 graph builder/runtime planner 消费；
- `iwm_node_alignment.hidden_key.json`：GT clue interval 在 graph freeze 后对齐到 L1
  node groups，仅供训练准备和离线评估；
- `iwm_supervision_readiness.json`：按 split 报告真实 node-level bridge 覆盖；
- `full_video_l1_generation_queue.json`：205 个视频的整视频、question-independent L1
  生成队列，不包含 clue interval 或 answer。

grounding 前的 readiness snapshot 曾有 672 条 belief-delta targets、0 条 grounded
observation descriptor；已有 8-video smoke 只对齐出 6 个 train bridge groups，validation/test
均为 0。每个 clue group 可能对齐多个 nodes，因此 149 个 node pairs 是 multi-positive
展开，不是 149 个独立标注。Phase 3 仍明确为未训练状态。

更新：Qwen-VL grounding 后，672 条 executed transitions 中 670 条已有真实 observation
descriptor，2 条保留为 unavailable。`iwm_supervision_phase2/grounded_action_transition_corpus.json`
将它们导出为 video-disjoint 的 train 516 / validation 74 / test 80 条记录。该文件只用于未来
IWM post-training 或 split-safe 离线评估；禁止作为同 case runtime lookup，当前仍未训练模型。

运行时另有两个不同边界的 cache：transition cache 只缓存基于安全可见输入的 action-conditioned
categorical prediction；response cache 只保存请求哈希和 categorical JSON，不保存 prompt/question/
GT。它们用于 matched arms 复用同一预测和断点续跑，不是监督标签，也不能被解释为 calibrated
probability 或 numeric reward。

## Fixed held-out full-video L1/L1.5 status

固定的一个 test 视频与一个 validation 视频已完成 question-independent 全视频构建：分别产生
377 与 535 个 L1 observations，并在 capacity 64 下各保留 64 个 nodes。冻结后 evaluator 才接入
GT：5/5 eligible clues 被 native candidates 覆盖；唯一具有两端 L1 overlap 的连续 clue bridge
来自 test split，并由一条直接 soft-correlation edge 覆盖。validation bridge 仍缺少一端 L1
overlap。当前没有 trusted negative relation labels，因此 admitted-edge precision 仍为 unavailable，
不能从 1/1 positive bridge 推导 precision。构图未使用 question/GT、未进行 per-pair LLM 调用、
未应用 Top-K，也未训练模型。下一步是在这两个冻结 graph fingerprint 上运行五臂导航，而不是
继续调 graph admission policy。

当前 CG-Bench 尚无 persisted L1.5 graph。86/205 个视频存在 time-aligned SRT，可为 109/256
cases 提供 subtitle L1 fallback；其余 147 cases 暂无真实候选。subtitle fallback 不是 L1.5
完成状态，因此 generation manifest 仍包含全部 205 个视频。构建 graph 时明确禁止读取
question、choices、answer、clue intervals 或 hidden alignment。

候选文件约 19MB，但不会整体进入模型上下文。candidate sets 按视频去重；每一步只向 IWM/
planner 提供当前 cursor incident、由全局 admission policy 接纳的合法 hops，不使用每节点固定
Top-K 作为主方法。

`l15_graph_smoke_selection.json` 已固定 12 个不重复视频，覆盖 train/validation/test × 有/无
字幕六个 strata，每组 2 个。selection 不读取 question、answer 或 clue intervals。smoke 只扫描
每个视频前 120 秒；graph 冻结后才由 hidden evaluator 计算 native 与 embedding top-K clue
recall，且只统计完整落在 observation horizon 内的 clues。

## Full-graph proxy smoke（2026-07）

8-video smoke graph 已增加一次性、question-independent 的 grounded caption candidate
overlay。模型共提出 22 对，其中 17 对因已有 temporal/semantic topology 被去重，最终只保留
5 条 scoreless candidate hops；没有为了提高覆盖率而补造 change/contrast/causal edges。

在固定 puppy case 上，修正后的 shortest-path oracle 使用四次读取覆盖 4/4 GT clues，证明当前
L1/L1.5 substrate 存在可执行路径。零样本 proxy 结果仍不足以支持方法有效性：

- `openai/gpt-5-mini`：6 reads，0/4 clues；
- `openai/gpt-5-mini` no-WM（2-read matched diagnostic）：模型直接选择
  `abstain`，0 reads；IWM 改变了动作但尚未带来 coverage gain；
- `qwen/qwen3.7-max`：2 reads，1/4 clues，约 332 秒；
- oracle（6-read budget）：4 reads，4/4 clues。

两次模型实验都只使用 categorical observation/belief delta 与 ordinal/setwise preference，未向
模型提供 hidden clue intervals/answer，未让模型生成 numeric reward/confidence，也未训练模型。
因此目前应标记为 `graph_executable=true`、`zero_shot_iwm_ready=false`。Qwen 的 2-read 指标只
用于早期 action-quality 诊断，不能与 GPT-5-mini 的 6-read final recall 作同预算比较。
