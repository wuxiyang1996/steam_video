# IWM Runtime Train Collection v1

这是后续收集 multi-trajectory IWM/Planner supervision 的冻结 train-split cohort，
不是训练数据本身。

## Selection

- 40 cases / 40 distinct videos；
- 全部来自 CG-Bench train split；
- stable hash sampling，每个视频最多一例；
- selection 不读取 question text 或 hidden answer；
- L1/L1.5 必须按视频 question-independently 构建。

当前原始视频总时长约 68,641 秒。所有 graph、clue-retention、multi-trajectory
execution 和 independent review 状态目前都是 `pending`，没有启动 9B 训练。

每例采集后必须分类覆盖：

- weak-related-but-insufficient；
- identity unresolved/conflicting；
- incomplete state delta；
- hypothesis-discriminating support；
- counterevidence；
- empty/inconclusive controls；
- delayed second-hop completion。

`collection_manifest.json` 是冻结队列。只有完成 question-independent L1/L1.5、
localized clue-retention gate、真实 multi-trajectory execution、独立 review 和
video-disjoint split audit 后，才能生成另外的 human-locked SFT adapter。

`training_ready=false`，`training_performed=false`。

## Question-independent graph and collection launch

`l15_train_selection.json` 是唯一传给 L1/L1.5 worker 的 selection。它只含视频
ID、路径、时长、split 和 observation horizon，不含 case ID、问题、选项、答案或
clue。`l15_train_protocol.json` 单独保存 40 个 evaluator case；它只在 graph 冻结后
用于 clue-retention gate，绝不传给 graph builder。

可恢复的完整作业链由以下命令提交：

```bash
steam_video_new/implicit_world_model/cgbench_grounded_navigation/submit_l15_train_collection.sh
```

该提交器将偶数/奇数 shard 分别放到 L40S 与 RTX A6000，默认总并发最多 8、每个
GPU shard 申请 4 CPU，且每个 shard 只负责一个完整视频。后续阶段使用 Slurm
dependency 串联：

```text
40-video L1 extraction
  -> merge + Qwen3-VL-Embedding-2B correlation build
  -> hidden evaluator clue-retention gate
  -> frozen compile gate
  -> GPT-5-mini five-arm multi-trajectory collection
```

任一 graph/gate 阶段失败时，下游 collection 不会读取半成品。最后一阶段默认以 8 个
case workers 并行收集；每个 case 内仍保留完整 legal actions、joint trajectories 和
matched arms，不用 Top-K 缩减工作量。该阶段只收集 executed transitions 和
matched-arm diagnostics，不训练 GPT-OSS、Qwen 或 9B adapter。

若 cohort-level composition gate 因某个独立 slice（例如 delayed proxy 缺失）失败，
可以先对 clue-retained cases 生成独立的 passed compile-gate artifact，再设置
`SKIP_COMPILE_GATE=1` 复用它。runner 只从该 frozen artifact 的
`runnable_case_ids` 读取 case，不允许手工追加或绕过单例 clue/boundary checks。
