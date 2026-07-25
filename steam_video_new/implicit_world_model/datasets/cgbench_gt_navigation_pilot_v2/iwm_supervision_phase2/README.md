# IWM Supervision Phase 2

本目录保存训练数据与 schema 审计产物；仓库当前尚未执行训练。完成 grounding 后，
`grounded_action_transition_corpus.json` 已允许用于 scoped data-loader/schema smoke、
刻意的小样本 overfit、descriptor distillation 与 split-safe 离线检查：以
`question + pre-read checkpoint + GT interval action` 预测 grounded observation
descriptor 与 categorical clue-coverage belief delta。

- `iwm_supervision.untrained.json`：question、categorical belief-delta target 与
  trajectory preference；只允许 IWM training/offline evaluation 使用。
- `iwm_node_alignment.hidden_key.json`：GT clue intervals 在 graph freeze 后与 L1
  nodes 的 hidden alignment，禁止提供给 graph builder 或 runtime planner。
- `iwm_supervision_readiness.json`：覆盖与缺失报告。
- `full_video_l1_generation_queue.json`：整视频 L1 构建队列；不含 question、answer、
  clue interval 或 transition target。

旧 readiness snapshot 只有 8 个 smoke graph，得到 6 个 train bridge groups；
validation/test bridge groups 都是 0。149 个 distinct node pairs 来自 multi-positive
node alignment，不是 149 条独立人工关系标注。该 snapshot 中
`grounded_observation_descriptor_ready_transition_count=0` 已过时：当前 672 条 transitions
中已有 670 条 grounded records（train / validation / test = 516 / 74 / 80），2 条
unavailable。

`phase_3_training_ready=false` 继续表示完整 L1.5 closed-loop / preference-training
协议尚未通过 full-video graph、held-out bridge 和五臂门禁；它不阻塞上述 scoped
数据管线 smoke。

该 corpus 不能单独作为 runtime categorical-IWM 验证：其 `evidence_progress` 和
`answerability_after` 均为单一类别，action 缺少 runtime semantic node key，且 target
schema 与 full-graph IWM contract 不同。正式 categorical smoke 仍需 full-video train
graphs、post-freeze node alignment、categorical controls、corpus adapter 与 held-out
evaluator。该 corpus 禁止供 graph builder 或 same-case runtime lookup 使用，也不提供
identity、state-transition、causal 或 outside-clue negative 标签。

Phase 3 指后续完整 action-conditioned IWM，而不是独立的 learned L1.5 edge selector。
