# IWM Supervision Phase 2

本目录停在数据准备与 schema 审计，不执行训练。

- `iwm_supervision.untrained.json`：question、categorical belief-delta target 与
  trajectory preference；只允许 IWM training/offline evaluation 使用。
- `iwm_node_alignment.hidden_key.json`：GT clue intervals 在 graph freeze 后与 L1
  nodes 的 hidden alignment，禁止提供给 graph builder 或 runtime planner。
- `iwm_supervision_readiness.json`：覆盖与缺失报告。
- `full_video_l1_generation_queue.json`：整视频 L1 构建队列；不含 question、answer、
  clue interval 或 transition target。

当前只有 8 个 smoke graph，得到 6 个 train bridge groups；validation/test bridge
groups 都是 0。149 个 distinct node pairs 来自 multi-positive node alignment，不是
149 条独立人工关系标注。672 条 categorical belief-delta target 已存在，但 grounded
observation descriptor 仍为 0，因此 `phase_3_training_ready=false`。

Phase 3 指后续 action-conditioned IWM，而不是独立的 learned L1.5 edge selector。
