# Human Transition Review UI

这是一个只在本地运行的 outcome-blind 审核网页。它只读取公开 transition packet，不读取 hidden key、target stratum、stored verifier 或 GPT-5.6 provisional labels。

```bash
python -m steam_video_new.implicit_world_model.l15_graph_navigator.human_review_server \
  --packet steam_video_new/implicit_world_model/datasets/targeted_transition_review_v1/human_review_packet.unreviewed.json
```

浏览器打开 `http://127.0.0.1:8765`。草稿按 packet ID 自动保存在浏览器 localStorage；页面可筛选未完成/待复核条目、校验完整性并导出 `independent_human` review JSON。

导出后使用正式 CLI 锁定：

```bash
python -m steam_video_new.implicit_world_model.l15_graph_navigator.workflow \
  apply-transition-review \
  --packet steam_video_new/implicit_world_model/datasets/targeted_transition_review_v1/human_review_packet.unreviewed.json \
  --review /path/to/targeted-transition-review-v1.independent_human_review.json \
  --output steam_video_new/implicit_world_model/datasets/targeted_transition_review_v1/human_review_packet.locked.json
```

服务端 `/api/validate` 会调用与 CLI 相同的 schema、citation coverage、categorical-only 和 checksum 逻辑，但不会写文件或读取 hidden key。
