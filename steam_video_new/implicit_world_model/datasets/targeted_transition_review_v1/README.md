# Targeted Transition Review v1

本目录针对第一轮 GPT-5.6 provisional review 的失败类型，保存定向补采与独立人工盲审入口。当前阶段不训练模型。

## Failure slices

`failure_slice_inspection.gpt56.json` 将 23 条 `inconclusive` 做成互斥主分区：

- identity 未确认：4
- 缺少同属性 before/after：3
- counterevidence 只有相关内容、没有直接反驳：12
- relation 定义不清：1
- evidence retrieval 不充分：3

分区依据只使用公开 action、relation、executed evidence、显式 state attributes 和 categorical annotation，不读取 hidden provenance 或 stored verifier。

## Targeted executed gathering

`human_review_packet.unreviewed.json` 从现有真实 executed-transition dataset 定向选择 47 条记录，覆盖 9 个视频，并加入 6 条隐藏 consistency duplicates，共 53 个盲审 items。

目标分层包括：

- strict state delta candidates
- identity hard-negative candidates
- counterevidence support/refute adjudication
- empty/reject controls
- inconclusive controls
- delayed two-hop cases

这些分层只是 acquisition rule，不是 gold label，且不会出现在公开 packet 中。当前数据中 strict state-delta 只有 3 条可用候选，低于目标 6 条；报告保留该 deficit，没有用弱关系补齐。

## Human review website

本目录额外提供 outcome-blind 视觉绑定：

- `human_review_visual_index.json`：reviewer 可见的 node、角色、视频 ID 与时间窗；不含本地路径、采样分层、stored target 或 verifier outcome。
- `human_review_visual_coverage.json`：覆盖审计；当前 53/53 items、173/173 requested nodes 均可播放，覆盖 9 个视频。
- `human_review_visual_assets.hidden_key.json`：本地视频绝对路径绑定；受 `*.hidden_key.json` 保护，不能交给 reviewer。

如需从 packet 重建绑定：

```bash
python -m steam_video_new.implicit_world_model.l15_graph_navigator.workflow \
  build-visual-review \
  --packet steam_video_new/implicit_world_model/datasets/targeted_transition_review_v1/human_review_packet.unreviewed.json \
  --key steam_video_new/implicit_world_model/datasets/targeted_transition_review_v1/human_review_packet.hidden_key.json \
  --video-root /fs/gamma-projects/vlm-robot/datasets/Video-Holmes/Benchmark/videos_cropped \
  --output steam_video_new/implicit_world_model/datasets/targeted_transition_review_v1/human_review_visual_index.json \
  --key-output steam_video_new/implicit_world_model/datasets/targeted_transition_review_v1/human_review_visual_assets.hidden_key.json \
  --report steam_video_new/implicit_world_model/datasets/targeted_transition_review_v1/human_review_visual_coverage.json
```

启动带真实视频时间窗的本地网页：

```bash
python -m steam_video_new.implicit_world_model.l15_graph_navigator.human_review_server \
  --packet steam_video_new/implicit_world_model/datasets/targeted_transition_review_v1/human_review_packet.unreviewed.json \
  --visual-index steam_video_new/implicit_world_model/datasets/targeted_transition_review_v1/human_review_visual_index.json \
  --visual-key steam_video_new/implicit_world_model/datasets/targeted_transition_review_v1/human_review_visual_assets.hidden_key.json
```

打开 `http://127.0.0.1:8765`。网页并排显示 source/target/observation 的真实视频区间；服务端通过 HTTP Range 读取原 MP4，浏览器只播放相应的 `start_s/end_s`，不会公开本地文件路径。网页同时支持自动保存/恢复、未完成与待复核筛选、公开 evidence citation 选择、服务端 schema 校验，以及导出 `independent_human` review JSON。

审核期间不要读取：

- `human_review_packet.hidden_key.json`
- `human_review_visual_assets.hidden_key.json`
- `targeted_gathering_manifest.hidden_key.json`
- 第一轮 GPT-5.6 provisional annotations

上述 hidden 文件均受仓库 `*.hidden_key.json` 规则保护。

## Status

- formal eligible：false
- training ready：false
- training performed：false
- next gate：独立人工完成全部条目并导出、锁定为 `human_locked`
