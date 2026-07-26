# Targeted Transition Review v2

这是修复媒体完整性问题后的重新采样 packet。它仍然是 outcome-blind、未审核数据，不用于训练。

## 修复内容

- 隔离 `0jq3JQ9YTqg`。其公开 annotation 使用到 `02:25`，但本地文件和当前 YouTube source 都只有约 63.44 秒，且视频语义与 annotation 不符，无法通过时间偏移修复。
- 从原 executed-transition dataset 重新做分层采样，而不是直接删除 v1 的公开 items，因此 packet checksum、hidden key 和 consistency duplicates 都重新生成。
- 构建 visual bundle 时读取真实容器时长。当前 52/52 items、167/167 windows 完整位于媒体时长内。

## 当前边界

容器时长完整不等于语义正确，也不等于 relation 已被证据证明。v1 的 GPT-5.6 检查显示，一些 clips 是有效的 reject/inconclusive controls；另一些 atomic windows 仍需人工确认。网页必须继续分别标注：

1. clip 是否展示公开 description；
2. action 是否返回了适合的证据；
3. evidence 对 relation 是 supports、rejects 还是 inconclusive；
4. belief delta 是否合理；
5. record 是否可进入训练数据。

## GPT-5.6 provisional labels

`transition_review.gpt56.provisional.json` 包含基于公开 description 与视频帧的五字段分类审核；`human_review_packet.locked_ai_provisional.json` 是应用这些标签后的 packet，状态明确为 `ai_provisional`。本轮结果为 21 条 provisional Accept、31 条 Reject；没有数字 reward、score、confidence、probability 或 utility。

`transition_review_inspection.gpt56.json` 记录 schema、覆盖率和 categorical distributions；`failure_slice_inspection.gpt56.json` 保存后续补采方向。这些结果可用于清洗和准备人工审核，但不能替代独立人工标签，也不能通过 formal gate。

隐藏 duplicate 的一致性报告只用于检测导出错误。由于本次 provisional 标签对相同公开 evidence 使用一致的 categorical judgment，它不能作为独立 reviewer reliability 的证据。

## 启动审核网页

```bash
python -m steam_video_new.implicit_world_model.l15_graph_navigator.human_review_server \
  --packet steam_video_new/implicit_world_model/datasets/targeted_transition_review_v2/human_review_packet.unreviewed.json \
  --visual-index steam_video_new/implicit_world_model/datasets/targeted_transition_review_v2/human_review_visual_index.json \
  --visual-key steam_video_new/implicit_world_model/datasets/targeted_transition_review_v2/human_review_visual_assets.hidden_key.json
```

打开 `http://127.0.0.1:8765/?lang=zh` 或 `http://127.0.0.1:8765/?lang=en`。

## Status

- annotation status：`unreviewed`
- formal eligible：false
- training ready：false
- training performed：false
- next gate：以 GPT-5.6 provisional 结果为预筛选，独立人工重点复核 21 条 Accept 和边界性 Inconclusive；被人工 Reject 的错窗不得进入正例训练，正确执行的 empty/reject/inconclusive controls 可按人工类别保留。
