# Transition Review v1

本目录保存 executed transition 的第一轮 outcome-blinded、categorical-only 审核产物。当前阶段只用于数据收集与检查，不用于训练。

## 产物

- `transition_review_packet.unreviewed.json`：公开盲审包，共 60 条；包含 42 条 reviewed actions、按 action family 抽取的 native controls，以及 6 条隐藏一致性重复项。
- `transition_review.gpt56.json`：GPT-5.6 仅依据公开 packet 给出的 provisional categorical annotations。
- `transition_review_packet.locked_ai_provisional.json`：应用审核结果后的不可静默修改版本。
- `transition_review_inspection.gpt56.json`：锁定后揭盲生成的检查报告。
- `transition_review_packet.hidden_key.json`：本地揭盲 key；被 `.gitignore` 排除，不应发布或用于审核决策。

公开 packet 不暴露 action provenance、legality、stored verifier/target、GTSAM 状态、sampling stratum、source record id 或 overlay path。审核只输出 accept/reject、valid/invalid/inconclusive、relation outcome、公开 evidence refs 和文字理由；不输出数字 reward、score、confidence、probability 或 utility。

## 当前检查结果

- schema 与 lock checksum：通过
- annotation status：`ai_provisional`
- accept / reject：59 / 1
- relation outcome：18 supports、23 inconclusive、19 not applicable
- hidden duplicate consistency：6 / 6 组逐字段完全一致
- numeric annotation output：无
- formal eligible：否
- training ready：否
- training performed：否

这里的 `accept` 表示 executed transition 记录本身有可用、可引用的真实执行证据，不等同于关系被证实。尤其是 counterevidence search：检索执行可以有效，但若返回内容没有直接否定 source claim，relation 仍标为 `inconclusive`。

## 停止条件与下一门禁

本轮按设计停在 provisional inspection，不调用 GPT-OSS-120B，也不导出训练样本。进入训练前仍必须由独立人工 reviewer 在相同公开证据协议下完成并锁定正式标签；模型 provisional 标签与一致性重复项不能替代独立人工 gold 审核。
