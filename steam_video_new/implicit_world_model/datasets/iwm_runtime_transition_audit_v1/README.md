# IWM Runtime Transition Audit v1

该目录冻结第一批来自真实 multi-trajectory closed-loop execution 的 transition
calibration review 数据。它用于找出 IWM 为什么把弱证据预测成强 belief progress，
不是训练集。

## 数据来源

- model：`openai/gpt-5-mini`
- substrate：冻结的 CG-Bench L1/L1.5 graphs
- cases / videos：3 / 3
- split：36/36 records 都来自 test，只用于诊断；不能用于 SFT
- executed real reads：6
- hypothesis-conditioned comparisons：36
- hidden clue 或 answer：未包含
- reference：每次真实 read 后 hypothesis-conditioned corrected belief

`automatic_failure_slice` 只是程序根据 predicted delta 与 corrected belief 的差异生成
的诊断类别，不是人工标签：

- `over_crediting`：预测 ready、预测 resolved role 仍然缺失，或预测 advanced 但真实
  belief signature 未变化；
- `under_crediting`：预测 unchanged，但真实 correction 改变了 belief；
- `consistent`：当前可比较字段一致；
- `inconclusive`：不足以归入上述类别。

## 当前分布

```text
over_crediting  19
under_crediting  7
consistent       7
inconclusive     3
```

## 文件

- `review_packet.unreviewed.json`：blinded、严格 schema 的 36 条审核记录；
- `readiness_report.json`：case/video/类别/审核状态分别报告的门禁；
- `review.gpt56.provisional.json`：隐藏 automatic slice 后的 GPT-5.6 categorical
  provisional review；
- `review_inspection.gpt56.json`：review coverage 与 automatic/model confusion；
- schema：`../../full_graph_iwm/transition_audit.schema.json`；
- runtime contract：`../../full_graph_iwm/runtime_contract.v1.json`。

## 边界

当前 packet 覆盖四类 failure slice，但只有 3 cases / 3 videos，所有记录均为
`unreviewed`。因此：

- 不能导出 SFT；
- 不能声称 transition calibration 改善；
- 不能将 automatic slice 当 reward 或 preference；
- 旧 34-case single-belief replay 不能伪装成 multi-hypothesis correction 数据。

GPT-5.6 已完成 36/36 blinded provisional review，结果为 21 条
`over_crediting`、8 条 `under_crediting`、7 条 `consistent`。automatic
`over_crediting` 的 19 条全部被确认；automatic `under_crediting` 的 7 条和
`consistent` 的 7 条也全部被确认，原 3 条 `inconclusive` 被进一步分成 2 条
over-crediting 和 1 条 under-crediting。该一致性支持 failure diagnosis，但 GPT-5.6
不是独立人工 reviewer，所以 `formal_gate_eligible=false`、
`training_allowed=false`。

下一批必须从至少 30 个 frozen cases 中继续收集，并包含 weak-related-but-
insufficient、identity unresolved、state delta incomplete、hypothesis discrimination、
counterevidence 和 delayed second-hop completion。之后进行独立 categorical review，
再由单独的 locked adapter 决定哪些字段可监督。

`training_performed=false`。
