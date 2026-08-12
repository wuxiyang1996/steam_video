# CG-Bench Grounded Navigation Pilot v1（历史产物）

本 pilot 从本地 CG-Bench multi-clue QA 构建，当前只生成和审计数据，不训练模型。

## 规模

- 256 个 multi-clue cases；
- 199 个视频；
- 1,302 条 action-conditioned transitions；
- 256 条完整轨迹 categorical preferences；
- train / validation / test：195 / 38 / 23 cases，按 video ID 确定性切分；
- 媒体时长隔离：本 pilot 没有发现越界视频；
- answer leakage：未发现；
- numeric reward-like fields：未发现。

## 文件

- `navigation_dataset.unverified.json`：planner input、候选 reads、真实 interval、categorical transition targets 和 pairwise preference；不含正确答案。
- `terminal_targets.hidden_key.json`：正确答案、answer key 和人工 clue identity；禁止暴露给 planner/reviewer，受 `*.hidden_key.json` 忽略规则保护。
- `build_report.json`：规模、切分、隔离和门禁报告。

## 状态与替代协议

- annotation status：`ground_truth_anchored_unverified`（旧 v0.1 schema）
- formal eligible：false
- training ready：false
- training performed：false
- observation descriptor：等待 Qwen-VL grounded read
- embedding：保留 `Qwen/Qwen3-VL-Embedding-2B` placeholder

新增的闭环准备产物：

- `matched_ablation.unexecuted.json`：256 cases 的四臂 evaluation inputs；尚未执行 answer model；
- `blinded_visual_audit.unreviewed.json`：32 个 video-disjoint 视频、196 个 interval 的盲审 packet；
- 对应 `*.hidden_key.json`：GT 与公开 packet 分离，不得提供给 reviewer/planner。

真实 Qwen grounding 已通过 smoke，并扩展到下述 32-video audit subset。其余 1,106
transitions 仍保持 pending。人工审核不再是正式门禁。

CG-Bench clue intervals 不是穷尽标注；GT 外区间可能仍包含相关证据。因此当前 matched
controls 与 trajectory preferences 都是 unverified candidates，不是可直接训练的负例。GT
preference 不是数字 reward，也不声称 clue path 已经证明 identity、state、causality 或最终
answerability。

## 32-video grounded audit 实测

`qwen_audit_subset_v1/` 已完成：

- 32 个视频、39 cases、196 intervals；
- Qwen-VL grounded observations：196/196，0 failure；
- Qwen3-VL-Embedding-2B：`196 × 2048`，全部归一化并带 checksum refs；
- contact sheets：196/196；
- GPT-5.6 Sol blinded provisional decisions：196/196 completed，0 API failure；
- media alignment：195 accept、1 inconclusive；
- descriptor grounding：172 accept、22 reject、2 inconclusive；
- clue relevance：89 relevant、8 unrelated、1 inconclusive；
- control relevance：56 unrelated、39 relevant、3 inconclusive。

这说明 GT 外 control contamination 很高，原始 matched-control arm 不能使用。严格整-case
filter 只保留 5/39 cases（4 train、1 validation、0 test），因此还不足以做正式消融或训练。
旧 `gpt56_filtered_provisional` 只保留为 control-construction failure 的诊断记录，不应训练或
报告为正式结果。v0.2 GT-only 协议移除 matched controls，直接采用 CG-Bench 人工 clue 与
answer ground truth；可选人工/GPT 检查只用于媒体与 descriptor QA。
