# GTSAM Belief Graph Implementation Plan

## Architecture decision（2026-07-20）

主研究方法使用固定、共享的 L1/L1.5 Memory Graph；novelty 集中在 IWM-guided reasoning，
而不是 graph construction 或 propagation。GTSAM 不再是默认 planner dependency，而是可选
belief-correction backup，以及 baseline/diagnostic。

后续 runtime 统一为 `BeliefCorrector` 接口，并提供：

- `iwm_belief_only`：默认主方法；
- `iwm_with_gtsam_backup`：真实读取和 categorical verifier 后，在明确冲突/纠错条件下
  调用；
- `gtsam_always`：baseline 与消融。

GTSAM backup 只消费 real executed evidence，不消费 imagined transition；不生成候选、
不排名 reasoning operation、不产生 preference、不提供答案证据。Planner 必须继续依赖
IWM predicted future-belief trajectory。backup 不可用时显式返回
`backup_unavailable`，不得静默改变策略。

下一实现顺序：

1. 冻结共享 L1/L1.5 graph、candidate set 和 evidence budget；
2. 清除 graph priority、lexical 和 stable-order winner leakage；
3. 将现有 GTSAM backend 包装为 optional `BeliefCorrector`；
4. 实现 categorical backup triggers 和 correction audit；
5. 比较 `iwm_belief_only`、`iwm_with_gtsam_backup`、`gtsam_always`，分别报告 backup activation、
   correction persistence、action divergence、evidence completeness、accuracy 和成本。

这份计划把“求解器能运行”“能读真实 overlay”“能根据真实 query measurement
纠错”和“能提高导航结果”分成不同门禁，禁止用 synthetic 成功替代真实数据结论。

## 已实现的 GTSAM baseline 架构

```text
CausalTemporalOverlay
  → stable relation-variable keys
  → calibrated priors + compatibility factors

Planner / LLM
  → categorical transition descriptors
  → pairwise trajectory preference only

Executed Video_Skills query
  → grounded observation + provenance + verifier result
  → measurement adapter
  → append factor to GTSAM belief graph
  → categorical belief projection
  → replan
```

数值 prior、likelihood、marginal 和 calibration 只属于 factor graph。LLM 不产生
reward、score、confidence、probability 或 utility。

## Phase A：求解器 Pilot——完成

- 实现 `GTSAMDiscreteBeliefGraph`；
- 支持 binary/n-ary factor、MAP、exact marginal 与 state-space guard；
- identity loop-closure synthetic pilot；
- correction、frozen、no-loop、shuffled 四组控制；
- GTSAM 4.2.1 隔离环境测试通过。

验收：四个 synthetic mechanism gate 全通过，但结果只证明更新机制。

## Phase B：真实 Overlay Replay——完成

- 将现有 relation hypothesis 编译为稳定 GTSAM variable；
- 按 connected component 做 exact inference，避免 `2**N` 全联合枚举；
- replay persisted hard verifier，且未验证 prior 不升级为 measurement；
- 与 Python BP 做 categorical parity 和 bounded numeric drift 检查；
- 在 15 个真实 overlay 上运行 correction/frozen/no-loop/shuffled。

验收：runtime gate 全通过。报告输入带 SHA256；当前仍不是导航 benchmark。

## Phase C：真实 Measurement Adapter——完成

实现从一次真实 Video_Skills graph read 到 factor 的严格转换：

1. 输入必须包含 executed action、returned observation IDs、provenance 与 verifier；
2. query/imagined observation 不能创建 factor；
3. verifier 的 `passed/failed/inconclusive` 映射为不同 categorical measurement type；
4. likelihood 来自离线校准表，不从 LLM 文本解析数字；
5. 每个 factor 记录 action ID、evidence refs、verifier version 和 calibration version；
6. replay 同一 measurement 必须幂等；冲突 measurement 必须保留而不是覆盖历史。

验收数据：至少包含一个 verified identity measurement、一个 identity conflict，以及
一个同 identity track 的 verified before/after state delta。

当前已实现 strict schema、内部 calibration registry、append-only/idempotent journal、
action/observation/evidence grounding 校验，以及真实 Video_Skills
`retrieve_by_relation`。Phase D 又补上 post-read categorical verifier：它只读取本次
执行后已获得的两个端点、grounded L1 refs 和 relation contract，并在验证前移除旧
artifact 的 persisted verifier 字段，因此 replay 结论不能伪装成新 measurement。

## Phase D：Coupled Grounded-Graph Correction——Integration Smoke 已完成

在真实 artifact 上构造非空 coupled component：

- identity transitivity + type/co-occurrence/attribute/motion conflict；
- state-transition implies accepted identity；
- dependency implies temporal precedence；
- contradiction 与 positive relation mutex；
- temporal-cycle constraint。

验收：正常组必须产生 direct update 和至少一个 propagated update；frozen 不更新；
no-loop 只有 direct update；shuffled 不得复现正确 correction。若数据没有 verified
measurement，实验应 abstain，而不是注入人工正例。

已从真实 Video_Skills overlay 中固定抽取两个有 L1 grounding 和
`Qwen/Qwen3-VL-Embedding-2B` refs 的 event，透明地 remap 到同一个 accepted track，
并生成 identity、明确 expression before/after state delta、transition dependency 和
contradiction 四个 relation。该 fixture 明确标注为 derived integration smoke，不是
独立 gold label，也不改变源 artifact。

真实 `retrieve_by_relation` + post-read verifier + GTSAM 实验已通过 11 个 gate：
normal 的 state measurement 同时产生 direct state update 和 propagated identity
update；frozen 不注入 measurement；no-loop 阻止传播；shuffled 与 normal 不同；
dependency 非空；相互冲突的 categorical measurements 都保留在 journal。LLM-facing
记录只有 `supports/rejects/inconclusive`，没有数值字段。

## Phase E：Closed-Loop Navigation Pilot

- 从相同 immutable belief checkpoint 生成合法 sibling queries；
- world model 只输出 observation/belief-delta descriptor；
- preference model只输出四分类比较；
- 执行获偏好轨迹的第一步真实 skill；
- measurement adapter 更新 GTSAM；
- 重新生成 action 并规划。

matched arms：semantic-only、event-only、native L1 candidate、verified dependency、
IWM preference、null/shuffled/frozen WM、immediate-only、delayed-belief，以及独立的
GTSAM-always / no-loop / shuffled-measurement baseline。

验收：固定 gold case set、相同 graph-read budget、独立 preference/answer 标注，以及
所有答案证据可回溯到 L1。

### Phase E.1：Bounded next-hop reasoning（已实现）

- `ReasoningContextBuilder` 只向 world model / planner 暴露 query-conditioned 局部子图、
  categorical belief、少量 candidate hops 和短期 history；
- `GraphReadAction` 保留为执行兼容层，`ReasoningHop` 明确表示 multi-hop reasoning 的下一跳；
- 独立限制 local nodes、local edges、candidate hops 和 pairwise comparisons；
- 在 comparison budget 可覆盖的范围内保留最大候选集，并对保留轨迹做完整 all-pairs；
  不再使用顺序敏感的 staged tournament；
- candidate order 用稳定内容摘要规范化；多个不同 first-hop 同为 undominated 时显式
  `abstain`，不取第一个候选；
- GPT preference 只看到匿名 imagined outcomes，不看到 operator、target/edge ID 或 action rationale；
- L2 记录 retrieved/dropped node/edge、候选数、comparison budget、估算 prompt tokens；
- Qwen embedding 只在 LLM 外部参与召回排序，raw vector 不进入 prompt；
- implicit world model 和 preference planner 可显式使用 `openai/gpt-oss-120b`，且只允许
  categorical JSON；模型输出任何数值均失败，endpoint 缺失时不静默 fallback。

OpenRouter live smoke 已通过：在 Phase D grounded fixture 上，3-node / 2-edge /
2-hop / 1-comparison 预算产生两次 GPT-OSS 请求，选择 `inspect_state_delta`，真实读取后
belief 变为 answer-ready。两次 prompt token 分别为 804 和 445；这些 transport 计数只用于
输入成本审计，不进入 preference 或 belief。`stop_and_answer` 已固定为 deterministic no-op，
不会调用模型，也不能凭 imagined transition 解决 missing role。

下一实验步骤是做 context-budget ablation，并在独立 multi-video gold cases 上分别报告
answer accuracy、evidence completeness、reads 和 action divergence。

### Phase E.2：Executed transition supervision（pipeline 已实现，gold 未完成）

- 每个 case/action 都从重新初始化的 immutable belief checkpoint 执行，禁止 sibling 间
  correction state 泄漏；
- observation descriptor 不再由 provisional WM 填写；role 来自 legal operation contract，
  node kind 和 target IDs 来自真实 read；
- `BeliefDeltaDescriptor` 从 persisted before/after belief 重新推导 hypothesis、frontier、
  contradiction、path、recovery、uncertainty 和 answerability 类别；
- dataset checkpoint 规范化存储，action 只引用 checkpoint，避免重复完整 belief；
- 默认 `unreviewed/formal_eligible=false`；逐条独立 accept/reject review 之前不能锁定或训练导出；
- 8-video/29-case provisional pilot 得到 609 个 grounded records，但 state-transition role
  resolution 只有 1 条、counterevidence action 只有 2 条，当前覆盖严重不平衡。

同时必须清除 heuristic leakage：candidate generator 只负责 legality、provenance、blocking、
deduplication 和预算，不能通过 lexical/question-direction/factor-priority/stable-order 决定
winner。建立 candidate permutation test，以及 normal/null/shuffled/frozen/immediate-only/
delayed-belief WM intervention。只有破坏 WM prediction 会系统性改变 action trajectory 并降低
reasoning outcome，才能支持 world-model-guided reasoning claim。

## Phase F：生产门禁

只有同时满足以下条件才替换默认 Python backend：

- identity/state 人工审计 strict precision ≥90%；
- 非空 verified state/dependency edges；
- GTSAM 与 reference inference categorical parity；
- coupled real-graph propagation gate 通过；
- locked navigation case set 上 factor-guided arm 优于破坏性对照；
- 延迟、component size、内存和失败回退策略符合预算；
- GTSAM 不可用时显式失败或由配置选择 backend，不能静默改变结果。

当前阻塞已经从“GTSAM 机制与 coupled 数据接口”后移到“独立标注、校准和导航收益”：
Phase D 的 derived smoke 证明端到端链路能工作，但不能替代 ≥90% strict precision 的
独立人工 identity/state 审计，也不能替代多视频 locked gold navigation benchmark。
因此目前正确状态仍是 `runtime_pass=true, production_ready=false`。

### Phase E provisional checkpoint（2026-07-20）

已完成：

- 8 视频、29 case 的 `ai_provisional` 固定集合；
- 确定性随机左右顺序的 blinded preference packet；
- 29 条 GPT-5.6 categorical preference 与 83 条无数字 reward 的训练记录；
- categorical transition/preference pilot；
- 29-case matched-budget ablation，以及 frozen/shuffled 破坏性对照；
- 单视频 45-relation provisional identity/state 审计和 native-admission baseline matrix。

首轮兼容 backend 诊断门禁通过 2/3；随后 v1 closed-loop 的“3/3”已因 verifier/GTSAM
混淆被撤回。v2 已完成独立持久 GTSAM belief、persisted-verifier 隔离、corrected delta、
verifier-direct-only arm，以及 support/reject/inconclusive/conflict persistence 测试。

29-case v2 中 normal 与 verifier-direct-only 在 accuracy proxies、reads 和全部 action
sequences 上完全相同，所以正式结论是“尚无 GTSAM 导航收益”。三个 derived coupled
mechanism cases 则确认 support/reject propagation 会改变下一步，而 inconclusive 不会。
下一优先级是将这种 correction sensitivity 扩展到多视频独立人工 gold cases，再做 verifier
confusion matrix 和统计门限预注册。

状态文件：
[`experiments/phase_e_gpt56_provisional_v1/status.json`](experiments/phase_e_gpt56_provisional_v1/status.json)。
