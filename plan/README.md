# Multi-Path IWM + Planner 方案

## 2026-07 GPT-5-mini matched pilot

真实 `openai/gpt-5-mini` 已在 10 条 fingerprint 完整的 CG-Bench cases 上完成
五臂 matched-budget pilot。初始 cohort 冻结为 12 条，其中 2 条因 source graph 在
compile gate 后漂移而在模型运行前排除，没有重编 gate 或补样。

工程闭环已经通过：

- 10 cases × 5 arms 全部完成；
- runtime errors 与 method failures 均为 0；
- IWM → multi-path Planner → real read → belief correction → replan 可运行；
- WM joint-chain outcome retention 为 10/10；
- 没有 numeric reward、Top-K 或 hidden-label planner feedback。

科学结论仍为负：

- WM answer accuracy 为 0；
- mean clue recall 为 0.0091，oracle 为 0.4498；
- WM 只在 3/10 cases 执行真实 read，只在 1/10 命中任意 clue；
- 4 条 delayed cases 上 horizon-2 WM 全部 abstain，delayed success 为 0；
- immediate-only 在 delayed slice 平均执行 1 次 read，但 clue recall 仍为 0；
- WM 相对 no-WM/shuffled/immediate 的 action divergence 分别为
  0.3/0.3/0.5，说明 IWM 会改变动作，但还没有转化为有效导航收益。

因此当前状态应写为：

```text
infrastructure_complete = true
scientific_validation_passed = false
training_performed = false
```

本次仅导出 4 条 `training_allowed=false` 的 unreviewed executed-trace candidates。
现在不训练 9B。failure-slice inspection 随后确认了四个独立问题：原始 CG-Bench
存在可审计的 question/choice source defects；terminal no-read transition 可被模型错误
改写；完整 preferred frontier 与“必须唯一 first hop”之间形成 Planner deadlock；旧 cohort
在两跳预算下连 oracle 也无法完成 delayed cases。

修复后的 v2 协议是：

- source defect 只能通过 exact video/qid exclusion 隔离，并记录 source checksum，不能猜测
  或自动改写答案；
- `stop`/`abstain` 不进入 evidence-acquisition rollout，no-read action 完全绕过模型；
- 多条 preferred reasoning paths 全部保留，由独立 categorical evidence scheduler 只调度
  一次最能区分 hypotheses 的真实 read；
- conditioned outcomes 只按 exact descriptor equivalence 无损分组，不做 Top-K；
- scientific cohort 必须通过真实 graph adjacency 下的 oracle two-hop coverage gate，而不只
  检查 clues 是否存在于视频 observation horizon。

## 目标

本方案的核心不是让 Planner 每一步选择一条 reasoning path 并删除其余路径，而是：

> IWM 预测多条 reasoning path 的未来 belief effect；Planner 持续维护一组竞争路径，只决定当前最值得执行的共享 evidence action。执行真实读取后，所有路径分别修正 belief，再重新规划。

最终目标是将 IWM 与 Planner 蒸馏到 9B 级开源模型。模型不输出 scalar reward、utility、Q-value、概率或人工分数；外部决策接口保持 categorical preference，但模型内部必须学习比当前 categorical delta 更丰富的 transition representation。

## 总体闭环

```text
L1/L1.5 grounded evidence memory
        ↓
persistent multi-path reasoning pool
        ↓
为每条 path 编译 local legal actions
        ↓
IWM 预测 path × action 的一至二跳未来
        ↓
按共享 first-hop 组成 joint action trees
        ↓
Planner 比较完整 joint trees
        ↓
执行一个真实 evidence read
        ↓
对所有 persistent paths 分别 belief correction
        ↓
删除过期 imagined rollout，保留 viable paths
        ↓
重新调用 IWM 与 Planner
```

IWM 的 imagined state 永远不能直接写入 persistent belief。只有真实读取的 evidence 可以改变持久状态。

## SelectStream-style streaming video 主线

本项目主线应明确对齐 SelectStream-style streaming video memory，而不是只做
offline long-video QA。核心问题是：

> 在视频持续到来、未来问题未知、memory/context budget 固定的条件下，系统应决定
> 何时写入记忆、保留或合并哪些 evidence、query 到来时如何读取历史，以及何时
> wait/read/answer。

对应的在线闭环是：

```text
stream segment arrives
        ↓
surprise-driven L1 write / skip
        ↓
fixed-capacity memory consolidation
        ↓
question-independent L1.5 temporal/correlation graph update
        ↓
query-conditioned graph retrieval over retained memory
        ↓
calibrated evidence injection into frozen VLM / 9B controller
        ↓
decision: inspect current | read past | wait | answer | abstain
```

这里的 L1/L1.5 memory 是固定容量 evidence substrate。系统可以保留当前
observation 的直接视觉上下文，但历史只能通过 compact retained evidence 和
query-conditioned graph read 进入模型。随着 stream horizon 增长，context 和 GPU
memory 不应随完整历史线性增长。

### 与现有实现的对齐

当前仓库已经有部分 SelectStream-style primitives：

- `memory_graph.adaptive_windowing`：question-independent surprise windows；
- `memory_graph.video_l1`：video-only L1 extraction，可接 adaptive window provider；
- `memory_graph.selectstream_policy`：bounded-memory keep/merge/evict policy；
- `full_graph_iwm`：fixed-capacity graph、legal actions、matched closed-loop arms；
- CG-Bench fixed cohort：graph freeze 后 hidden evaluator 才读取 clue intervals。

下一步不是重写这些模块，而是补齐 streaming benchmark adapter、stream-time
supervision schema 和 SelectStream-style evaluation protocol。

### Benchmark 分工

训练、验证和测试必须按 video-disjoint split 隔离。最小防泄漏单位是原始视频，
不是 clip、window、query 或 trajectory。

主训练/验证来源：

- **CG-Bench**：继续作为 grounded evidence supervision 基准。它提供人工
  `clue_intervals`、hidden answer key、video-disjoint split 和 fixed L1/L1.5 gate。
- **StreamingBench**：作为 streaming QA / answer timing 主数据源，监督
  当前时刻下的 answer-now、wait、read-past 或 inspect-current 行为。
- **OVO-Bench**：作为 online video understanding 主数据源，覆盖 backward tracing、
  real-time visual perception 和 forward active responding。

只用于 testing / out-of-domain generalization：

- Video-MME；
- Video-Holmes；
- VRBench。

这些 testing benchmarks 不进入 SFT、preference optimization、prompt tuning、
adapter selection 或 early stopping。若未来要把其中某个 benchmark 纳入训练，
必须重新发布独立的 video-disjoint split 和 contamination report。

### Streaming supervision record

StreamingBench 与 OVO-Bench 应转换为与 CG-Bench 兼容的公开/隐藏双文件协议：

```yaml
public_record:
  benchmark: streamingbench | ovo_bench | cg_bench
  video_id: ...
  split: train | validation | test
  observed_until_s: ...
  current_time_s: ...
  question: ...
  stream_state:
    recent_visible_window: ...
    retained_l1_node_keys: [...]
    local_l15_edges: [...]
    acquired_evidence_refs: [...]
  candidate_actions:
    - inspect_current
    - read_past(node_key)
    - follow_temporal(edge_key)
    - wait_until(next_timestamp)
    - answer

hidden_record:
  answer_key: ...
  target_timestamp_or_interval: ...
  evidence_alignment: ...
  response_timing_label: ...
```

Public planner input 不能包含 answer、hidden clue identity、future target interval
或未读取 node 的完整 evidence value。Hidden key 只允许 evaluator、offline
supervision exporter 和 leakage checker 读取。

### SFT / post-training / validation / testing

SFT 使用 train videos：

```text
question + stream state + memory checkpoint + candidate action
  → observation descriptor
  → categorical belief delta
  → action decision: inspect_current | read_past | wait | answer | abstain
```

CG-Bench 提供 clue-acquisition transition；StreamingBench/OVO-Bench 提供
streaming timing 与 online action decision。现有 CG-Bench positive-only
transition corpus 可以继续用于 scoped SFT smoke 和 descriptor distillation，但
不能单独声明为 runtime categorical IWM 训练完成。

Post-training 只使用 train videos 上的 grounded sibling trajectories：

```text
same stream checkpoint:
  trajectory A
  trajectory B
→ prefer_A | prefer_B | tie | incomparable
```

偏好监督来自 grounded future outcome，例如 evidence coverage、answer timing、
terminal correctness、false early answer avoidance 和 read efficiency。禁止把
embedding similarity、graph distance、Top-K rank、model self-score 或人工 dense
reward 写入数据协议。

Validation 使用 CG-Bench / StreamingBench / OVO-Bench 的 validation videos，
只用于 early stopping、schema/prompt/model selection、calibration 和 ablation
debugging。Test 使用三套主 benchmark 的 held-out test videos，再加
Video-MME、Video-Holmes、VRBench 做 out-of-domain report。

### SelectStream-style 对照组

正式结果必须在相同问题、相同 stream prefix、相同 frozen evidence memory、
相同 legal action compiler 和相同 read budget 下比较：

```text
SelectStream-style L1/L1.5 memory + IWM/Planner
recent-window only / SimpleStream-style baseline
no-memory
random or shuffled memory
graph-only retrieval
oracle evidence ceiling
```

至少报告：

- answer accuracy；
- answer timing accuracy；
- evidence / clue recall；
- complete evidence coverage；
- read efficiency；
- wait/read/answer action accuracy；
- delayed success；
- current-scene perception retention；
- peak memory / context tokens / latency；
- abstention 与 schema/transport failure。

主 claim 应绑定到固定 memory/context budget 下的 horizon-scaling：随着视频从短流
增长到长流，Selective L1/L1.5 evidence memory 是否比 recent-window、naive
retrieval 和 graph-only traversal 更稳定地保留可验证 evidence chain。

## 三类状态

### 1. Grounded evidence memory

L1/L1.5 提供 question-independent evidence substrate：

- addressable evidence nodes；
- caption、embedding、timestamp 与 provenance；
- temporal edges；
- soft correlation edges；
- 从当前 graph state 可执行的 legal reasoning hops。

L1/L1.5 correlation 是导航候选，不是校准概率、因果事实、identity 真值或最终 action score。

### 2. Persistent reasoning paths

Planner 长期维护多个竞争路径。每条路径至少保存：

```text
hypothesis
acquired real evidence
entity / event / state bindings
supported and contradicted claims
missing reasoning links
open contradictions
current graph cursor
real action history
path status
```

建议的 path status：

```text
active | contradicted | answerable | suspended | merged
```

路径不能因为当前较弱而被任意删除：

- `incomparable` 与 `tie` 路径继续保留；
- 暂未执行的路径继续保留；
- 只有真实 evidence 明确反驳时才标记为 `contradicted`；
- 只有 hypothesis、grounded bindings、缺失依赖和历史状态都等价时才允许合并；
- 不同 hypothesis 不能仅因选择了相同 first-hop 而合并。

### 3. Imagined rollouts

Imagined rollout 是 IWM 为规划临时生成的未来：

```text
path_i + action_a
    → predicted observation
    → predicted belief-event patch
    → possible next legal actions
```

它不能成为最终答案证据。每次真实 read 后，旧 rollout 必须全部失效，并根据修正后的 persistent paths 重新生成。

## IWM V2

### 输入

每个 transition prediction 应条件化于：

```text
question
current persistent path and hypothesis
acquired grounded evidence
missing variables / reasoning links
contradictions
candidate legal action
target node address descriptor
local temporal and correlation context
reasoning prefix
```

不得向 IWM 泄漏未读取 node 的完整 evidence value、hidden clue interval、答案或 evaluator target。

### 输出

当前 `advanced / unchanged / regressed` 等 categorical delta 应继续保留为控制与审计接口，但不能作为完整的 IWM state。

IWM V2 应预测 structured belief-event patch：

```yaml
predicted_observation:
  target: node alias
  event_or_state: structured descriptor
  entity_bindings: predicted bindings
  temporal_binding: before | after | overlaps | unknown
  evidence_role: identity | state | bridge | counterevidence | other
  alternatives:
    - possible grounded outcome
    - possible inconclusive outcome

predicted_belief_patch:
  supported_claims: []
  contradicted_claims: []
  newly_bound_variables: []
  opened_dependencies: []
  resolved_dependencies: []
  unresolved_dependencies: []
  affected_hypotheses: []

categorical_audit:
  observation_outcome: support | counterevidence | identity_evidence |
    state_evidence | bridge_evidence | empty | inconclusive
  progress: advanced | unchanged | regressed
  contradiction_change: opened | resolved | unchanged
  frontier_change: opened | closed | shifted | unchanged
  answerability_after: ready | not_ready | abstain
```

模型仍不输出数字。训练时的 logits、loss 和内部连续 representation 属于正常模型计算，不等于让模型产生 numeric utility。

### Multi-hop imagined transition

对于每条 persistent path，IWM 预测 horizon-one 或 horizon-two chain：

```text
z_hat(t+1) = T(z_t, action_t)
z_hat(t+2) = T(z_hat(t+1), action_(t+1))
```

第二跳必须条件化于第一跳的 predicted outcome，而不是复制一个与 prefix 无关的静态 action descriptor。

当前阶段保持短 horizon，并在每个真实 action 后重新规划，避免长 rollout drift。

## Multi-Path Planner

### Persistent trajectory pool

Planner 是多路径状态的唯一 owner，负责：

- 初始化 answer hypotheses 或 competing interpretations；
- 保存每条路径的 persistent grounded belief；
- 维护 active、suspended、contradicted、answerable 与 merged 状态；
- 将一次真实 observation 分别应用到所有相关路径；
- 保留暂时 incomparable 或 recoverable 的路径；
- 使所有旧 imagined continuations 失效；
- 调用 IWM 生成新 rollout。

Factor graph / GTSAM 仅作为可选 belief-correction backup。它可以融合真实 evidence 和处理全局一致性，但不能给 action 排名，也不是主方法的必要组件。

### Joint action trees

Planner 不应逐个选择“某条 hypothesis 最喜欢的 action”。相同 first-hop 应跨所有 persistent hypotheses 组成 joint tree：

```text
first-hop: read node_31
  ├─ hypothesis A conditioned outcome
  ├─ hypothesis B conditioned outcome
  └─ hypothesis C conditioned outcome
```

这样 Planner 比较的是一个真实读取对整个 competing hypothesis pool 的未来作用，而不是过早承诺某个答案。

每棵 joint tree 必须覆盖所有当前可扩展 hypothesis-conditioned outcomes。缺失 coverage 时应 fail closed，不能静默丢弃不方便的路径。

Horizon-two 时，joint tree 按共享 first-hop 分组，而不是要求所有 hypotheses
拥有完全相同的第二跳。一个 first-hop tree 应保留每个 hypothesis 下所有合法的
one-/two-hop continuations。这样即使不同 hypothesis 的第二跳分化，Planner 比较的
仍然是“现在执行哪个共享真实 action”，不会把只覆盖部分 hypotheses 的二跳 chain
误当成完整候选。

### Learned partial preference

Planner 比较完整 predicted joint chains，并输出：

```text
prefer_left | prefer_right | tie | incomparable
```

这是偏序，不要求生成强制全排序：

- `tie` 不使用 node ID 或输入顺序打破；
- `incomparable` 分支继续保存在 trajectory pool；
- 不采用 heuristic Top-K；
- 不用固定规则规定 identity、state、temporal 或 bridge 谁总是更重要；
- 不把 cosine similarity 或 correlation strength 当作 action utility；
- Planner 的 preference 必须来自 learned trajectory representation。

每一步只执行一个真实 action，但其余 reasoning paths 仍然存在。若 preferred frontier
包含多个 first hops，Planner 不再把它直接折叠成 abstain，而是调用 categorical evidence
scheduler 选择最能区分 competing hypotheses 或打开下一条 grounded continuation 的 read。
这个 scheduler 不删除路径、不输出数字、不使用 node 顺序或 heuristic Top-K；只有当它也
明确返回 incomparable 时才 fail closed。

## 真实 evidence correction

同一真实 observation 必须对不同 hypothesis 产生不同 belief correction。例如：

```text
real observation: node_31 shows Bob, not Alice

Alice path → contradicted
Bob path   → identity resolved
unrelated path → remains possible
```

更新顺序必须是：

```text
execute real read
→ attach provenance
→ update every persistent path independently
→ update optional GTSAM backup from real evidence only
→ invalidate imagined rollouts
→ compile new legal actions
→ call IWM
→ call Planner
```

禁止将 predicted observation、predicted relation 或 imagined belief 当作 acquired evidence。

同一个真实 graph action 被执行后，所有 persistent paths 都必须同步该 action 的
navigation state，包括 acquired evidence、cursor、cursor history、read budget 和
action history。随后才对各 path 分别应用 hypothesis-conditioned semantic correction。
如果只移动 source path 的 cursor，下一轮 legal-action space 会人为分裂，joint tree
将无法覆盖所有 hypotheses。

## 9B 模型设计

建议使用一个共享的 Qwen 9B backbone，通过两个 task mode 或 heads 完成 IWM 与 Planner：

### Transition mode

```text
<IWM_TRANSITION>
path state + hypothesis + action + local graph context
→ structured belief-event patch
```

### Comparison mode

```text
<PLAN_COMPARE>
trajectory pool + multiple joint imagined chains
→ partial preference over first-hop groups
```

本地开源模型不应继续自回归生成大量重复 JSON。建议采用：

- categorical classification heads；
- node/entity/role pointer heads；
- structured transition representation；
- trajectory representation head；
- pairwise/listwise preference head；
- constrained serialization 作为审计输出。

这可以降低 provider incomplete JSON、schema invalid、重复 token 开销和长尾延迟。

## 训练信号

训练目标不能主要来自当前大模型的未经验证 zero-shot 输出。优先级应为：

1. 执行 action 后获得的真实 observation；
2. dataset GT clue intervals、clue identity 与 clue order；
3. 真实 evidence 对各 hypothesis 的 support/counterevidence；
4. 相同 read budget 下的 grounded trajectory outcome；
5. terminal answer correctness 与完整 clue coverage；
6. 审核后的 teacher structured annotations。

Teacher 可以补充 descriptor、binding 和 preference proposal，但 teacher prediction 不能自动成为真值。

### 必需的数据类型

- 同一 evidence 支持一个 hypothesis、反驳另一个；
- identity/state hard negatives；
- empty、reject、inconclusive controls；
- counterevidence；
- 两条路径暂时 incomparable、后续 evidence 才能区分；
- 第一步无直接收益、第二步补全关键 evidence 的 delayed cases；
- 错误路径被真实证据纠正、替代路径恢复的 cases；
- 相同 first-hop 对多个 hypotheses 产生不同 outcomes 的 joint examples。

当前 positive-only transition corpus 可以用于 descriptor pipeline smoke 或受限 SFT，但不能单独训练 runtime IWM/Planner。

## 建议训练阶段

### Stage 1：transition representation

使用 train-split grounded executed transitions 训练：

- observation descriptor；
- node/entity/role pointers；
- temporal bindings；
- hypothesis-conditioned belief patch；
- categorical audit heads。

数据来源优先级：

1. CG-Bench train：clue acquisition 与 grounded evidence progress；
2. StreamingBench train：streaming answer timing、inspect-current/read-past/wait；
3. OVO-Bench train：backward tracing、real-time perception、forward active responding。

### Stage 2：multi-path preference

在 train videos 上构造完整 joint trajectory pairs/sets，训练：

- pairwise preference；
- listwise/partial ordering；
- tie 与 incomparable；
- hypothesis coverage；
- candidate permutation invariance。

可以使用 DPO、IPO 或其他 ordinal preference objective，但监督必须来自 grounded future outcome，而不是人工 utility 数字。

### Stage 3：closed-loop refinement

当离线 transition calibration 和 preference accuracy 达标后，再考虑 on-policy collection 或轻量 RL。RL 信号应尽量使用：

- terminal answer correctness；
- grounded clue completion；
- false hypothesis elimination；
- matched-budget success。

不应先设计大量手工 dense rewards 来替代 world-model learning。

## 验证协议

代码测试通过只代表闭环接口可运行，不等于方法有效。正式验证需要固定、
多视频、split-safe streaming cohort，并比较：

```text
SelectStream-style L1/L1.5 memory + IWM/Planner
recent-window only / SimpleStream-style baseline
no-memory
random or shuffled memory
graph-only retrieval
immediate-only
oracle evidence ceiling
```

至少分别报告：

- answer accuracy；
- answer timing accuracy；
- clue recall / complete clue coverage；
- current-scene perception retention；
- transition observation calibration；
- belief-patch accuracy；
- first-hop action divergence；
- delayed-planning advantage；
- read count 与 read efficiency；
- correct-hypothesis survival；
- false-hypothesis elimination；
- joint-chain hypothesis coverage；
- post-read belief divergence；
- abstention 与 schema/transport failure。

成功门禁应要求：

1. IWM transition prediction 在 held-out executed transitions 上可校准；
2. Planner preference 与 grounded future trajectory outcome 一致；
3. IWM 优于 no-WM、shuffled-IWM 和 immediate-only；
4. delayed cases 上存在稳定优势；
5. 多路径维护不会过早删除正确 hypothesis；
6. 结果不是来自 heuristic Top-K、固定 tie-break 或 hidden GT leakage。

## 当前实现与下一步

当前 V1 已有：

- L1/L1.5 graph 与 legal actions；
- persistent competing hypothesis trajectories；
- horizon-one/two imagined transitions；
- joint-chain construction；
- categorical pairwise/setwise preference；
- real read、hypothesis-conditioned correction 与 replan；
- matched five-arm evaluator；
- optional isolated GTSAM backup。

当前缺口主要是：

- categorical transition representation 过于有损；
- runtime-aligned hard negatives 与 delayed transitions 不足；
- 现有 grounded corpus 的 label distribution 过于单一；
- persistent multi-trajectory frozen cohort 尚未证明导航收益；
- OpenRouter JSON transport/schema failure 较多；
- 尚未训练或蒸馏 9B IWM/Planner。

### GPT-5-mini H2 smoke 的最新结论

修复后的三例、五臂、两次 real-read smoke 已完整跑通，零 runtime error，且没有
训练。multi-path joint chains 对全部 hypothesis-conditioned outcomes 的保留率为
1.0；每一步都由 categorical scheduler 从仍然存活的完整轨迹集合中选择一次真实
read，未使用 Top-K、固定 tie-break 或数字 utility。

但该结果只证明闭环可执行，尚未证明 IWM 有效：

- IWM / no-WM / shuffled-IWM 的 mean clue recall 都是 0.333；
- immediate-only 为 0.667；
- 五组 answer accuracy 都没有形成正结果；
- IWM observation/progress calibration 只有 0.305 / 0.218；
- IWM 虽然改变 action sequence，并在真实 read 后产生 hypothesis belief 分化，
  但这些分化尚不够准确。

日志还分离出两个此前混在一起的问题。第一，shuffled-IWM 曾错误地打乱 no-read
control；现在只打乱真实 read consequence。第二，旧 oracle 可以从全图任意节点
开始，而真实方法只能从模型给出的 entry frontier 开始。现在 matched oracle 共享
同一 entry frontier：三例平均 clue ceiling 从 compile-time 的 1.0 降为 0.833，
其中 gluten case 在两次 read 内最多只覆盖一半 clue。因此 compile-time full-graph
oracle 只能表示 L1/L1.5 substrate ceiling，正式 case eligibility 必须再经过
model-localized entry frontier 的 budget-matched oracle preflight。

在 localized oracle 完整可达的两例上，IWM clue recall 为 0.5，与 no-WM 和
shuffled-IWM 相同，而 immediate-only 为 1.0。当前最具体的错误是把
`hand holds object` 这类粗粒度 address descriptor 想象成“已解决全部问题 roles、
已经 answerable”。这说明下一步应优先补齐 grounded action-conditioned transition
与 hypothesis-discrimination supervision，并单独报告 entry recall；不能通过新的
planner heuristic 掩盖 upstream transition over-crediting。

### 证据链修复（已实现）

为避免把 IWM 错误与 L1 证据降质混在一起，runtime 现在保持两条严格分离的链：

```text
imagined transition -> 只供 Planner 比较，不能进入 acquired evidence
executed legal action -> real observation -> belief correction -> terminal answer
```

此前 belief correction 能看到 `action_kind / participants / states /
state_change`，但 terminal answer 只看到一句扁平 caption。现在 executed real
observation 会作为独立对象贯穿 trace，answer selector 同时接收 timestamp、
provenance 和上述 structured L1 fields，不再在最后一步丢失信息。

同时加入可插拔的 locked raw-clip reread adapter。它只允许在合法 action 已执行后
替换同一 video、同一 node ID、同一 timestamp 的 observation descriptor；未审核
artifact 默认 fail closed，reread 不能新增 graph address，也不能参与 imagined
rollout。这样后续可以用 Qwen-VL 对模糊 L1 window 做高保真重读，而不改变 L1/L1.5
导航图或泄漏 hidden answer。

代码同时提供 blinded node-reread artifact builder：输入仅为 frozen node、原视频和
该 node 的时间窗，不输入 question、choices 或 GT answer；产物默认为
`unreviewed / training_allowed=false`，独立锁定后 runtime adapter 才会接受。

评估现在另外报告两个互不替代的门禁：

- `localized_entry_preflight`：在模型 localization 之后，以相同 read budget 跑
  evaluator-only oracle，区分“证据不可达”和“Planner 选错”；
- `evidence_sufficiency_evaluator_only`：直接判断实际读到的内容是否足以区分 GT
  answer，而不是把 timestamp overlap 当作证据充分。

两者都使用 hidden evaluator 信息，因此明确标记
`fed_back_to_planner=false`，绝不进入 IWM/Planner input。分析脚本还会把 imagined
`ready / resolved_roles / advanced` 与真实 correction 后 belief 对齐，导出
`over_crediting / under_crediting / consistent / inconclusive` categorical candidates。
这些记录默认 `unreviewed` 且 `training_allowed=false`，审核锁定前不能训练。
hidden sufficiency evaluator 使用独立的 uncached transport；代码会拒绝把它接到
Planner response cache，避免 evaluator answer 污染未来 IWM/Planner 数据。

model-backed smoke 还暴露出大 transition batch 偶发 `finish_reason=length`。runtime
现已在 transport/schema 失败时递归拆分 shared-action-group batch，并完整恢复全部
request；拆分只改变请求封装，不删除 candidate、hypothesis 或 reasoning path，审计中
单独记录 `adaptive_transport_split_count`，因此这不是 heuristic Top-K。

对原三例 smoke 的 post-hoc 重分析得到 36 条 hypothesis-conditioned executed
transition 对照：19 条 provisional `over_crediting`、7 条 `consistent`、7 条
`under_crediting`、3 条 `inconclusive`。这不使用 hidden clue/answer，reference 是真实
read 后的 corrected hypothesis belief；记录仍需独立审核，不能直接训练。

新 rich-evidence 路径的一例 GPT-5-mini 五臂 replay 已完成且零 runtime error。该例中
localized clue oracle 和部分方法能达到 clue recall 1.0，但独立 evaluator 仍判断实际
证据 `insufficient`；因此旧的 temporal clue overlap 确实高估了 answer evidence
sufficiency。其余两例的完整远程 replay 遇到 OpenRouter 长 JSON 延迟/截断，已用于
复现并修复 adaptive split，但尚未形成新的完整三例结果，不能宣称方法收益改善。

当前 runtime 已冻结为 `steam-multi-trajectory-runtime/v1.0`，对应 pilot/run artifact
schema v0.2。第一批 36 条记录已导出到
`datasets/iwm_runtime_transition_audit_v1/`；packet 使用独立
`steam-iwm-transition-audit/v0.1` schema，并逐项报告 case、video、failure slice 与
review coverage。它刻意不提供 aggregate reward 或一条综合 gate boolean。

这 36 条全部来自 test split，只能用于 diagnosis。新的 train collection 已冻结为
40 cases / 40 videos，selection 使用 stable hash、每视频一例，不读取 question text
或 hidden answer。对应 manifest 位于
`datasets/iwm_runtime_train_collection_v1/collection_manifest.json`；在完成独立
L1/L1.5 构建、真实执行和审核前，所有状态保持 pending，不能启动 9B SFT。

建议接下来的实现顺序：

冻结 train collection 中通过 graph gate 的 31 例已全部跑完，且 `runtime_error=0`、
`method_failure=0`；因此 infra 可以工作，但 zero-shot IWM 尚未通过实证。WM-guided
answer accuracy / clue recall 为 0.129 / 0.145，no-WM 为 0.161 / 0.266。
763 条 executed WM transition 中，realized `inconclusive` 占主导，原 micro exact
accuracy 会让经常预测 control 类的 shuffled arm 虚高。现在评估同时报告 pooled
confusion matrix、macro-recall balanced accuracy 与 categorical over-credit FDR；
WM 的 outcome/progress balanced accuracy 为 0.332/0.419，对应 over-credit FDR 为
0.759/0.857，predicted-ready FDR 为 0.974。所以下一步数据目标必须是 grounded
action-conditioned negative/inconclusive/counterevidence/delayed transitions，而不是
增加 Top-K、固定 tie-break 或手工 reward。

1. 冻结 `structured belief-event patch` V2 schema；
2. 编写 V1 executed-transition 到 V2 training record 的 adapter；
3. 补采 negative、inconclusive、counterevidence 和 delayed cases；
4. 实现共享 9B backbone 的 transition/comparison 两种训练模式；
5. 先做小规模 overfit、schema 与 multi-path coverage smoke；
6. 在 held-out split 上评估 transition 与 preference；
7. 最后运行固定 cohort 的五臂 closed-loop 实验。

## 先实现双任务，再由数据门禁决定训练

IWM transition SFT 与 multi-path Planner SFT 的代码应一起实现，但不能因为
训练入口存在就绕过数据门禁。两者共享同一个 Qwen 9B backbone，第一阶段分别使用：

```text
iwm_transition_adapter
planner_preference_adapter
```

分离 LoRA adapter 可以避免当前 transition 数据量较大、preference 数据极少时相互
干扰，也便于分别检查 transition calibration 和 trajectory preference。数据充分后，
再比较两个 adapter 与统一 multi-task adapter。

现有 670 条 grounded transitions 可以用于：

- V2 dataset adapter 和 masked-supervision smoke；
- train/validation/test video-disjoint 检查；
- tiny-set overfit；
- checkpoint、constrained decoding 与 runtime adapter 验证；
- observation descriptor 和已有 categorical 字段的 scoped SFT。

它们不能单独证明 runtime IWM 有效，因为当前记录全部是 positive clue advance，
`answerability_after` 没有有效类别变化，而且缺少 runtime graph target、
hypothesis-conditioned counterevidence、hard negatives 和 delayed outcomes。

当前 local-choice packet 只有一个 test case 和三个 pair comparisons，因此只能验证：

- multi-path joint-tree serialization；
- pairwise preference contract；
- left/right permutation consistency；
- `tie` / `incomparable` 输出；
- Planner readiness fail-closed。

它不得用于 Planner post-training。真正 Planner SFT 必须等待 train-split、多视频、
grounded joint-trajectory preference packet。

### Masked supervision

V2 中无法从现有记录可靠获得的字段必须显式标为未监督，不能用空字符串、
`none` 或 `unchanged` 冒充负标签：

```yaml
observation_descriptor:
  value: grounded descriptor
  supervised: true

hypothesis_support:
  value: null
  supervised: false

contradiction_change:
  value: null
  supervised: false
```

训练 loss 只覆盖 `supervised=true` 的字段。每个字段同时保留 label provenance，
使 dataset GT、executed observation、human-locked review 和 teacher proposal 可以
分别审计。

### 分层验证

1. **静态数据 gate**：检查 split、hidden leakage、provenance、mask、alias 和
   joint hypothesis coverage。
2. **tiny overfit gate**：分别验证两个 LoRA adapter 的 loss、checkpoint 和结构化
   decode；synthetic Planner 数据只用于代码测试。
3. **现有数据 gate**：IWM 在 516/74/80 transition split 上报告字段覆盖和
   held-out 结果，同时明确 positive-only 分布。
4. **closed-loop replay gate**：确认 trained IWM 会改变 action-conditioned
   prediction，不同 hypotheses 会分化，shuffling 会影响 action，并且 imagined
   belief 不会进入 persistent belief。
5. **正式方法 gate**：补齐 train-split grounded joint preferences 后，重新运行
   IWM、no-WM、shuffled-IWM、immediate-only 和 oracle 五臂实验。
