# Exploration Factor Graph

本目录是 `steam_video` 中因子图的唯一设计、GTSAM 实现和实验入口。旧的
`memory_graph` 与 `l15_graph_navigator` 文档只保留接口摘要和链接，避免出现多套
相互冲突的定义。

分阶段实施与生产门禁见 [`PLAN.md`](PLAN.md)。

## 1. 它在系统中的位置

```text
L1 / L1.5 grounded graph
  → 提供 observation、候选 relation、provenance 与 embedding reference

Factor graph / GTSAM（本目录）
  → 维护数值 belief
  → 融合 identity、temporal、state、dependency 与 contradiction factor
  → 用新证据纠正旧假设，并给出待检查的冲突/不确定关系

Implicit world model
  → 为每个合法 action 预测 observation descriptor 与 belief-delta descriptor
  → 对完整候选轨迹只给 pairwise preference

Planner
  → 为 multi-hop reasoning 选择偏好轨迹的第一个 reasoning hop
  → 执行真实 evidence query / graph read，读取真实 observation
  → 把 measurement 加入 factor graph，更新 belief，再规划

L2
  → 只记录实际执行、真实证据、belief delta 和 preference 的审计轨迹
```

所以因子图不替代 L2，也不替代 implicit world model。它是探索期间的 belief
state；L2 是执行日志；world model 是候选轨迹的预测器和序数比较器。

这里的 `action` 不是机器人物理动作，而是多跳推理的下一跳：读取事件、沿
temporal/causal/identity/state edge 扩展、验证 relation、查询冲突证据，或
`stop_and_answer`。若接口继续使用 `skill`，它表示 reasoning skill。Planner 只执行
获偏好轨迹的第一跳，然后用真实新证据更新 GTSAM belief 并重新规划。

Planner/LLM 的输入边界定义在
[`l15_graph_navigator/README.md`](../steam_video_new/implicit_world_model/l15_graph_navigator/README.md#41-bounded-reasoning-context)。
它只接收经检索和裁剪的局部子图、categorical belief 摘要和少量候选轨迹；完整
Memory Graph、GTSAM 数值、raw embedding 向量和全部历史不进入 prompt。

## 2. Active SLAM 类比

可以把“reasoning + evidence query + correction”理解为 Active SLAM，但变量的
语义不同：

| Active SLAM | 图导航 |
|---|---|
| robot pose / trajectory checkpoint | 当前 belief checkpoint / 探索历史 |
| landmark | entity、event、state 与 dependency hypothesis |
| data association | identity relation / identity track |
| sensor action | evidence query / graph-read skill |
| measurement | skill 执行后返回的真实 L1 observation |
| loop closure | 后续证据纠正早期 identity、temporal 或 state 假设 |
| active exploration | 比较候选 evidence query，执行第一步并重规划 |

关键约束：query 本身不是 factor。只有 query 被执行并返回真实证据后，经过
provenance 与 verifier 的 measurement 才能加入图。模型想象的 observation 绝不
进入答案证据或数值 belief。

## 3. 变量和因子

首个 GTSAM pilot 使用离散二值 hypothesis：

- `identity_a_b`、`identity_b_c`：两个 identity association；
- `conflict_a_c`：A 与 C 是否存在不能属于同一 identity track 的冲突；
- `decoy_conflict`：用于 shuffled-measurement 破坏性对照。

当前/下一阶段的正式变量映射：

- identity：同一实例、类型冲突、同时出现、稳定属性冲突、不可能位移；
- temporal：before、overlap、during 与 cycle consistency；
- state：只有同一 accepted identity track 上有明确 before/after delta 才成立；
- dependency：verified support、explain/enables 与 temporal precedence；
- optional continuous：未来才加入 latent time、位置、轨迹与非线性 measurement。

数值 prior、likelihood、potential 和 posterior 全部属于求解器内部。它们来自经
校准 verifier、传感/检索测量模型或数据统计，而不是 LLM 生成的数字。

## 4. LLM 的严格输出边界

LLM（当前可用 `gpt-oss-120B` 做 transition descriptor 基线）最多输出：

- categorical observation descriptor；
- categorical belief delta；
- `prefer_left`、`prefer_right`、`tie`、`incomparable`；
- 支持该比较的文本 rationale。

LLM 不输出 reward、utility、Q-value、score、confidence 或 probability。LLM 也
不能把 preference 转成伪数值 reward。训练监督来自 sibling trajectory 的人工
或受控 provisional pairwise preference；正式训练必须经过 content lock 与独立
标注门禁。

`categorical.py` 是从内部 marginal 到 LLM-facing `accepted / uncertain /
rejected` 的唯一投影边界。内部数值仍保留在 solver audit 中供校准与调试，但不
放进模型 prompt。

## 5. GTSAM 实现状态

`gtsam_backend.py` 是实际的 `gtsam.DiscreteFactorGraph` wrapper：

- 注册有限基数变量；
- 校验并追加 n-ary `DecisionTreeFactor`；
- 在真实 observation 到达后增量追加 measurement factor；
- 用 GTSAM 做 MAP optimization，并从 joint factor 得到 pilot 的 exact marginal；
- 对 exact state-space 设置硬上限，避免把 pilot 求解器误用于大图；
- 没有安装 GTSAM 时直接报错，不静默切换到自写后端。

GTSAM 4.2.1 的 wheel 在 Python 3.12 暴露 `DiscreteFactorGraph`，但没有暴露
`HybridFactorGraph`/`HybridGaussianISAM` 的 Python 类。当前实现因此是“追加
factor 后重新精确消元”，不是 iSAM2。iSAM2 仍适合未来连续/高斯状态，但不能
拿 SLAM 名称掩盖离散 identity inference 的实际求解路径。

对应的官方接口说明见 [DiscreteFactorGraph](https://gtsam.org/doxygen/a03411.html)
与 [iSAM2](https://borglab.github.io/gtsam/isam2/)。

现有 navigator 的 `FactorGraphBeliefBackend` 仍是生产兼容的 Python loopy
sum-product adapter；本目录的 GTSAM pilot 是替换它的验证路径。在完成 overlay
adapter、规模化消元策略和 parity test 前，不会让环境是否安装 GTSAM 自动改变
默认结果。

## 6. Pilot 与四组实验

Pilot 构造 `A≈B`、`B≈C`，然后主动查询 `A` 与 `C` 的稳定属性/共现冲突：

```text
选择 inspect_a_c_conflict
  → 模拟执行 query，并返回固定的 synthetic measurement fixture
  → 加入 conflict measurement factor
  → identity loop-closure factor 全局传播
  → 较弱的 B≈C 从 accepted 降为 rejected
  → 下一步 inspect B≈C 或 split track
```

四个固定 arm：

1. `correction`：真实 measurement + identity loop closure；
2. `frozen`：不把 observation 加入 posterior；
3. `no_loop_closure`：有 measurement，但不传播到 identity；
4. `shuffled_evidence`：把 measurement 故意接到 decoy variable。

已运行的 `gtsam_correction_pilot_v1.json` 是工程 pilot，不是 benchmark。当前四个
门禁均通过：正常组将目标 conflict 从 rejected 更新为 accepted，并让较弱
identity 变为 rejected；三个破坏性对照没有伪造同样的 identity correction。
这些 prior/likelihood 是明确标记的 pilot 常量，不是已经完成的数据校准结果。
该实验验证的是 GTSAM 更新与纠错机制，不声称已经调用真实 Video_Skills；真实
overlay/read adapter 明确列在下一阶段。

## 7. 运行

PyPI 已提供 Python 3.13 的 GTSAM 4.2.1 wheel，但仓库默认环境使用 NumPy 2.1.3，
而该 GTSAM wheel 要求 `numpy<2`。为避免降级并破坏共享环境，pilot 使用隔离的
Python 3.12 venv（同一官方 4.2.1 wheel）：

```bash
python3.12 -m venv /tmp/steam-video-gtsam-venv
/tmp/steam-video-gtsam-venv/bin/python -m pip install -r factor_graph/requirements-gtsam.txt pytest
PYTHONPATH=. /tmp/steam-video-gtsam-venv/bin/python -m factor_graph.pilot
PYTHONPATH=. /tmp/steam-video-gtsam-venv/bin/python -m factor_graph.experiment \
  --output factor_graph/experiments/gtsam_correction_pilot_v1.json
```

测试：

```bash
PYTHONPATH=. /tmp/steam-video-gtsam-venv/bin/python -m pytest \
  factor_graph/tests/test_gtsam_backend.py -q
```

## 8. Embedding 边界

graph node、query 与 evidence 继续保留 embedding reference，默认由
Qwen3-VL-Embedding-2B 生成，供将来的 semantic navigation、candidate retrieval
和 factor proposal 使用。embedding 只能提出候选或提供可校准特征，不能绕过
identity/state/dependency verifier，也不能直接成为 accepted edge。

## 9. 接下来如何进入真实图

Pilot 通过后仍需要以下步骤，才可替换 navigator 默认后端：

1. 将 `CausalTemporalOverlay` 的 relation hypothesis 映射为稳定 GTSAM key；
2. 将真实 Video_Skills read 的 provenance/verifier 输出映射为 measurement factor；
3. 在同一 locked case set 上做 Python BP 与 GTSAM parity；
4. 对 normal、frozen posterior、no-loop/shuffled factors 做 matched-budget 消融；
5. 用独立人工 identity/state 标注校准 factor，而不是手设 pilot likelihood；
6. 扩大固定 gold case set，补齐非空 verified state/dependency edges；
7. 只有规模、延迟和校准均合格后，才接入默认闭环。

当前最大实证缺口仍是正式数据：现有 Video Holmes 产物没有合格的
`state_transition`，verified dependency 也极少。GTSAM 能维护一致 belief，但不能
凭空创造通过双门禁的 edge；这些 edge 必须来自重新生成的 L1/L1.5 artifact 与
独立审计。

## 10. 真实 Overlay Replay Pilot

`overlay_adapter.py` 已把现有 navigator 的 relation-hypothesis factor specification
编译到 GTSAM：

- 每个 `(edge_id, relation)` 对应稳定的 binary variable；
- relation prior 作为内部 unary prior；
- deterministic、hard verifier、mutex、identity/state implication 与
  temporal/dependency compatibility 保留原 source；
- persisted hard-verifier factor 可以先隐藏，再作为 query 后到达的 measurement
  replay；
- disconnected component 分别做 exact GTSAM inference，避免建立真实 overlay 的
  `2**N` 全联合表；
- 当前最大 component 只有 2 个变量，但 adapter 的硬上限为 20，超限直接失败。

真实 replay 使用四组固定控制：

1. `correction`：按真实 target replay persisted verifier measurement；
2. `frozen`：不更新 posterior；
3. `no_loop`：保留 measurement，但删除 pairwise propagation factor；
4. `shuffled`：把 measurement 确定性移到错误 variable。

运行方法：

```bash
PYTHONPATH=. /tmp/steam-video-gtsam-venv/bin/python \
  -m factor_graph.overlay_experiment \
  --overlay-root memory_graph/outputs \
  --output factor_graph/experiments/gtsam_overlay_replay_v1.json
```

当前固定结果覆盖 15 个真实 overlay：13 个含 persisted verified measurement，2 个
不含；正常 replay 产生 21 个 direct categorical update，shuffled control 与正常组
有 18 个 categorical disagreement。GTSAM 与 Python BP 的 categorical mismatch 为
0；最大 posterior 数值差约 `1e-4`。该数值差来自 legacy BP 每次 `_normalize()` 的
`1e-4` message floor，因此验收是“categorical parity + numeric drift < 1e-3”，不是
伪称两个求解器机器精度完全相同。

运行正确性门禁已经通过，但 `production_ready=false`：

- 真实数据中没有 verified measurement 落在 coupled component，因此 propagated
  update 仍为 0；
- 目标 Video_Skills graph smoke 有 169 条 edge hypothesis、7 条
  `state_transition` candidate、4 条 `transition_support`、27 个 embedding ref，
  但 verified relation 为 0；
- 因此目标 artifact 上 correction、frozen 与 shuffled 都保持 0 update，这是正确的
  abstention，不是 GTSAM 故障。

完整可复现实验记录见
[`gtsam_overlay_replay_v1.json`](experiments/gtsam_overlay_replay_v1.json)。下一步不是
调整 pilot 常量，而是生成至少一个带 verified identity/state 或 dependency
measurement 的 coupled real component，再验证 loop-closure propagation。

## 11. Executed Read → Measurement → GTSAM

Phase C 已实现 `measurement.py` 与 `session.py`：

- `VerifierDecision` 只有 `supports / rejects / inconclusive` 和文本 reasons，不含
  confidence/score/probability；
- `measurement_from_execution` 要求 skill status 为 `executed`、observation 确实在
  invocation outputs 中、action 覆盖 relation 两端、evidence refs 可由此次 read
  grounding；
- graph read 本身不创建 factor；只有独立 verifier decision 通过上述校验后才形成
  `RelationMeasurement`；
- numeric likelihood 只能由版本化的内部 `MeasurementCalibrationRegistry` 提供；
- `MeasurementJournal` append-only、重复 replay 幂等、相反 measurement 并存；
- journal audit record 不包含 numeric belief；GTSAM solver audit 可单独保存内部
  marginal。

真实 Video_Skills integration pilot：

```bash
PYTHONPATH=. /tmp/steam-video-gtsam-venv/bin/python \
  -m factor_graph.executed_measurement_experiment \
  --overlay memory_graph/outputs/video_holmes_structured_state_embedding_smoke_v1/\
memory_graph/mKqiGQrHtW8/causal_temporal_overlay.json \
  --output factor_graph/experiments/gtsam_executed_measurement_pilot_v1.json
```

结果：真实 `retrieve_by_relation` 返回一个 atomic event 和一个 grounded L1 ref；
read 后、verifier 前 factor count 不变；categorical `supports` measurement 到达后只
增加一个 factor，目标 identity 从 uncertain 变 accepted。受控 coupled arm 产生
1 个 direct 和 1 个 propagated categorical update；no-loop 的 propagated update 为
0；shuffled、幂等和 conflicting-evidence retention 均通过。九个 Phase C gate 全部
通过。

该结果仍不是 calibration benchmark：真实 arm 使用 persisted categorical hard
verifier 做 replay，内部 likelihood 标为 `not-calibrated`；真实 arm 尚未出现 coupled
propagation。完整记录见
[`gtsam_executed_measurement_pilot_v1.json`](experiments/gtsam_executed_measurement_pilot_v1.json)。

## 12. Phase D：Post-read Verifier 与真实耦合 Smoke

`post_read_verifier.py` 把“读取图”和“产生 belief measurement”严格分开：

- 只接受 status 为 `executed` 的真实 graph read；
- relation 两个端点都必须已经读取，不能只看到 target 就推断 source；
- decision 只含 `supports / rejects / inconclusive`、evidence refs 和文本 reasons；
- 验证前删除 edge provenance 中旧的 `hard_verifier` / `visual_verification`，防止把
  persisted replay 当成新的观测；
- `measurement_from_execution` 可显式接收此前已经 grounded 的 source refs，同时仍
  拒绝任何不在本次或历史已读集合内的 verifier citation。

fixture 生成方法：

```bash
python -m factor_graph.phase_d_smoke \
  memory_graph/outputs/video_holmes_structured_state_embedding_smoke_v1/\
memory_graph/mKqiGQrHtW8/causal_temporal_overlay.json \
  factor_graph/fixtures/phase_d_coupled_overlay.json
```

该 fixture 来自真实 Video_Skills L1/L1.5 event 与 evidence refs；两个
`Qwen/Qwen3-VL-Embedding-2B` row 会复制到 fixture 自有的 `.npy`，serialized ref 使用
相对路径，由 loader 按 overlay 所在目录解析，因此运行时不依赖旧 smoke 输出。它透明地将源 edge 的两个人物 mention
remap 到固定 accepted identity track，并将 `downward / looking down` 归一为同义状态，
使唯一明确 delta 是 `expression: neutral → focused`。这是 integration smoke，不是
新增人工真值；源 artifact 不会被修改。

运行真实耦合实验：

```bash
PYTHONPATH=. /tmp/steam-video-gtsam-venv/bin/python \
  -m factor_graph.phase_d_experiment \
  --overlay factor_graph/fixtures/phase_d_coupled_overlay.json \
  --output factor_graph/experiments/gtsam_phase_d_grounded_v1.json
```

当前 11 个 gate 全部通过：3 次实际 Video_Skills relation read 均执行成功；post-read
state、dependency 和 contradiction verifier 均产生 categorical support；state 的直接
更新通过 GTSAM factor 传播到 identity；no-loop 阻止传播；shuffled target 改变结果；
dependency measurement 非空；冲突 measurement 被 append-only journal 保留。完整记录
见 [`gtsam_phase_d_grounded_v1.json`](experiments/gtsam_phase_d_grounded_v1.json)。

这仍不是生产校准：内部 likelihood 明确标记为 `not-calibrated`，accepted-track remap
不是独立人工 gold。下一步必须建立多视频固定 case set，完成 identity/state 双盲人工
审计与 categorical verifier confusion matrix，再运行 closed-loop matched-budget 导航
消融。只有这些门禁通过后，才应考虑替换默认 backend。

## 13. Phase E：GPT-5.6 provisional pilot（2026-07-20）

本轮建立了可复现、但不冒充正式 gold 的实证基线，完整状态见
[`phase_e_gpt56_provisional_v1/status.json`](experiments/phase_e_gpt56_provisional_v1/status.json)。
当前会话中的 GPT-5.6 只给出 categorical 判断和轨迹 pairwise preference；没有输出
reward、confidence、probability 或 utility 数字，也没有发生远程模型 API 调用。

- 从 9 个真实 overlay 加 Phase D fixture 挖出 30 条候选，逐条检查后锁定 29 条
  `ai_provisional` cases，覆盖 8 个视频；其中 identity 7、temporal 8、
  state-transition 1、verified-dependency 1、delayed bridge 12。
- blinded sibling packet 的左右顺序改为基于 packet/case/index 的确定性 hash 随机化，
  避免固定 left/right 位置泄漏。29 条比较全部形成 categorical provisional labels，
  并导出 83 条无数字 reward 的训练记录。
- 视频隔离的 preference pilot 在 6 条 held-out comparisons 上 exact accuracy 为 66.7%；
  样本很小，只用于验证数据与训练链路。
- 29-case matched-budget ablation 中，semantic/event/native/verified 分别为
  55.2%/58.6%/58.6%/58.6%，factor-graph direct 为 75.9%，rule lookahead 为
  82.8%，shuffled 为 72.4%。lookahead 优于 direct 且 relation shuffle 有伤害。
- normal lookahead 与 frozen-posterior 同为 82.8%，所以第三个诊断 gate 失败。这表明
  当前 matched evaluator 尚未把“执行真实 read → categorical measurement → GTSAM belief
  correction → replan”真正接入策略差异，不能把提升归因于在线 belief correction。
- 单视频 45 条旧 L1 admitted candidates 的 GPT-5.6 临时审计得到 identity strict
  precision 55.9%、state-transition 42.9%，均未达到 90%。admission baseline 的
  confusion matrix 只是“候选存在即 supports”的诊断，不是 post-read verifier matrix。

因此本轮结论仍为 `runtime_pass=true, production_ready=false`。正式门禁还需要：多视频
独立人工 identity/state 标注；真实 post-read verifier 的三分类 confusion matrix；以及
更多非空、correction-sensitive 的 state/dependency/counter-evidence cases。闭环接入后的
严格复核结果见下一节。

### 13.1 Executed-read GTSAM closed loop v2

严格复核后，v1 的“3/3 gate”已经撤回：唯一的读取差异来自 categorical verifier 直接
消除 missing role，并不能归因于 GTSAM。v2 做了以下隔离修复：

- GTSAM backend 独立、持久地维护 acquired evidence、missing roles、relation grounding、
  priority、blocked edges 和 contradiction，不再从 compatibility backend 重建每一步；
- 初始化和后续更新均删除 persisted `hard_verifier` 状态，只有当前 session 中真实执行后
  的 measurement 能改变 relation belief；
- corrected belief 之后重新计算 `BeliefDeltaDescriptor`，不再复用 correction 前的 delta；
- 新增 `categorical_verifier_direct_only_lookahead`，用于隔离 verifier 直接更新与 GTSAM
  propagation；
- VERIFY action 显式携带 typed relation endpoint，避免 provenance reread 无法归因到 edge；
- accuracy proxy、ready+evidence accuracy、真实读取次数和 action divergence 分开报告，
  不再聚合成 lexicographic pass/fail。

```bash
PYTHONPATH=. /tmp/steam-video-gtsam-venv/bin/python -m \
  steam_video_new.implicit_world_model.l15_graph_navigator.workflow evaluate \
  --cases factor_graph/experiments/phase_e_gpt56_provisional_v1/\
cases.locked_ai_provisional.json \
  --output factor_graph/experiments/phase_e_gpt56_provisional_v1/\
matched_ablation_gtsam_closed_loop_v2.json \
  --allow-ai-provisional --gtsam-closed-loop
```

固定 29-case set 的严格结果见
[`matched_ablation_gtsam_closed_loop_v2.json`](experiments/phase_e_gpt56_provisional_v1/matched_ablation_gtsam_closed_loop_v2.json)：

- 38 条 journal records 中只有 14 条激活 factor；24 条 `inconclusive` 正确 abstain；
- 产生 2 个 direct categorical changes 和 1 个 propagated change；
- normal 与 verifier-direct-only 的 evidence-completion proxy、ready+evidence accuracy、
  mean reads 和全部 29 条 action sequences 完全相同；
- 因此该固定集合仍未证明 GTSAM propagation 带来导航收益。

为确认机制确实能影响重新规划，新增三个 derived grounded correction-sensitive cases，见
[`correction_sensitive_navigation.json`](experiments/phase_e_gpt56_provisional_v1/correction_sensitive_navigation.json)。
state support 传播到 identity、identity reject 传播到 state rejection 时，normal 的下一步
均与 verifier-direct-only/frozen/shuffled 不同；inconclusive 不激活 factor，且 normal 与
verifier-direct-only 下一步相同。它证明了 propagation→replan 机制，而不是正式 accuracy
收益；这些 cases 来自 Phase D fixture，不是独立人工 gold。因此生产状态仍为 false。
