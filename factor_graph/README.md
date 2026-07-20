# Exploration Factor Graph

本目录是 `steam_video` 中因子图的唯一设计、GTSAM 实现和实验入口。旧的
`memory_graph` 与 `l15_graph_navigator` 文档只保留接口摘要和链接，避免出现多套
相互冲突的定义。

## 0. 架构决策：GTSAM 是可选 backup，不是主方法

目标主方法已明确为：在固定、共享的 L1/L1.5 Memory Graph 上进行
implicit-world-model-guided reasoning。Memory Graph 是 evidence substrate，不是主要
novelty；主要贡献是预测 reasoning hop 的 future-belief effect，并让 Planner 依赖该预测。

本目录中的 GTSAM 实现继续保留，职责收缩为：

- 可选的 real-evidence belief correction backup；
- competing hypotheses、contradiction、identity consistency 和 correction persistence；
- graph-based baseline、diagnostic oracle、teacher 和 visualization；
- 机制与破坏性对照实验。

GTSAM 不生成 reasoning operation，不做候选排序、graph traversal、world-model prediction、
trajectory preference 或 final-answer grounding。只有执行真实 evidence query 后，经
categorical verifier 得到的 measurement 可以写入 persistent factor state；imagined rollout
绝不能写入。

主导航 CLI 现已提供三个显式模式：

```text
iwm_belief_only             # 默认主方法
iwm_with_gtsam_backup       # categorical trigger 按需纠错
gtsam_always                # baseline / ablation
```

backup trigger 只能是 `contradiction_detected`、`identity_ambiguous`、
`state_history_conflict`、`multiple_competing_hypotheses`、`correction_failed` 或
`long_dependency_unresolved` 等 categorical condition，不使用手工 posterior threshold。
对外只暴露 `accepted/rejected/unresolved/conflicted`；solver 数值保持内部。GTSAM 不可用时
必须显式报告 `backup_unavailable`，不能静默切换 planner 或启用 heuristic ranking。

当前 GTSAM closed loop 与 categorical-triggered backup 均已实现、可运行；主方法继续使用同一个 L1/L1.5
Memory Graph，但必须清除 graph priority/lexical/stable-order 对 winner 的影响。完整定义见
[`l15_graph_navigator/README.md`](../steam_video_new/implicit_world_model/l15_graph_navigator/README.md#architecture-decision-fixed-l1l15-memory-iwm-reasoning-novelty)。

分阶段实施与生产门禁见 [`PLAN.md`](PLAN.md)。

## 1. 它在 graph baseline / backup 中的位置

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

所以因子图不替代 L2，也不替代 implicit world model。在 `gtsam_always` baseline 中它是
探索期间的 belief state；在 `iwm_with_gtsam_backup` 中它只在真实 evidence correction
后按需维护一致性。L2 是执行日志；world model 是候选轨迹的预测器和序数比较器。

这里的 `action` 不是机器人物理动作，而是多跳推理的下一跳：读取事件、沿
temporal/causal/identity/state edge 扩展、验证 relation、查询冲突证据，或
`stop_and_answer`。若接口继续使用 `skill`，它表示 reasoning skill。Planner 只执行
获偏好轨迹的第一跳，然后用真实新证据更新 GTSAM belief 并重新规划。

Planner/LLM 的输入边界定义在
[`l15_graph_navigator/README.md`](../steam_video_new/implicit_world_model/l15_graph_navigator/README.md#41-bounded-reasoning-context)。
它只接收经检索和裁剪的局部子图、categorical belief 摘要和少量候选轨迹；完整
Memory Graph、GTSAM 数值、raw embedding 向量和全部历史不进入 prompt。

该边界现已由 `ReasoningContextBuilder` 接入主闭环和 sibling 分支生成。Planner 使用
受 comparison budget 限制的 staged pairwise tournament，并在 L2 保存检索、裁剪、候选数、
比较数和估算 prompt tokens 的 audit。`GraphReadAction` 继续作为执行兼容类型；Planner
对外同时暴露 `ReasoningHop`，明确它选择的是 multi-hop reasoning 的下一跳。

当前 implicit world model 与 preference planner 可显式选择
`openai/gpt-oss-120b`。该 adapter 只接受 categorical descriptor/preference，模型 JSON
中出现数值会直接失败；没有配置 OpenAI-compatible endpoint 时也不会静默退回 rule
baseline。GPT-OSS 的预测或 preference 不是 factor、真实 evidence 或独立人工 gold。
OpenRouter 可通过 `--reasoning-keys-py /fs/gamma-projects/vlm-robot/keys.py` 读取
`OPENROUTER_API_KEY`；密钥不会进入 prompt、artifact 或日志。

### 1.1 Non-heuristic 边界

本项目的目标是 world-model-guided reasoning，不是 factor-priority-guided traversal。
Factor graph 只维护当前 belief，并用于拒绝 illegal、blocked、重复或无 provenance 的 hop；
embedding 只负责候选召回。两者都不能通过手工 score 或固定顺序决定最终下一跳。

最终选择必须依赖 world model 预测的 categorical future-belief transition，再由 preference
planner 比较候选轨迹。question-direction、lexical overlap、factor priority、stable structural
order 和 rule projection 只能作为明确标记的 candidate-generation/baseline 机制，不能成为
production winner policy。`tie/incomparable` 应扩大证据、继续比较或 abstain，不能默认选择
第一个候选。

正式门禁包括 candidate-order permutation invariance，以及 normal、null、shuffled、frozen、
immediate-only、delayed-belief WM 六组固定条件干预。保持 query、belief、candidate set、planner
和预算不变，只替换 WM prediction；分别报告 action divergence、categorical prediction
accuracy、evidence completeness、reads 和 final-answer accuracy。若破坏 WM 后 action 与结果
不发生系统性变化，就不能声称 planner 依赖 world model。完整定义见
[`l15_graph_navigator/README.md`](../steam_video_new/implicit_world_model/l15_graph_navigator/README.md#43-non-heuristic-planning-boundary)。

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
state support 传播到 identity 时，normal 与 verifier-direct-only 的下一步不同；identity
reject 也确实传播为 state rejection，但去除 factor-priority winner heuristic 后，两 arm
都保守 abstain，所以下一步相同。inconclusive 不激活 factor，且两 arm 下一步相同。报告
因此把 `propagated belief change` 与 `action divergence` 分开，禁止用前者冒充导航收益。
这些 cases 只证明 correction 能持久改变 belief；只有 support case 显示 action divergence，
且它们来自 Phase D fixture、不是独立人工 gold。因此 production 状态仍为 false。

### 13.2 Executed transition supervision pilot

新增的 [`executed_transitions.unreviewed.json`](experiments/phase_e_gpt56_provisional_v1/executed_transitions.unreviewed.json)
从相同 immutable checkpoint 独立执行每个合法 action。每条 target 都在真实 read 与 backend
correction 后，从 before/after belief 重新计算；imagined WM delta、preference 和 solver 数值不进入
target。checkpoint 只存一次，records 使用引用，endpoint-local context 保留 Qwen embedding
metadata 但不保存 raw vector。

该 pilot 明确使用 `persisted_replay`，不是 live Video_Skills runtime 或新的 post-read verifier
测量。现有 8-video / 29-case `ai_provisional` 集合产生 609 条 grounded records，其中 99 条变为
answer-ready；resolved roles 为 temporal 78、identity 20、state-transition 1。只有 2 个
counterevidence actions。因此该 artifact 是 `unreviewed`、`formal_eligible=false` 的 pipeline
与 coverage pilot，不能作为正式 WM accuracy 结果。下一步必须补充 state/counterevidence/
reject/inconclusive cases，并由独立 reviewer 逐条 accept/reject 后锁定。

### 13.3 Balanced correction-sensitive cases 与 verifier provenance

新增的 balanced miner 对 delayed two-hop、identity/state 的
support/reject/inconclusive、contradiction open/resolve、counterevidence、blocked-path
recovery 和 ambiguity/abstention 分别设置 quota。稀缺类别不足时保留 deficit，禁止用普通
temporal case 跨类别回填。当前 11-video / 15-unique-overlay pilot 共选出 54 个 draft
candidates：15 delayed、10 identity-support、1 state-support、4 state-reject、10
empty-counterevidence、4 blocked-path-recovery 和 10 ambiguity。identity negative、state
inconclusive、contradiction 与 useful counterevidence 仍然缺失，因此不是正式 gold set。

产物位于 [`phase_e_gpt56_provisional_v1`](experiments/phase_e_gpt56_provisional_v1/)：

- `balanced_cases.draft.json`：只含待审候选；
- `balanced_case_mining_report.json`：逐类别 available/selected/deficit；
- `balanced_case_review.unreviewed.json`：独立 reviewer queue；
- `video_skills_runtime_verifier_availability.json`：live runtime 可用性检查。

review queue 使用 opaque ID、broad relation stratum、sanitized question/tags 和稳定 hash
乱序，不向 reviewer 暴露 miner 预期的 support/reject/inconclusive outcome；逐条填写
accept/reject、categorical verifier outcome、evidence-chain validity、first-action validity、
delayed-effect 与文字理由。只有完整 review 才能 lock，只有
locked queue 中 accepted cases 才能导出。pre-admission hard-verifier failure 被标记为
`offline_verifier_challenge`：它可以帮助构造 negative review 样本，但在独立审查恢复之前
不是合法在线 graph edge。

executed-transition artifact 现在显式区分 verifier provenance。persisted replay 的 609 条
记录中 `post_read_verifier_count=0`；旧 hard-verifier 只能标为 persisted/pre-read，没有
verifier 的 replay 只能是 `inconclusive`。额外的 live Video_Skills arm 确实执行了 236 次
post-read claim verifier，但 236 次全部返回 `supports`、没有 reject，因此只证明 runtime
wiring 可用，不能证明 relation-level verifier 有效，更不能形成 confusion matrix。support
检查失败一律映射为 `inconclusive`，不得自动解释为 contradiction。

所以 GTSAM backup 的下一项正式实验不能直接消费这些 draft。必须先补齐 negative、
inconclusive、contradiction 和 useful-counterevidence 数据，完成独立 review/lock，再在同一
fixed gold set 上分别报告 verifier-direct-only 与 GTSAM propagation 的 accuracy、read
efficiency 和 action divergence。

### 13.4 Evidence gathering v1：停在训练之前

新增 public evidence packet 与 separate hidden key。public packet 为每个候选提供 endpoint、
bounded temporal context、participant/state、真实 evidence refs，以及不含 raw vector/path 的
opaque Qwen embedding reference；它不包含 teacher probability、hard-verifier、confidence
或 miner 预期 outcome。annotation 必须引用 packet 内可见证据，且 numeric model output 会被
拒绝。

GPT-5.6 当前会话对 54 条 outcome-blinded items 做 categorical provisional review：52 accept、
2 reject；outcome 为 18 supports、1 rejects、17 inconclusive、18 not-applicable。identity 是
9 support / 1 inconclusive，state 是 3 support / 2 inconclusive。特别值得注意的是，miner 的
pre-admission state-reject challenges 经盲审后没有形成 state reject，这再次证明 mined label
不能当训练监督。

随后从 52 条 `ai_provisional` cases 通过 live Video_Skills 收集 994 条 executed-transition
records，其中 978 条 grounded。严格 inspection 发现：只有 44 条 record resolve identity；
没有 state-transition 或 counterevidence resolution；`inspect_state_change`、
`search_counterevidence`、`find_bridge` action family 完全未执行；只有 identity cases 的
proposed actions 获得完整 grounded coverage。354 个 post-read claim-verifier calls 中仅 19
supports、335 inconclusive，但它仍不是经过校准的 relation verifier。

因此本轮明确停止在 data gathering：所有 transition targets 保持 `unreviewed`，报告为
`training_ready=false, training_performed=false, formal_eligible=false`，没有训练或调用
GPT-OSS-120B。完整诊断见
[`balanced_transition_gathering_inspection.json`](experiments/phase_e_gpt56_provisional_v1/balanced_transition_gathering_inspection.json)。

### 13.5 Evidence gathering v2：真实执行 reviewed action

v1 的主要缺陷不是 evidence packet，而是 transition builder 只执行 native candidate pool，导致
审核接受的 state、counterevidence 和部分 bridge first action 从未实际执行。v2 将 native legal
actions 与 accepted reviewed actions 合并，但明确保留 execution boundary：`review_restored`
只能直接读取 packet 中已存在的 endpoint，`offline_diagnostic` 只能做诊断，二者都不能新增
relation 或修改 L1.5 graph。

v2 共收集 1025 条 records、1009 条 grounded records。42 个非 STOP reviewed cases 均达到
exact-action execution 与 exact-action grounding；provenance 为 994 `native_legal`、27
`review_restored`、4 `offline_diagnostic`，`graph_mutated` 始终为 false。resolved role 为
identity 44、bridge 13、counterevidence 10、state_transition 2。state 的 2 条 positive 只来自
post-read strict categorical delta verifier；没有明确 grounded attribute delta 的 3 条仍保持
inconclusive。counterevidence role 的 resolution 表示搜索义务已真实完成，不等价于 verifier
宣告 contradiction。

审核集合没有接受 `find_bridge` action，因此它被报告为 optional unrepresented family，而不是
为了覆盖率手工生成启发式 action。当前唯一正式 blocker 是 transition target 仍为 unreviewed，
需要独立人工逐条 accept/reject 后才能 lock/export；当前仍
`training_ready=false, training_performed=false, formal_eligible=false`，未训练 GPT-OSS-120B。
v2 诊断见
[`balanced_transition_gathering_inspection.review_anchored_v2.json`](experiments/phase_e_gpt56_provisional_v1/balanced_transition_gathering_inspection.review_anchored_v2.json)。

### 13.6 GTSAM backup engineering finalization

optional backup 已完成工程封口，但没有升级为 production-calibrated 方法：

- 主入口通过 `--belief-mode iwm_belief_only|iwm_with_gtsam_backup|gtsam_always`
  显式选择模式；GTSAM 缺失会失败，不会静默切换 backend；
- `iwm_with_gtsam_backup` 使用 verifier-direct categorical update 维护主 belief，并只由
  categorical policy 启动 GTSAM propagation。普通 support 正常更新主 belief但不触发
  GTSAM；reject、已有 contradiction、competing categorical outcomes 或 identity
  inconclusive 会产生明确 audit；
- trigger 与 factor activation 分开：inconclusive 可以触发检查，但不会增加数值 factor；
- checkpoint 保存 categorical navigation state、append-only measurement journal、overlay
  checksum 和 calibration version；不保存 solver marginal。checksum、overlay、mode 或
  calibration 不一致时拒绝恢复；
- sibling branches 使用 checkpoint fork，每个 action 拥有独立 GTSAM session，不发生
  sibling measurement leakage；
- grounded empty counterevidence 只更新“搜索已完成”的 operation state，不生成 relation
  factor；review-restored/offline action 若没有已存在 typed edge，也不能写入 GTSAM；
- `--belief-checkpoint-in/--belief-checkpoint-out` 支持跨进程恢复，restart 前后 categorical
  belief 必须完全一致。

运行主导航：

```bash
PYTHONPATH=. /tmp/steam-video-gtsam-venv/bin/python -m \
  steam_video_new.implicit_world_model.l15_graph_navigator.run \
  --overlay /path/to/causal_temporal_overlay.json \
  --question "..." \
  --belief-mode iwm_with_gtsam_backup \
  --belief-checkpoint-out /path/to/gtsam_checkpoint.json \
  --output-dir /path/to/run
```

finalization pilot 通过 5/5 engineering gates：support 不触发 backup；reject 触发 factor；
identity inconclusive 触发诊断但不加 factor；三类 case restart categorical parity；checkpoint
不暴露 numeric solver state。见
[`gtsam_backup_finalization_v1.json`](experiments/gtsam_backup_finalization_v1.json)。正式状态是：

```text
runtime_ready = true
backup_engineering_complete = true
production_calibrated = false
navigation_benefit_proven = false
```

原因不变：measurement likelihood 仍是明确标记的 pilot calibration，且尚无独立人工 gold
confusion matrix 或 matched-budget navigation benefit。工程完成不能替代实证门禁。
