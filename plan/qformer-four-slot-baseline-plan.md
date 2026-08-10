# 双 Q-Former 四槽特征 Baseline：编码、训练与验证计划

## 1. 目标与范围

本计划定义第一版 node-aligned 双 Q-Former baseline：每个 L1 memory node 由四类预计算
feature embeddings 表示；QF1 使用 8 个问题无关 query tokens 融合四槽特征，QF2 使用
8 个问题条件化 query tokens 生成供后续 IWM 使用的 proposal tokens。

```text
每个 node 的 4 个 feature slots
        │ [B, 4, source_dim]
        ▼
QF1：8 个 question-independent queries
        │ U_i [B, 8, 768]
        ▼
冻结并缓存 U_i
        ▼
QF2：8 个 question-conditioned queries
        │ P_i [B, 8, 768]
        ▼
node relevance score / IWM proposal interface
```

第一版采用分阶段训练，不使用最终 QA loss 端到端更新 QF1/QF2：

\[
L_{QF1}=L_{\text{4-slot distillation}}
\]

\[
L_{QF2}=L_{\text{multi-positive retrieval}}
\]

QF1 和 QF2 的 query 数量相同，但 query embeddings、Transformer 参数与职责均不共享。

## 2. 当前仓库状态与实现缺口

设计目标中的四槽输入为：

\[
F_i=
[
f_{\text{caption}}(i),
f_{\text{entity/state}}(i),
f_{\text{visual}}(i),
f_{\text{time}}(i)
]
\in\mathbb R^{4\times d}
\]

当前代码尚未完整实现该接口：

- `memory_graph/embedding.py` 当前把 event text、participants 和 states 序列化为一段文本，
  生成单个 2048 维 `embedding_ref`；
- `reasoning_v2/evidence/contracts.py` 的 `EvidenceAddress` 当前也只携带一个
  `embedding_ref`；
- 独立 caption、entity/state、visual 和 time feature sidecars 尚未形成统一的四槽 artifact；
- 当前直接 materialize 的主要是 text-based event embedding，独立 visual clip embedding
  仍需补齐；
- `EvidenceValue` 属于真实 READ 后才可见的 grounded 内容，不能直接用于构造未读 node 的
  pre-read feature，否则会破坏事实边界。

因此实现顺序必须从四槽 feature contract、materializer 和 leakage audit 开始，不能直接跳到
Q-Former 训练。

## 3. 四类 feature 定义

### 3.1 Caption feature

\[
f_{\text{caption}}
=
\operatorname{Proj}_{cap}
(\operatorname{TextEnc}(\text{bounded pre-read caption}))
\]

它表示 node 的粗粒度事件描述。输入是经过边界审计的有界 caption，模型侧使用 caption
embedding，而不是把完整 caption 直接当 Q-Former query token。

### 3.2 Entity/state feature

\[
f_{\text{entity/state}}
=
\operatorname{Proj}_{ent}
(\operatorname{TextEnc}(\text{bounded entities + states}))
\]

它可以表示人物、物体、entity role、可见属性、当前状态和经过允许的状态变化。V1 可将这些
字段稳定序列化为有界文本，再使用冻结 text encoder 得到一个 embedding。

未读 node 只能使用 pre-read contract 明确允许的 bounded fields；不得从隐藏的
`EvidenceValue` 中提取实体或状态。

### 3.3 Visual feature

\[
f_{\text{visual}}
=
\operatorname{Proj}_{vis}
(\operatorname{VideoEnc}(\text{node clip}))
\]

它表示人物外观、物体、动作、场景和运动变化，可来自多帧视觉 encoder pooled feature、
video encoder clip embedding 或经过边界审计的 VLM event embedding。第一版冻结 visual
encoder，只训练输入 projector 与 Q-Former。

不能用现有 text-only event embedding 冒充 visual slot。

### 3.4 Time feature

\[
f_{\text{time}}
=
\operatorname{TimeMLP}
\left(
[
t_{start}/T,
t_{end}/T,
(t_{end}-t_{start})/T,
t_{center}/T
]
\right)
\]

精确 timestamps 仍保存在 metadata 中供合法性判断与审计；learned time embedding 只用于模型
计算。

## 4. 四槽输入组织

四类 feature 的原始维度可以不同：

```text
caption embedding       [d_caption]
entity/state embedding  [d_entity]
visual embedding        [d_visual]
time values             [4]
```

分别投影到统一 hidden dimension 后堆叠：

```python
caption_token = caption_proj(caption_embedding)
entity_token = entity_proj(entity_state_embedding)
visual_token = visual_proj(visual_embedding)
time_token = time_mlp(time_values)

features = torch.stack(
    [caption_token, entity_token, visual_token, time_token],
    dim=1,
)  # [B, 4, d]

features = features + feature_type_embeddings
```

数学表示为：

\[
\widetilde F_i=F_i+E_{type}
\]

每个 feature 携带独立 type embedding：

```text
caption       + TYPE_CAPTION
entity/state  + TYPE_ENTITY_STATE
visual        + TYPE_VISUAL
time          + TYPE_TIME
```

四槽应堆叠，不应先平均：

\[
\frac{1}{4}\sum_{r=1}^{4}f_r
\]

会丢失 feature 来源和 modality identity。Q-Former 的 cross-attention 负责学习融合方式。

每个 node 同时携带：

```text
feature tokens: [B, 4, d]
validity mask:  [B, 4]
```

缺失 slot 使用 validity mask，不把零向量当作真实 feature。例如：

```text
caption       valid=1
entity/state  valid=0
visual        valid=1
time          valid=1
```

全部 slot 均无效时必须 fail closed。

## 5. QF1：问题无关的 node feature fusion

QF1 使用 8 个 learned queries：

\[
U_i
=
QF_1(Q^{base},\widetilde F_i,M_i)
\in\mathbb R^{8\times d}
\]

其中：

- `Q_base` 是 8 个 learned query tokens；
- `F_i` 是 4 个 feature tokens；
- `M_i` 是 4 个 feature 的 validity mask；
- QF1 不读取 question；
- QF1 不做跨 node attention；
- 输出始终与原 node ID 绑定。

```python
u_i = qf1(
    query_tokens=q_base,              # [B, 8, d]
    encoder_hidden_states=features,   # [B, 4, d]
    encoder_attention_mask=feature_mask,
)
```

推荐 V1 配置：

```text
hidden_size = 768
num_queries = 8
num_layers = 4
```

输入只有 4 个 tokens、输出为 8 个 tokens，所以 QF1 不是 token-count compressor，更准确的
职责是 `question-independent node feature fusion`：将四种异构 feature 映射为统一、可缓存的
latent slots。

### 5.1 QF1 训练目标

使用轻量 slot decoder 从 `U_i` 恢复四个输入槽：

\[
\widehat F_i=D(Q^{slot},U_i)
\]

`Q_slot` 包含四个 modality-specific decoder queries：

```text
caption reconstruction query
entity/state reconstruction query
visual reconstruction query
time reconstruction query
```

唯一 loss 是 masked slot cosine distillation：

\[
L_{QF1}
=
\frac{
\sum_{r=1}^{4}M_{ir}
\left[1-\cos(\widehat f_{ir},\operatorname{sg}(f_{ir}))\right]
}{
\sum_{r=1}^{4}M_{ir}
}
\]

```python
target = features.detach()
predicted = slot_decoder(u_i)  # [B, 4, d]

cosine_loss = 1 - F.cosine_similarity(predicted, target, dim=-1)
loss_qf1 = (
    cosine_loss * feature_mask
).sum() / feature_mask.sum().clamp_min(1)
```

第一版不增加独立 time、node contrastive、diversity 或 masked-feature loss。query collapse、
time preservation 等先作为 validation diagnostics；只有观测到对应 failure mode 后才增加 loss。

训练完成后删除 slot decoder，冻结 QF1，并导出每个 node 的 `U_i` cache。

## 6. QF2：问题条件化的 proposal encoder

QF2 使用另一组独立的 8 个 learned queries。问题 token 与 query tokens 通过 self-attention
交互，query positions 再 cross-attend 到冻结的 `U_i`：

\[
P_i
=
QF_2([Q^{cond};E_q(q)],U_i)_{query\ positions}
\in\mathbb R^{8\times d}
\]

```python
qf2_inputs = torch.cat([q_conditioned, question_tokens], dim=1)

p_i = qf2(
    inputs_embeds=qf2_inputs,
    encoder_hidden_states=u_i,
)[:, :8]
```

推荐 V1 配置：

```text
hidden_size = 768
num_queries = 8
num_layers = 2
```

第一版 question encoder 可复用冻结的 Qwen embedding encoder，先以一个 pooled question
token 作为稳定 baseline；token-level question encoder 作为后续扩展。

### 6.1 Node score

先计算每个 proposal token 与 question embedding 的匹配：

\[
a_{ik}=\cos(W_pP_{ik},W_q q)
\]

再聚合为 node score：

\[
s_i
=
\tau_q\log\sum_{k=1}^{8}\exp(a_{ik}/\tau_q)
\]

`logsumexp` 允许不同 query token 分别捕获 caption、entity、visual 或 temporal clue，不要求
每个 query 都匹配同一 node label。

### 6.2 QF2 训练目标

对同一问题的全部合法 evidence nodes 使用 multi-positive retrieval：

\[
L_{QF2}
=
-\frac{1}{|\mathcal P|}
\sum_{p\in\mathcal P}
\log
\frac{
\exp(s_p/\tau)
}{
\exp(s_p/\tau)+
\sum_{n\in\mathcal N_{trusted}}\exp(s_n/\tau)
}
\]

训练标签必须区分：

- `positive`：与人工 clue interval 明确匹配的 node；
- `ignore`：同视频中未标注、边界不确定或尚未审计的 node；
- `trusted negative`：跨视频 in-batch negatives，或经过人工/可信协议确认的 same-video hard
  negatives。

CG-Bench clue interval 之外的 node 当前是未标注而非正式 negative，不能直接放入训练
denominator 当作错误候选。

## 7. 分阶段训练

### Stage 1：QF1

```text
4 feature embeddings
→ QF1（8 independent queries）
→ slot decoder
→ 重建 4 个 feature embeddings
```

训练 QF1、四个输入 projectors、feature type embeddings 和 slot decoder。冻结原始 feature
encoders。

### Stage 2：QF2

```text
4 feature embeddings
→ Frozen QF1
→ cached U_i
→ QF2（8 question-conditioned queries）
→ node score
→ multi-positive retrieval loss
```

只训练 QF2、question projector 和 scoring head。QF1 全部冻结，不从最终 QA loss 回传。

### Optional Stage 3：低学习率联合微调

V1 默认不执行。只有满足以下条件时再考虑：

- QF1 四槽 preservation gate 已通过；
- QF2 在 frozen `U_i` 上已收敛；
- 直接使用四槽输入的 QF2 明显优于 QF1 cache 路径，证明 QF1 是实际瓶颈。

若启用，QF1 学习率不超过 QF2 的 0.1 倍，并保留 QF1 slot-distillation replay。

## 8. 代码布局

在 `steam_video_new/implicit_world_model/reasoning_v2/qformer/` 下新增：

```text
qformer/
├── __init__.py
├── contracts.py
├── feature_store.py
├── materialize_features.py
├── model.py
├── losses.py
├── dataset_qf1.py
├── dataset_qf2.py
├── train_qf1.py
├── train_qf2.py
├── export_qf1.py
├── retrieval.py
└── evaluate.py
```

职责划分：

- `contracts.py`：四槽名称、shape、validity mask、artifact version；
- `feature_store.py`：matrix sidecar 与 manifest 的加载、checksum 和 node-row 绑定；
- `materialize_features.py`：从批准的 pre-read fields、视频片段和 timestamps 生成四槽 artifact；
- `model.py`：输入 projectors、QF1、slot decoder、QF2 和 scoring head；
- `losses.py`：masked slot distillation 与 multi-positive retrieval；
- `dataset_qf1.py`：以 node 为单位加载四槽数据；
- `dataset_qf2.py`：以 question 为单位构造 positive/ignore/trusted-negative candidate pool；
- `train_qf1.py`、`train_qf2.py`：两个独立训练入口；
- `export_qf1.py`：冻结 checkpoint 后导出 `U_i` cache；
- `retrieval.py`：proposal-token matching 和 node-score aggregation；
- `evaluate.py`：统一 gates、matched baselines 和报告导出。

建议新增 `requirements-qformer.txt`，显式声明 PyTorch、Transformers、NumPy 和训练依赖；模型
模块保持 lazy optional imports，避免破坏现有非神经 runtime/tests。

## 9. Feature artifact

建议按 slot 分开持久化，避免不同源维度强行混入一张 matrix：

```text
feature_store/
├── caption.npy          # [N, d_caption]
├── entity_state.npy     # [N, d_entity]
├── visual.npy           # [N, d_visual]
├── time.npy             # [N, 4]
├── validity.npy         # [N, 4]
└── manifest.json
```

Manifest 至少记录：

```text
schema_version
node_id ↔ row_index
video_id
source field contract
encoder model/version per slot
dimension/dtype/normalization per slot
matrix checksum per slot
node lineage/source checksum
pre-read boundary audit version
```

当 node source span、caption/entity fields、visual encoder、merge lineage 或任何 matrix checksum
变化时，旧 `U_i` cache 必须失效。

工程 smoke 可以允许某些 slot 缺失并通过 validity mask 屏蔽；正式四槽实验必须报告每个 slot
覆盖率，不能把不完整输入称为完整四槽 baseline。

## 10. 单元与集成测试

在现有 `memory_graph/tests/` 目录增加：

```text
test_qformer_contracts.py
test_qformer_features.py
test_qformer_models.py
test_qformer_losses.py
test_qformer_training_smoke.py
```

必须覆盖：

1. 四槽 shape、dtype、slot name 和 node-row binding 正确；
2. validity mask 正确屏蔽缺失 slot；
3. 全 slot 缺失时 fail closed；
4. feature 输入顺序变化但 type ID 不变时，QF1 输出保持一致；
5. QF1 输出严格为 `[B, 8, 768]`；
6. QF2 输出严格为 `[B, 8, 768]`；
7. QF1/QF2 query embeddings 和 Transformer 参数不共享；
8. Stage 2 冻结 QF1 后没有 QF1 梯度；
9. slot decoder loss 只统计 valid slots；
10. multi-positive loss 不把另一个 positive 标为 negative；
11. ignore nodes 和 padding 不进入 denominator；
12. checkpoint export/import 与 `U_i` cache 可复现；
13. manifest/checksum 或 node lineage 改变后 cache 正确失效；
14. 未执行 READ 时，Q-Former dataset 中不存在 `EvidenceValue` grounded payload。

训练 smoke 使用小型 synthetic cohort：

```text
8 个 videos
每个 video 4–8 个 nodes
每个 question 1–2 个 positives
100–500 个 optimizer steps
```

检查 loss 下降、无 NaN、query/projector 有梯度、冻结 encoder 无梯度、checkpoint resume 一致，
以及单 GPU forward/backward 可完成。

## 11. QF1 validation gates

所有数据按 video-disjoint split 划分。在 held-out videos 上报告：

- 每个 slot 的 cosine reconstruction；
- macro 和 validity-weighted slot reconstruction；
- 缺失单个 slot 时的 reconstruction degradation；
- QF1 cache determinism；
- feature-order permutation consistency；
- mask corruption sensitivity；
- query-token pairwise cosine，仅作为 collapse 诊断；
- 每个 slot 的数据覆盖率。

QF1 进入下一阶段的必要条件：

```text
训练和验证 loss 稳定下降
每个高覆盖 slot 均能在 held-out videos 上恢复
缺失 slot 不产生 NaN 或污染其他样本
shuffle 后对应 node 表示保持一致
cache checksum、node ID 与 lineage 可审计
pre-read boundary audit 通过
```

checkpoint 由 held-out slot preservation 与 boundary gates 选择，不由最终 QA accuracy 选择。

## 12. QF2 validation gates

在固定完整 candidate pool 上报告：

- Recall@1、Recall@4、Recall@8；
- MRR；
- all-required-clue coverage@K；
- single-clue 与 multi-clue cases 分开统计；
- trusted same-video hard-negative accuracy；
- candidate-order permutation consistency；
- caption/entity-state/visual/time 各 slot 的 mask-out ablation；
- proposal-only answer leakage probe；
- 不同 slot-coverage slice 上的性能；
- video-disjoint bootstrap confidence interval。

主 gate 使用与执行预算对齐的：

```text
Recall@4
all-required-clue coverage@4
```

QF2 checkpoint 由 held-out clue recall、完整 clue-set coverage、permutation 和 leakage gates
共同选择，不由最终 QA accuracy 单独选择。

## 13. 必须运行的 matched baselines

因为输入只有 4 个 feature tokens，而 QF1 输出 8 个 tokens，必须证明 QF1 提供了有效融合，
而不只是增加层数和参数。

| Baseline | 输入与结构 | 目的 |
| --- | --- | --- |
| B0 | 当前单一 event embedding 与 question cosine | 现有 embedding baseline |
| B1 | 4 slots 直接 mean/attention pool 后 retrieval | 无 Q-Former baseline |
| B2 | 4 slots → QF2(8) → retrieval | 直接 question-conditioned baseline |
| B3 | 4 slots → frozen QF1(8) → QF2(8) → retrieval | 主方法 baseline |

只有 B3 在 video-disjoint held-out cohort 上稳定优于 B1/B2，才能说明 QF1 的问题无关 feature
fusion 有实际价值。

后续容量 ablation：

```text
QF1/QF2 query 数：4 / 8 / 16
```

第一轮先完成主版本 `8/8`，不同时扩张模型容量和 loss 数量。

## 14. 编码执行顺序

1. 实现四槽 contracts、manifest、loader、validity mask 与 checksum tests；
2. 实现 feature materializer，并生成 slot coverage/leakage report；
3. 补齐独立 visual slot，不以 text event embedding 代替；
4. 实现 QF1、slot decoder、唯一 loss 与 synthetic training smoke；
5. 在真实 video-disjoint 数据上训练 QF1，验证后导出 frozen `U_i` cache；
6. 实现 QF2 dataset，严格区分 positive、ignore 和 trusted negative；
7. 实现 QF2、node scoring、唯一 retrieval loss 与训练入口；
8. 实现 Recall@K、clue-set coverage、permutation、leakage evaluator；
9. 运行 B0/B1/B2/B3 matched comparison；
10. 只有 QF1/QF2 gates 均通过后，才固定 `P_i` interface 并接入 IWM。

## 15. V1 最终决策

```text
输入：4 个 feature tokens，stack，不平均
QF1：8 个 independent queries，4 层，question-independent
QF1 loss：一个 masked 4-slot cosine distillation
QF2：8 个 conditioned queries，2 层，question-conditioned
QF2 loss：一个 trusted-negative multi-positive retrieval
训练：先 QF1，冻结并 cache，再训练 QF2
Video/text encoders：冻结
最终 QA loss：不更新 QF1/QF2
Joint tuning：V1 默认关闭
```

第一阶段的工程停止点是可靠的四槽 feature coverage 和 boundary report；第一阶段的模型停止点
是 B3 相对 B1/B2 在 held-out clue recall/coverage 上获得稳定收益。任一 gate 失败时，保留结果
作为 negative baseline，不通过增加 auxiliary losses、隐藏 Top-K 或最终 QA 梯度掩盖失败。

## 16. 实施状态（2026-08-09）

已完成 contracts、immutable feature store、四槽 materializer、QF1/QF2、raw-slot
distillation、trusted-negative retrieval、QF1 cache、question embedding、label compiler、完整候选
评估和 B0/B1/B2/B3 launcher。当前单元测试为 22 passed；synthetic CPU/GPU smoke 与 40-node
真实 QF1 pilot 已通过。

数据 split 审计补充了一个必须遵守的边界：

- `iwm_runtime_train_collection_v1/l15_fixed_train_v1_vllm`：40 个 train 视频、56 个可对齐问题；
- `cgbench_gt_navigation_pilot_v2/l15_fixed_cohort_v2_vllm_c8`：36 个 held-out 视频，含
  21 个 validation 与 27 个 test 问题，不含 train 视频；
- QF1/QF2 只允许在前者拟合；后者只应用冻结 checkpoint；
- test 问题暂不进入 checkpoint selection，当前 matched comparison 只报告 validation。

GPU 执行链（均已完成，exit code 0）：

```text
7228688: train-only four-slot materialization + QF1 + train cache/questions
7228666: held-out four-slot materialization（其中附带 QF1 仅为 transductive diagnostic）
7228713: train-afterok / heldout-afterany dependency；frozen held-out cache + B0/B1/B2/B3 validation
```

B2 为严格 fusion ablation：复用 train-only QF1 checkpoint 的 frozen four-slot projector，但绕过
QF1 的 8-token fusion，直接令同构 QF2 读取 4 个 typed slots。这样 B2/B3 之间只改变 QF1
fusion，labels、candidate pool、question embedding、QF2、optimizer 与 epochs 保持一致。

### 16.1 完整运行结果

- train：40 videos / 16,246 nodes；caption、visual、time coverage 100%，entity/state
  coverage 96.13%；QF1 video-disjoint validation loss 从 1.0625 降至 0.0596；
- held-out：36 videos / 15,181 nodes；caption、visual、time coverage 100%，entity/state
  coverage 96.79%；
- retrieval supervision：train 56 questions；validation 21 questions / 17 videos；test 保持未用；
- leakage/permutation gates：B2/B3 均通过，permutation max error 为 0；
- 以下指标全部使用 validation 视频内的完整 node candidate pool，而不是 sampled negatives。

| Baseline | MRR | Recall@1 | Recall@4 | Recall@8 | Coverage@8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| B0 caption cosine | 0.2876 | 0.0132 | 0.0734 | 0.0797 | 0.0000 |
| B1 caption/entity/visual mean cosine | **0.4373** | **0.0584** | **0.1023** | **0.1758** | 0.0000 |
| B2 four projected slots → QF2 | 0.1933 | 0.0014 | 0.0065 | 0.0597 | **0.0476** |
| B3 frozen QF1(8) → QF2 | 0.2009 | 0.0032 | 0.0398 | 0.0441 | 0.0000 |

因此 V1 的结论是：

```text
QF1 reconstruction gate：通过
QF2 optimization/boundary/permutation gates：通过
B3 > B2：部分指标混合，不构成稳定提升
B3 > B1：失败
```

目前没有证据表明 QF1 fusion 值得接入 IWM。B1 是当前最强且最简单的检索 baseline；B2/B3
保留为可复现的 negative result。下一步若继续，应优先诊断 sampled cross-video training objective
与 full same-video ranking 的分布错配，并加入可信的 same-video hard negatives；不能通过增加
auxiliary losses 或使用 test split 来掩盖当前结果。
