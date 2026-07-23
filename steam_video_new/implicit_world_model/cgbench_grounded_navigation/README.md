# CG-Bench Grounded Navigation

本目录把 CG-Bench 的人工 `clue_intervals` 转换为 ground-truth-anchored reasoning navigation 数据。它是独立的数据构建组件，不依赖 factor graph，也不会训练 GPT-OSS/Qwen。

## 监督边界

CG-Bench ground truth 可以可靠监督：

- question 对应的相关视频时间窗；
- 是否读取了新的人工 clue；
- 多 clue 覆盖是否仍为 partial 或已经 complete；
- 完整 clue path 相对于 leave-one-clue-out path 的 categorical coverage preference；
- terminal answer，单独保存在 hidden key。

它不能自动监督：

- `same_entity`；
- `state_transition`；
- 因果关系；
- 读取全部 clue 后必然 answerable。

因此 belief delta 使用 `advances_required_clue_coverage`、`does_not_advance_required_clue_coverage`、`partial`、`complete` 和 `unknown` 等 categorical descriptor，不把 clue relevance 伪装成更强的关系标签。

## 数据设计

每个 multi-clue QA 构造：

1. 视频级确定性 train/validation/test split；
2. 每个 GT clue 对应一个 `read_video_interval` action；
3. action ID 与 preference 左右顺序经过稳定哈希，不向模型暴露 hidden answer；
4. 一组 action-conditioned categorical clue-acquisition transitions；
5. 一条完整 clue trajectory 与 leave-one-clue-out trajectory 的 ordinal preference；
6. Qwen3-VL embedding placeholder，等待真实读取后写入。

正确答案、answer key 和 clue action identity 保存在 `*.hidden_key.json`，不能进入 planner
input。当前阶段只收集与检查数据，不训练，因此产物保持 `training_ready=false` 和
`training_performed=false`。

CG-Bench clue intervals 是正例定位而不是穷尽标注。区间位于 GT clue 之外，只表示它
没有覆盖该条人工 annotation，不能推出它在语义上无关。因此 v0.2 不再构造或训练
`matched_control`，也不要求重复审核已有人工 clue ground truth。

这里的正式标签只来自 CG-Bench：人工 clue intervals、question/choices 和 terminal answer。
Qwen observation descriptor 是执行 action 后的观测，不冒充人工标签。GPT-5.6 或人工视觉
检查仅用于发现解码、时间戳或描述忠实度问题，是可选诊断，不是训练/评测门禁。

## 构建

```bash
cd /fs/gamma-projects/vlm-robot/steam_video

python -m steam_video_new.implicit_world_model.cgbench_grounded_navigation \
  --input /fs/gamma-projects/vlm-robot/datasets/CG-Bench/cgbench.json \
  --video-root /fs/gamma-projects/vlm-robot/datasets/CG-Bench/cg_videos \
  --dataset-id cgbench-gt-navigation-pilot-v2 \
  --min-clues 2 \
  --max-cases 256 \
  --output steam_video_new/implicit_world_model/datasets/cgbench_gt_navigation_pilot_v2/navigation_dataset.gt_only.json \
  --hidden-output steam_video_new/implicit_world_model/datasets/cgbench_gt_navigation_pilot_v2/terminal_targets.hidden_key.json \
  --report steam_video_new/implicit_world_model/datasets/cgbench_gt_navigation_pilot_v2/build_report.json
```

移除 `--max-cases` 可扫描并构建全部合格 multi-clue cases。

## 数据检查门禁

1. 用 Qwen-VL 实际读取每个 action interval，生成 observation descriptor；
2. 用 `Qwen/Qwen3-VL-Embedding-2B` 写入 embedding sidecar；
3. 验证所有 executed transitions 都对应人工 GT clue，且没有 outside-clue negative；
4. 验证 answer/clue hidden key 不进入 planner input；
5. 分别报告 train/validation/test，validation/test 永不训练。

抽样 GPT-5.6/人工 visual QA 可以运行，但结果不得改变 CG-Bench 标签，也不得阻塞正式实验。

完整 grounding 后运行 `validate_artifact.py`，统一检查 grounded/failed counts、embedding
shape/checksum、hidden-key checksum、split leakage 和 data-only 状态。该检查不读取人工审核
结果，也不会启动训练。

## 与真实 L1.5 navigation 的边界

GT-only 数据用于学习/检查“读取某个 reasoning hop 后 belief 如何变化”，不是完整的候选检索
benchmark。若训练时只给 GT clue actions，下一跳筛选会过于简单。因此真实 closed-loop planner
仍从 L1.5 evidence memory 取得候选 hops；CG-Bench clue mapping 只判断命中的 hop 是否覆盖新
GT evidence，terminal answer 判断整条轨迹是否成功。未命中的 L1.5 candidates 保持 unlabeled，
不自动变成 negatives。这样不需要额外人工标签，也避免通过手工 heuristic 教 planner。

`l15_candidates.py` 实现这一边界：优先读取 persisted L1/L1.5 graph nodes；没有 graph 时，
只允许使用数据集已有的 time-aligned SRT cues 作为 L1 fallback。它不会用固定窗口或 GT clue
intervals 生成公开候选。GT temporal overlap 只写入 hidden evaluator key；无 overlap 的 hop
保持 `unlabeled`，从不变成 semantic negative。

候选按视频规范化保存，多个 QA case 共享一个 candidate set。完整 candidate artifact 仅供
构建冻结图。正式路径先对全部安全 semantic addresses 做一次 categorical entry
localization；它不读取未执行 evidence value，不输出数值分数，也不做 Top-K。之后每步 planner
只接收当前 cursor 的 temporal/L1.5 local closure，不能把整份字幕或 graph 反复塞进 prompt。
`closed_loop_protocol.json` 固定比较：

- world-model-guided；
- no-world-model；
- shuffled world-model prediction；
- immediate-effect-only；
- oracle-clue ceiling（仅为诊断上界，不是方法结果）。

`modality_coverage.py` 分开报告 SRT availability、Qwen visual descriptor 和 in-frame readable
text。SRT 只称为 time-aligned text，不能冒充直接 audio observation；该报告只衡量模态是否
可观测，不重写 CG-Bench 标签。

## Question-independent L1/L1.5 graph smoke

`l15_graph_worker.py` 消费不含 question/GT 的 graph-generation manifest。smoke selection 仅按
video-disjoint split、字幕 fallback availability 和视频时长分层，固定选择 12 个视频；graph
worker 只能读取 raw video frames。为了控制工程 smoke 成本，每个视频仅扫描前 120 秒，产物
明确标记 `smoke_prefix_only=true`，不能冒充完整视频 graph。

Qwen visual-L1 prompt 只输出 categorical `observed|inconclusive`，不输出 confidence、reward、
score、probability 或 utility。旧图 schema 中的 numeric grounding field 只是确定性兼容标记，
不进入 IWM/planner 或训练监督。

graph 写盘冻结并计算 checksum 后，独立 evaluator 才能读取 question 与 hidden clue intervals。
它分别报告 native candidate recall 和 embedding top-K recall；超出 120 秒 smoke horizon 的 clues
从 recall 分母排除。无 temporal overlap 的 nodes 仍是 unlabeled，不是 semantic negatives。

```bash
python -m steam_video_new.implicit_world_model.cgbench_grounded_navigation.l15_graph_worker select \
  --manifest /path/to/l15_graph_generation_manifest.json \
  --dataset-root /fs/gamma-projects/vlm-robot/datasets/CG-Bench \
  --output /path/to/l15_graph_smoke_selection.json

DEPENDENCY_JOB_ID=<grounding-validation-job> \
  steam_video_new/implicit_world_model/cgbench_grounded_navigation/submit_l15_graph_smoke.sh
```

## Fixed 49-case held-out L1/L1.5 cohort

正式 IWM 实验前先运行 `fixed_l15_cohort.py`。默认 cohort 是 CG-Bench 中完整的
validation+test population：49 cases、36 个 video-disjoint videos。它不是按 question、answer
或 clue 挑选的；graph worker 收到的 selection 只包含 video path、split 和完整视频时长。

流程严格分为两侧：

1. graph side 只读取 raw video，冻结完整 L1、embedding sidecar、temporal backbone 和
   question-independent L1.5 correlation；
2. evaluator side 在 checksum 冻结后才读取 hidden clue intervals，并把失败分成
   `raw_l1_missing_clue`、`bounded_consolidation_dropped_clue` 和 `l15_path_missing`。

只有所有 clue 都有 retained L1 overlap、且相邻 clue 在冻结图中连通的 case 才进入 locked
case set。目标是锁定 30–50 cases；不足 30 条时 gate fail-closed，不会启动 IWM 调用。

`structural_delayed_candidate=true` 只表示两个相邻 clue sets 的最短路径至少为两跳。它不能
证明“第一步无收益、第二步成功”；真正的 delayed success 必须在 matched-budget executed
rollout 中观察到，不能从 GT interval 或图距离直接制造标签。

当前长视频诊断显示 `capacity=64` 会因 consolidation 丢 clue；两条完整视频 smoke 在
`capacity=192` 时才都恢复完整 temporal retention。但 49-case 正式 cohort 在 192 下仅锁定
29 条，低于 30 条门槛。冻结 L1 上的 192/256/384/512 sweep 不重新调用 VLM，也不覆盖原图；
结果分别锁定 29/34/39/46 条，平均 retained nodes 为 186.0/234.9/310.8/353.3，平均
navigation edges 为 564.5/724.4/969.6/1102.7。正式协议选择**最低合格容量 256**，而不是
为了追求更多 case 使用 384/512。三条 `raw_l1_missing_clue` case 与另外 12 条 capacity-256
consolidation failures 被明确排除，不进入 IWM 指标。

`fixed_l15_cohort.py sweep` 从冻结 overlay 内存重编译每档容量，输出 clue-retention、delayed
candidate 数和 storage/read-cost proxy；`promote-sweep` 把选中档提升为正式 gate。
`full_graph_iwm.cgbench_pilot --fixed-cohort-gate ...` 只接受该 gate 的 locked case IDs，并校验
容量一致。graph storage capacity 与 planner 每步 action interface 始终分开；禁止使用
question-aware Top-K 掩盖 memory loss。

2026-07-22 的 capacity-256 delayed-case 局部导航诊断中，entry localizer 从 256 个地址返回
3 个 anchors，初始 legal actions 从旧路径的 258 降为 5，真实读取后的 local actions 为 9。
localizer 覆盖 1/2 hidden clues，但 IWM 把 105 秒的普通枪击错误预测为可解决
`agent/action/victim`，没有选择已进入 anchors 的 1263 秒导弹 clue；真实读取结果为
inconclusive。该结果把问题定位为 action-conditioned transition calibration，而非继续压缩
图或增加 heuristic Top-K。正式数据应记录 local anchors 上 predicted/realized belief delta
的差异，并保持 evaluator-only clue 命中不反馈给 planner。

`full_graph_iwm/local_choice_data.py` 已实现该记录协议：保留全部 entry-anchor 首跳预测，生成
完整 categorical pair comparisons，并把 GT-derived labels 隔离在 hidden key。一个 anchor
命中 clue、另一个没有命中时，GT 只监督本数据集 navigation preference；未命中端仍标记
`not_established`，不会成为 identity、causal 或一般 semantic negative。executed correction
同样明确限定为 clue-coverage navigation delta，不冒充完整 observation truth。

当前 packet 来自 test split，只用于诊断：3 个 anchors 形成 3 个 pairs，其中 2 个 strict
preferences、1 个 incomparable，并包含 1 个 predicted-versus-grounded navigation mismatch。
其 `training_ready=false`、`training_performed=false`。正式 post-training 前必须从冻结的
train videos 生成独立 packet，并保持 validation/test video-disjoint。

### 34-case zero-shot IWM + Planner 验证

冻结 capacity-256 cohort 上已完成 34 cases / 27 videos / 170 case-arm slots 的 zero-shot
matched evaluation；没有训练。统一使用两次真实读取、horizon two、相同 graph fingerprint 和
`qwen/qwen3.6-flash` proxy：

| Arm | Mean clue recall | Complete coverage | Delayed success |
|---|---:|---:|---:|
| intact IWM | 0.201 | 0.118 | 0.088 |
| no-WM | 0.000 | 0.000 | 0.000 |
| shuffled-IWM | 0.098 | 0.059 | 0.059 |
| immediate-only | 0.137 | 0.088 | 0.059 |
| oracle | 0.730 | 0.471 | 0.324 |

paired clue-recall delta 为：相对 no-WM +0.201、相对 shuffled +0.103、相对 immediate-only
+0.064。intact 对 immediate-only 只有 3 cases 更好、0 更差、31 相同；因此已有 provisional
navigation effect，但尚不能宣称强 delayed advantage。

失败切片同样保留在正式分母中：2/34 cases 的 entry localization 违反 categorical contract，
2 个单独 arms 的 setwise schema 无效；它们全部 fail-closed abstain，不做 Top-K、alias
guessing 或规则修复。localizer 至少包含一个 clue 的比例为 55.9%；在这些 cases 中 intact
first read 命中 clue 的比例为 47.4%。intact executed transitions 的 exact outcome match 为
1/31，exact belief-delta match 为 0/31，说明 transition calibration 仍是主要缺口。

validation/test 的 intact clue recall 分别为 0.200/0.202。完整结果明确保持
`training_performed=false`、hidden clue feedback=false、numeric reward=false、
Top-K=false。

完整 cohort 使用 `grounded_single_pass` L1：每个 surprise window 只调用一次 VLM，同时返回
event、sampled-frame endpoints/evidence、participants 和 visible states；任何没有合法 frame
provenance 的 event 都被拒绝。旧 `coarse_to_fine` 两遍模式保留为消融。正式默认采用 balanced
窗口：0.75 秒 representation sampling、4–20 秒自适应窗口和 0.85 surprise quantile；稳定内容
扩窗，变化内容缩窗，所有窗口仍覆盖完整时间轴。

真实 120 秒 benchmark 中，旧流程需要 30 次 coarse scan，并产生 85 个需要第二次调用的
fine candidates；优化版仅做 9 次调用，在 6 分 25 秒内生成 25 个 frame-grounded L1 nodes，
调用量约减少 92%。但该 aggressive 8–32 秒设置在第一条完整视频上遗漏了一个 GT clue，
因此只保留为速度上界，不用于正式 cohort。Balanced 配置在同一 623 秒视频产生 86 个窗口，
仍远少于旧流程约 185 coarse + 391 fine calls；扩大运行前必须重新通过完整视频 clue-retention
gate，不能只凭速度替换。

### Qwen3.5-9B serving backend

`run_l15_graph_smoke_job.sh` 现在显式支持 `SERVER_BACKEND=transformers|vllm`。
两者使用相同的 OpenAI-compatible multimodal request、prompt、sampling temperature、
窗口和 parser；backend 名称写入 L1 artifact，并参与 resume contract，禁止把两个 backend
的节点混入同一 graph。当前 balanced full-video validation 仍是 Transformers 基线；切换
vLLM 前必须在同一完整视频上比较 wall time、accepted/rejected L1 nodes、三条 clue retention
和 L1.5 path gate。

仓库原有 `vllm 0.8.5.post1` 早于 Qwen3.5 支持，不能用于该比较。新环境保持隔离：

```bash
cd /fs/gamma-projects/vlm-robot/steam_video
steam_video_new/implicit_world_model/cgbench_grounded_navigation/setup_qwen35_vllm_env.sh

SERVER_BACKEND=vllm \
GRAPH_ROOT="$PWD/steam_video_new/implicit_world_model/datasets/cgbench_gt_navigation_pilot_v2/l15_vllm_validation_v1" \
RUN_STAGE=all VIDEO_LIMIT=1 MEMORY_CAPACITY=192 \
  steam_video_new/implicit_world_model/cgbench_grounded_navigation/run_l15_graph_smoke_job.sh
```

vLLM 在这里是吞吐/serving 优化，不改变 L1/L1.5 定义，也不是新增 supervision。只有上述
matched-video gate 不退化，正式 36-video cohort 才设置 `SERVER_BACKEND=vllm`。
集群节点不提供系统级 `nvcc`，因此启动脚本默认设置
`VLLM_USE_FLASHINFER_SAMPLER=0`，只让 sampling 回退到 vLLM native implementation；
FlashAttention、Qwen GDN kernel 和 multimodal serving 仍保持启用。
vLLM 还必须使用 `--default-chat-template-kwargs '{"enable_thinking":false}'`，与
Transformers baseline 的 `--reasoning off` 对齐。未设置时 Qwen3.5 会把 token budget 用于
thinking，导致 HTTP 200 但没有可解析的 `events`；这种运行必须作为协议失败，而非有效加速。

`L1_REQUEST_CONCURRENCY` 控制同一视频内独立 surprise windows 的并发请求数。每个 worker
使用独立 client，HTTP metadata 不共享；结果按原始 window index 回收，因此 node/rejection
顺序保持确定。并发只改变 serving schedule，不改变 windows、sampled frames、prompt 或 parser。
正式脚本现已预设为 8；禁止通过减少
frames、扩大窗口或 heuristic Top-K 冒充 serving 加速。

2026-07-22 的 matched full-video 结果已将正式配置冻结为 vLLM concurrency=8、thinking
disabled：相同 623 秒视频和 88 个 windows，Transformers 用时 46:22，生成 170 nodes / 23
rejections；vLLM c8 用时 8:20，生成 173 nodes / 19 rejections。两者都保留 3/3 GT clues，
并覆盖 2/2 L1.5 clue bridges（1 direct、1 multi-hop）。约 5.6× wall-time improvement 来自
serving/batching，不来自窗口、帧、prompt、parser 或 gate 放宽。早期未关闭 thinking 的 c8
虽然 88 次 HTTP 均为 200，却产生 0 nodes；该 artifact 明确视为 protocol failure。

生成冻结 selection/protocol，并以一视频一 checkpoint、最多 16 个并发 GPU 的方式提交：

```bash
cd /fs/gamma-projects/vlm-robot/steam_video
NUM_SHARDS=36 MAX_PARALLEL=16 GPU_TYPE=rtxa6000 MEMORY_CAPACITY=192 \
QOS=gamma-huge-long CPUS_PER_TASK=8 MEMORY_PER_TASK=64G \
SERVER_BACKEND=vllm L1_REQUEST_CONCURRENCY=8 \
  steam_video_new/implicit_world_model/cgbench_grounded_navigation/submit_l15_fixed_cohort.sh
```

最终产物位于 `l15_fixed_cohort_v1/`：

- `build_report.json`：36 个 graph 的完整性与 shard 合并报告；
- `coverage_report.json`：冻结后 embedding/native coverage；
- `l15_correlation_evaluation.json`：相邻 clue bridge 的 evaluator-only 汇总；
- `fixed_cohort_gate.json`：不含 clue intervals 的 locked case IDs 与失败统计；
- `fixed_cohort_gate.hidden_key.json`：clue-to-node 对齐细节，仅供 evaluator。

## 已实现的 grounded-read 闭环

`grounding.py` 将每个 opaque `read_video_interval` 送入本地
`Qwen/Qwen3.5-9B`。模型只能看到 question、choices、采样帧和帧时间，不会看到
terminal answer 或 clue target。输出只允许：

- 直接可见的 entity、action、state 与画面内可读文字；
- `relevant | unrelated | inconclusive` 三分类 relevance；
- 可追溯到采样帧的 evidence indices；
- 不允许数字 confidence、utility 或 reward。

当前读取协议只观测视频帧及画面内文字，不声称读取了音频。`audio` 会明确记录在
`unobserved_requested_modalities`；依赖纯语音的 case 必须在 audit 中标记
`inconclusive`，后续接入独立 ASR 后才可升级。

每处理完一个 case 都会原子式重写 checkpoint artifact。真实 observation 写入后，
`Qwen/Qwen3-VL-Embedding-2B` 只编码 grounded descriptor；2048 维归一化向量保存为
`.npy` sidecar，JSON 只保存 row、dimension、model 和 checksum reference。embedding
仅供未来 retrieval/navigation，不是 reward 或 preference supervision。

GPU smoke：

```bash
cd /fs/gamma-projects/vlm-robot/steam_video
steam_video_new/implicit_world_model/cgbench_grounded_navigation/submit_qwen_grounding_smoke.sh
```

smoke 默认只处理 2 cases。通过后，以相同 worker 设置空 `CASE_LIMIT` 才运行完整
256-case / 672-transition grounding；不能将部分 smoke artifact 标成完整数据。

## GT-only 无训练消融

`evaluation.py` 当前默认生成：

- `all_clues`：完整 GT evidence；
- `leave_one_clue_out`：移除一个 GT clue；
- `shuffled_clue_order`：保留完整 evidence，只改变读取顺序；
- `transition_prediction_shuffle`：执行相同 GT reads，但打乱 imagined transition 对应关系；
- `cross_video_distractor`：读取其他视频的结构性干预，仅用于测试，绝不是 semantic-negative
  训练标签。

hidden key 保存 terminal answer 与被移除 clue，仅供离线评分。可选 blinded packet 只做媒体/
descriptor 质量诊断，带有 `formal_gate=false`。

三类指标必须分开报告：multiple-choice answer accuracy、read efficiency、以及 order
sensitivity/action divergence。GT clue coverage 只能作为 intervention sanity check，不能冒充
模型 answer accuracy。当前 artifacts 是 `unreviewed` / `unexecuted`，没有训练，也没有性能结论。

`audit_assets.py` 为可选诊断 packet 的每个 interval 生成 6 帧、带时间戳的 contact sheet；不会
读取 hidden key。`gpt56_audit.py` 使用 `openai/gpt-5.6-sol` 做 provisional visual review，
输出仅包含 `accept|reject|inconclusive`、`relevant|unrelated|inconclusive`、evidence kind 和
文字 rationale。模型不输出数字 confidence/reward。所有决策写完后才与 hidden role join，
并报告 media/descriptor diagnostics；该结果始终为 `model_provisional`，不覆盖 CG-Bench
ground truth，也不是人工审核门禁。

32-video audit subset：

```bash
steam_video_new/implicit_world_model/cgbench_grounded_navigation/submit_qwen_audit_subset.sh
```

可以提交严格 `afterok` 后处理，使 Qwen grounding/embedding 完成后才启动 GPT-5.6，并在
审核结束后生成 provisional clean-case ablation subset：

```bash
QWEN_JOB_ID=<grounding-job-id> \
  steam_video_new/implicit_world_model/cgbench_grounded_navigation/submit_gpt56_after_qwen.sh
```

以下内容仅是 v0.1 control 设计的历史诊断，不属于 v0.2 正式门禁：当前 32-video pilot 曾
完成 196/196 Qwen reads 和 196×2048 embeddings。GPT-5.6 盲审
发现：98 个 control candidates 中 39 个仍相关、3 个 inconclusive；98 个 clues 中 8 个
unrelated、1 个 inconclusive；descriptor grounding 为 172 accept、22 reject、2
inconclusive。严格 filter 最终仅保留 5/39 cases，而且没有 test case。这是 control
construction 失败的证据，不是 IWM/planner 性能结论。v0.2 已移除这类 controls，不再重采样、
过滤或等待人工锁定。

正式 GPT-5.6 audit 只能使用完成 Qwen grounding 后重新生成的 public packet：

```bash
python -m steam_video_new.implicit_world_model.cgbench_grounded_navigation.gpt56_audit \
  --packet /path/to/blinded_visual_audit.with_assets.qwen_grounded.json \
  --hidden-key /path/to/blinded_visual_audit.hidden_key.json \
  --keys /fs/gamma-projects/vlm-robot/keys.py \
  --output /path/to/visual_audit.gpt56.provisional.json \
  --inspection /path/to/visual_audit.gpt56.inspection.json
```
