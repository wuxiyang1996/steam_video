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
external retrieval；每步 planner 仍通过 embedding/structure 选取 bounded top-K，不能把整份
字幕或 graph 塞进 prompt。`closed_loop_protocol.json` 固定比较：

- world-model-guided；
- no-world-model；
- shuffled world-model prediction；
- immediate-effect-only；
- oracle-clue ceiling（仅为诊断上界，不是方法结果）。

`modality_coverage.py` 分开报告 SRT availability、Qwen visual descriptor 和 in-frame readable
text。SRT 只称为 time-aligned text，不能冒充直接 audio observation；该报告只衡量模态是否
可观测，不重写 CG-Bench 标签。

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
