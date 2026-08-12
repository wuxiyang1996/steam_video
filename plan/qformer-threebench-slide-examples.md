# Q-Former 三 Benchmark：30 秒 Slide Examples

三个样例都来自冻结 full evaluation，且满足 `Uniform 正确 / Visual 错误 / Q-Former 错误`。
已生成无音频、8 FPS、640×360 的 30 秒 MP4 和 poster：

`/fs/gamma-projects/vlm-robot/steam_video_runs/qformer_threebench_v01/slide_examples/`

## Slide 1 — OVO-Bench：语义检索漏掉关键时间段

- Example：`ovo_bench:1014`
- 30 秒窗口：原视频 `69.37–99.37s`
- Video：`ovo_1014_30s.mp4`
- Poster：`ovo_1014_poster.jpg`
- Question：**What is the material of the object being cleaned?**
- Options：A Plastic / **B Metal** / C Wood / D Glass
- Gold：**B. Metal**
- Predictions：Uniform **B ✓**；Visual **A ✗**；Q-Former **A ✗**

讲解：Uniform 的 `60–90s` clip 覆盖正在清洗金属锅的画面；Visual/QF 用 `30–60s`
替换了它。QF 的 top-3 与 visual 相同，因此继承了错误。

## Slide 2 — VideoMME：相同 clips，不同顺序导致答案翻转

- Example：`videomme:139-1`
- 30 秒窗口：原视频 `0–30s`
- Video：`videomme_139_1_30s.mp4`
- Poster：`videomme_139_1_poster.jpg`
- Question：**Which of the following is the main color of the racetrack?**
- Options：A Purple / B White / **C Pink** / D Blue
- Gold：**C. Pink**
- Predictions：Uniform **C ✓**；Visual **A ✗**；Q-Former **A ✗**

讲解：三种方法看到同一组 `{0–30s, 30–60s}` clips。Uniform 按时间顺序输入；Visual/QF
按 similarity 倒序输入，Qwen 把粉色判断成紫色。这是 evidence ordering sensitivity，而不是
top-k recall 差异。

## Slide 3 — StreamingBench：排序覆盖了正确事件，但推理仍偏移

- Example：`Anomaly Context Understanding_sample_40_1`
- 30 秒窗口：原视频 `18–48s`
- Video：`streaming_sample_40_1_30s_v2.mp4`
- Poster：`streaming_sample_40_1_poster.jpg`
- Question：**What unusual event just occurred?**
- Gold：**C. An orange player stole the ball from purple No. 95 and scored.**
- Predictions：Uniform **C ✓**；Visual **D ✗**；Q-Former **D ✗**

讲解：两组 clips 都覆盖事件；Visual/QF 将 `30–48s` 放在 `0–30s` 前，最终错误归因成
purple No. 95 自己把球扔进门。再次说明当前 QF 主要复现 visual 的顺序，而没有改善证据组织。

## 推荐版式

每页左侧放 30 秒 autoplay/controls 视频或 poster，右侧放 question/options；底部一行放：

```text
Uniform ✓     Visual ✗     Q-Former ✗
```

三页的 takeaway 分别用：`missed interval`、`order sensitivity`、`wrong event attribution`。
