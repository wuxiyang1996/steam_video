# steam_video

Steam video agents: support-progressive navigation problem formulations.

## Active packages

- `memory_graph/` — temporal / candidate-causal memory graph
- `dynamic_navigation/` — progressive navigation formulations
- `streaming_3bench/` — **5 baselines × 3 benchmarks** streaming eval package

### Streaming eval contract (`streaming_3bench/`)

Benchmarks: OVO-Bench, VideoMME, StreamingBench.

Baselines:

1. Dispider
2. StreamBridge
3. M3-Agent
4. Iterative RAG
5. Qwen3.5-9B + FAISS

See [`streaming_3bench/README.md`](streaming_3bench/README.md).
