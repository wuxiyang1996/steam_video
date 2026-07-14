# Streaming Video Agent and Infinite Memory Paper Notes

**Context:** Notes for the "Streaming Video LLM with Infinite Memory" internship direction.

This file collects the papers and systems covered so far, with a short working abstraction for each. The recurring theme is how to combine streaming perception, long-horizon memory, retrieval, and agentic reasoning over video.

---

## 1. RIVER: Real-Time Video Interaction Benchmark

**Link:** https://github.com/OpenGVLab/RIVER

**Working abstraction:** RIVER is a benchmark for real-time video interaction. Instead of evaluating offline video question answering after the full video is available, it asks models to answer while a video stream is unfolding. Its tasks are organized around three abilities: remembering past events, perceiving the current moment, and anticipating or responding to future events.

**Core capabilities:**

- **Retrospective Memory:** Answer questions about things that happened earlier in the stream, such as where an object was placed.
- **Live Perception:** Describe or reason about what is happening now with low latency.
- **Proactive Anticipation:** Monitor for future events and respond when they occur.

**Why it matters:** RIVER is very close to the target problem of a streaming video agent with infinite memory. It gives a useful benchmark structure for separating memory, current perception, and anticipation.

**Limitation for this project:** RIVER is still mainly a video QA benchmark. It does not fully evaluate agentic self-question discovery, memory consolidation, active exploration, tool use, or long-horizon problem solving.

---

## 2. Memorizing Transformers

**Link:** https://arxiv.org/pdf/2203.08913

**Working abstraction:** Memorizing Transformers extend a Transformer with a non-differentiable external memory. The model stores past hidden states or key-value pairs and retrieves relevant entries with approximate kNN attention. This allows the model to use contexts far beyond the standard attention window.

**Core idea:** Keep a large external key-value memory and let the Transformer retrieve from it at inference or training time. The paper shows useful scaling to long memory sizes, including 131K and 262K tokens, across language modeling and code tasks.

**Why it matters:** This is a strong baseline for explicit memory. For the internship project, it suggests a simple starting point: encode video frames, clips, or captions into a vector store such as FAISS, then retrieve relevant memories at query time.

**Limitation for this project:** The memory is primarily token or hidden-state based, not naturally structured for video, spatial scenes, event boundaries, or multi-hop temporal reasoning. It gives retrieval capacity but not full memory management.

---

## 3. Deep Video Discovery: Agentic Search with Tool Use for Long-Form Video Understanding

**Link:** https://proceedings.neurips.cc/paper_files/paper/2025/file/8190b210e9808e54ee16263b673a847d-Paper-Conference.pdf

**Working abstraction:** Deep Video Discovery treats long-video understanding as agentic search. It builds a multi-granularity database over the video, then lets an LLM agent use tools to browse summaries, search clips, and inspect frames for evidence.

**Core tools:**

- **Global Browse:** Read global video or entity-level summaries.
- **Clip Search:** Retrieve relevant clips by semantic query.
- **Frame Inspect:** Return to specific frames for visual evidence.

**Why it matters:** DVD is a strong baseline for memory navigation. It directly matches the loop of query, plan, search memory, inspect evidence, and answer.

**Limitation for this project:** DVD is mainly offline long-video QA. The video is already available and indexed. It does not solve real-time streaming updates, learned memory consolidation, forgetting, or proactive interaction.

---

## 4. Mirage: Latent Spatial Memory for Video World Models

**Link:** https://microsoft.github.io/LatentSpatialMemory/

**Working abstraction:** Mirage stores static 3D scene content as latent tokens inside a video world model. During generation, the model can initialize, read, denoise, and update this latent spatial memory instead of repeatedly rendering and re-encoding an RGB point-cloud memory.

**Core idea:** Maintain a persistent latent spatial cache that represents the scene. The model reads from and updates this cache as the video evolves.

**Why it matters:** Mirage suggests that infinite memory for video does not have to be only captions plus vector search. A stronger system could maintain a latent spatial memory field that preserves scene structure and can be queried by an agent.

**Limitation for this project:** Mirage is closer to video world modeling and spatial scene persistence than interactive agent memory. It is useful as an architectural inspiration, but it is not itself a complete streaming video agent benchmark or retrieval system.

---

## 5. ReMEmbR: Robotic Episodic Memory

**Link:** To be filled in.

**Working abstraction:** ReMEmbR is a robotic memory system focused on remembering object, place, and event information over time. It is relevant as a practical baseline for embodied, spatial, and episodic memory.

**Core idea:** Store robot observations in a structured memory that can support later retrieval about where things were seen, what happened, and how the environment changed.

**Why it matters:** The target video-memory agent has a similar need: it must remember temporally grounded events and spatial facts from a stream, then retrieve them when queried.

**Limitation for this project:** Robotic memory systems are usually environment- and embodiment-specific. They may not directly handle open-ended long-form video, rich language queries, or internet-scale visual diversity.

---

## 6. Diffusion as Associative Memory

**Link:** To be filled in.

**Working abstraction:** Diffusion-as-memory work frames generative diffusion models as implicit associative memories. Instead of storing memories only as explicit key-value entries, the model can recover or complete patterns through a learned generative process.

**Core idea:** Memory can be represented implicitly in a generative latent space, where queries act like partial observations and generation retrieves or reconstructs associated content.

**Why it matters:** This gives an alternative to FAISS-style explicit retrieval. For the infinite memory project, it motivates a possible flow-matching or diffusion memory field that can retrieve, interpolate, or reconstruct memories.

**Limitation for this project:** Implicit generative memory is harder to inspect, update, and evaluate than explicit retrieval. It may also hallucinate, so a practical agent may still need evidence-grounded retrieval.

---

## Project-Level Takeaways

The covered work suggests three useful baseline families:

1. **Benchmark baseline:** RIVER for streaming memory, perception, and anticipation.
2. **Explicit retrieval baseline:** Memorizing Transformers and DVD-style vector/tool memory.
3. **Latent memory baseline:** Mirage and diffusion-style associative memory.

A possible research framing is to extend RIVER-like real-time interaction with DVD-like agentic search and Mirage-like latent memory. The gap is an agent that can decide what to remember, consolidate memory over time, retrieve evidence across long horizons, and proactively act on future events.
