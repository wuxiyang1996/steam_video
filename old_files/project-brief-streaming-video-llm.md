# Streaming Video LLM with Infinite Memory

**Source:** Internship kickoff email  
**Recipient:** Xiyang  
**Context:** Summer internship project overview

---

## Overview

Thanks for reaching out and accepting the internship offer.

For the summer, we will work on **"Streaming Video LLM with Infinite Memory."** The goal is to adapt existing LLM architectures into streaming-friendly models that can process video streams **without requiring all past frames in the context window**.

---

## Project scope (two parts)

| Part | Focus |
|------|--------|
| **1. Stream processing** | How can we efficiently process video streams? |
| **2. Memory** | How can we efficiently store and retrieve from memory? |

Part 2 is likely the **least explored** in existing literature. Prior work proposes memory and streaming architecture adaptations, but **none of them can hold infinite video context**.

---

## Starting direction: external memory index

One direction to explore first:

1. **Extract** frame embeddings from the video stream.
2. **Store** them in an external index (e.g. FAISS).
3. **Use the LLM** to navigate and retrieve from this space at query time.

### Example query

> *"What did I do between taking my phone out and putting it back in my pocket?"*

**Expected answer (illustrative):**

> You opened Instagram and scrolled through memes, then you played Candy Crush and took some selfies.

### Why this is hard

- Requires storing **entire days' worth of video** and retrieving relevant information efficiently.
- Standard vision–language embedding models (e.g. CLIP) are **not sufficient** on their own: this task needs **multi-hop reasoning**:
  1. Find the event of taking out the phone.
  2. Find the event of putting the phone back in the pocket.
  3. Walk through the frames in between in a streaming fashion and summarize what happened.

An **agentic approach** may work here as well and is worth considering.

### Benchmarks

Existing benchmarks can be adapted for this setup, e.g. **Ego4D** and datasets used in related papers.

---

## Relevant work

### Streaming video / vision

- [StreamBridge (Apple)](https://github.com/apple/ml-streambridge)
- [Flash-VStream](https://github.com/IVGSZ/Flash-VStream/tree/main)

### Infinite context / memory (NLP)

- [LongMem](https://github.com/Victorwz/LongMem)
- [Memorizing Transformers (arxiv)](https://arxiv.org/pdf/2203.08913)

---

## Next steps

Think through this problem setup and reach out with any questions.
