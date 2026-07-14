"""Multi-hop QA loaders + candidate construction for m_k scoring."""

from __future__ import annotations

import re
import random
from dataclasses import dataclass
from typing import Any

from datasets import load_dataset


@dataclass
class MHExample:
    id: str
    question: str
    context: str
    answer: str
    qtype: str
    candidates: list[str]  # [gold, ...] distractors
    action_sketch: list[str]


BRIDGE_SKETCH = ["GROUND", "TRACE", "COMPOSE", "STOP"]
COMPARE_SKETCH = ["GROUND", "CHECK", "COMPOSE", "STOP"]
DEFAULT_SKETCH = ["GROUND", "RESOLVE", "TRACE", "COMPOSE", "STOP"]


def _flatten_context(context: dict[str, Any] | list) -> str:
    # Hotpot style: {"title": [...], "sentences": [[...], ...]}
    if isinstance(context, dict) and "sentences" in context:
        parts = []
        titles = context.get("title") or []
        sents = context.get("sentences") or []
        for i, sent_list in enumerate(sents):
            title = titles[i] if i < len(titles) else f"doc{i}"
            body = " ".join(sent_list)
            parts.append(f"{title}. {body}")
        return "\n".join(parts)
    if isinstance(context, list):
        # list of [title, [sents]]
        parts = []
        for item in context:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                title, sents = item[0], item[1]
                parts.append(f"{title}. {' '.join(sents)}")
        return "\n".join(parts)
    return str(context)


_ENTITY_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b")


def _distract_answers(context: str, gold: str, k: int = 3) -> list[str]:
    gold_l = gold.strip().lower()
    cands: list[str] = []
    for m in _ENTITY_RE.finditer(context):
        ent = m.group(1).strip()
        if ent.lower() == gold_l:
            continue
        if ent.lower() in {c.lower() for c in cands}:
            continue
        if len(ent) < 2:
            continue
        cands.append(ent)
        if len(cands) >= k:
            break
    # pad with generic distractors if needed
    fallback = [
        "New York",
        "London",
        "Paris",
        "Berlin",
        "Tokyo",
        "unknown",
        "none",
        "not stated",
    ]
    for f in fallback:
        if len(cands) >= k:
            break
        if f.lower() != gold_l and f.lower() not in {c.lower() for c in cands}:
            cands.append(f)
    while len(cands) < k:
        cands.append(f"decoy-{len(cands)}")
    return cands[:k]


def _sketch_for_type(qtype: str) -> list[str]:
    t = (qtype or "").lower()
    if "compar" in t:
        return list(COMPARE_SKETCH)
    if "bridge" in t:
        return list(BRIDGE_SKETCH)
    return list(DEFAULT_SKETCH)


def load_hotpot_subset(
    split: str = "validation",
    n: int = 256,
    seed: int = 0,
    n_distractors: int = 7,
) -> list[MHExample]:
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)
    ds = ds.shuffle(seed=seed)
    if n is not None and n > 0:
        ds = ds.select(range(min(n, len(ds))))

    out: list[MHExample] = []
    for row in ds:
        ctx = _flatten_context(row["context"])
        ans = row["answer"].strip()
        distractors = _distract_answers(ctx, ans, k=n_distractors)
        qtype = row.get("type") or "bridge"
        out.append(
            MHExample(
                id=str(row.get("id") or row.get("_id") or len(out)),
                question=row["question"].strip(),
                context=ctx,
                answer=ans,
                qtype=str(qtype),
                candidates=[ans] + distractors,
                action_sketch=_sketch_for_type(str(qtype)),
            )
        )
    return out


def _flatten_musique_paragraphs(
    paragraphs: list[dict[str, Any]], words_per_paragraph: int = 80
) -> str:
    """Give every paragraph a query-independent share of the token budget."""
    parts = []
    for paragraph in paragraphs:
        title = str(paragraph.get("title") or "untitled")
        words = str(paragraph.get("paragraph_text") or "").split()
        parts.append(f"{title}. {' '.join(words[:words_per_paragraph])}")
    return "\n".join(parts)


def _answer_kind(answer: str) -> str:
    normalized = answer.strip().lower()
    if normalized in {"yes", "no"}:
        return "boolean"
    if any(character.isdigit() for character in normalized):
        return "numeric"
    return "text"


def load_musique_subset(
    split: str = "validation",
    n: int = 256,
    seed: int = 0,
    n_distractors: int = 7,
    min_hops: int = 3,
) -> list[MHExample]:
    """Load answerable MuSiQue examples with hard, shuffled answer candidates."""
    dataset = load_dataset("dgslibisey/MuSiQue", split=split)
    rows = [
        row
        for row in dataset
        if row.get("answerable", True)
        and len(row.get("question_decomposition") or []) >= min_hops
    ]
    rng = random.Random(seed)
    rng.shuffle(rows)
    if n is not None and n > 0:
        rows = rows[:n]

    answer_pools: dict[str, list[str]] = {"boolean": [], "numeric": [], "text": []}
    for row in rows:
        answer = str(row["answer"]).strip()
        answer_pools[_answer_kind(answer)].append(answer)

    out: list[MHExample] = []
    for row in rows:
        answer = str(row["answer"]).strip()
        context = _flatten_musique_paragraphs(row["paragraphs"])
        candidates: list[str] = []

        # Intermediate-hop answers are especially hard negatives because they
        # are supported by the same context but do not answer the composition.
        for step in row.get("question_decomposition") or []:
            candidate = str(step.get("answer") or "").strip()
            if candidate and candidate.lower() != answer.lower():
                candidates.append(candidate)

        candidates.extend(_distract_answers(context, answer, k=n_distractors))
        pool = list(answer_pools[_answer_kind(answer)])
        rng.shuffle(pool)
        candidates.extend(pool)

        unique: list[str] = []
        seen = {answer.lower()}
        for candidate in candidates:
            key = candidate.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            unique.append(candidate.strip())
            if len(unique) == n_distractors:
                break
        while len(unique) < n_distractors:
            unique.append(f"decoy-{len(unique)}")

        hops = len(row.get("question_decomposition") or [])
        out.append(
            MHExample(
                id=str(row["id"]),
                question=str(row["question"]).strip(),
                context=context,
                answer=answer,
                qtype=f"{hops}hop",
                candidates=[answer] + unique,
                action_sketch=list(DEFAULT_SKETCH),
            )
        )
    return out
