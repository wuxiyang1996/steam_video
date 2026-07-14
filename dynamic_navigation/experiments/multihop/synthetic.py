"""Controllable synthetic multi-hop stories that require compositional reads."""

from __future__ import annotations

import random
from dataclasses import dataclass

from data import MHExample

PEOPLE = [
    "Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace", "Helen",
    "Ivan", "Julia", "Kevin", "Laura",
]
OBJECTS = [
    "red bag", "blue box", "green cup", "black book", "silver key",
    "yellow hat", "white letter", "brown notebook", "purple scarf", "orange ball",
]
PLACES = [
    "the kitchen", "the desk", "the window", "the shelf", "the garden",
    "the hallway", "the car", "the office", "the balcony", "the closet",
]


@dataclass
class SynthConfig:
    n: int = 2000
    seed: int = 0
    n_distractors: int = 7  # total candidates = 1 gold + n_distractors
    hops: tuple[int, ...] = (2, 3)


def _pick(rng: random.Random, items: list[str], k: int) -> list[str]:
    return rng.sample(items, k)


def _make_story(rng: random.Random, hops: int) -> MHExample:
    """
    Build a possession/location chain that requires multi-hop composition.
    Example 3-hop:
      Alice picked up the red bag in the kitchen.
      Alice gave the red bag to Carol.
      Carol left the red bag near the window.
      Q: Where is the object Alice first picked up?
      A: near the window
    """
    people = _pick(rng, PEOPLE, 3)
    obj = rng.choice(OBJECTS)
    places = _pick(rng, PLACES, 3)
    p0, p1, p2 = people
    loc0, loc1, loc2 = places

    sents = [
        f"{p0} picked up the {obj} in {loc0}.",
    ]
    # distractor events about other objects/people
    other_obj = rng.choice([o for o in OBJECTS if o != obj])
    other_person = rng.choice([p for p in PEOPLE if p not in people])
    other_place = rng.choice([p for p in PLACES if p not in places])
    sents.append(f"{other_person} moved the {other_obj} to {other_place}.")

    if hops == 2:
        sents.append(f"{p0} left the {obj} near {loc1}.")
        gold = f"near {loc1}"
        q = f"Where is the object {p0} first picked up?"
        sketch = ["GROUND", "TRACE", "COMPOSE", "STOP"]
        qtype = "two-hop"
    else:
        sents.append(f"{p0} gave the {obj} to {p1}.")
        sents.append(f"{p1} left the {obj} near {loc2}.")
        # extra distractor handoff
        sents.append(f"{p2} later picked up a different item in {loc1}.")
        gold = f"near {loc2}"
        q = f"Where is the object {p0} first picked up?"
        sketch = ["GROUND", "TRACE", "LINK", "COMPOSE"]
        qtype = "three-hop"

    # shuffle sentence order partially but keep first pickup early for readability
    body = sents[1:]
    rng.shuffle(body)
    context = "\n".join([sents[0]] + body)

    # hard candidates: gold + other places/people/objects mentioned
    pool = [
        f"near {loc0}",
        f"near {loc1}",
        f"near {loc2}",
        f"in {loc0}",
        f"in {other_place}",
        other_person,
        p1,
        p2,
        other_obj,
        f"with {p0}",
        f"with {p1}",
        "unknown",
    ]
    distractors = []
    for c in pool:
        if c.lower() == gold.lower():
            continue
        if c.lower() in {d.lower() for d in distractors}:
            continue
        distractors.append(c)
        if len(distractors) >= 7:
            break
    while len(distractors) < 7:
        distractors.append(f"decoy-{len(distractors)}")

    return MHExample(
        id=f"synth-{rng.randrange(1<<30)}",
        question=q,
        context=context,
        answer=gold,
        qtype=qtype,
        candidates=[gold] + distractors,
        action_sketch=sketch,
    )


def make_synthetic_split(n: int, seed: int, hops: tuple[int, ...] = (2, 3)) -> list[MHExample]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        h = hops[i % len(hops)]
        out.append(_make_story(rng, h))
    return out


def load_synthetic_sets(
    n_train: int = 2048,
    n_val: int = 512,
    seed: int = 0,
) -> tuple[list[MHExample], list[MHExample]]:
    train = make_synthetic_split(n_train, seed=seed, hops=(2, 3))
    val = make_synthetic_split(n_val, seed=seed + 17, hops=(2, 3))
    return train, val
