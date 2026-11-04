"""Perceive-then-Compare chain of thought (PtC-CoT): the training target and its parser.

Target, after the thinking-off prefix of the chat template::

    Video 1: {h1}. Video 2: {h2}. ({x})

``h1`` / ``h2`` are the per-video facts the question was built from, rendered as short perception statements
(``per_video_halves``); ``x`` is the displayed letter of the correct option. The letter is thereby conditioned on the
model's own statements about each video.

Per-video statements come from the item's fact fields only::

    presence (A1, A6, T1, T6)          cell BOTH / ONLY_A / ONLY_B / NEITHER  ->  "visible" / "not visible" (A1),
                                       "used" / "not used" (T1), "engaged" / "not engaged" (T6),
                                       the state word of subject_keys[-1] or "not {state}" (A6)
    count (T5)                         value_a / value_b (instrument counts)  ->  "2 instruments"
    attribute (A2, A4, T3, C1, C2, C4) value_a / value_b; None = the subject is not visible in that video
    skill (S1, S2)                     score_a / score_b  ->  "score 3"

Augmentation, one draw per training instance from a seeded RNG (so the run is reproducible)::

    order    p = ½: Video 1 <-> Video 2 ("swap"); the halves swap and the gold becomes the swap answer (the
             video-swap distractor D1 for an asymmetric gold, the same option for a symmetric one); items with no
             swap answer offered keep their order
"""

from __future__ import annotations

import random
import re
from collections import Counter
from typing import Dict, List, Tuple

from endodiffvqa.metrics import swap_answer
from endodiffvqa.dataset import LETTERS, template_kind

PTC_INSTRUCTION = ("First state what each video shows for this question, then give the letter of the correct option "
                   "(a, b, c or d).")

_PRESENCE = {  # template -> (yes phrase, no phrase); A6 uses the state word from subject_keys
    "A1": ("visible", "not visible"),
    "T1": ("used", "not used"),
    "T6": ("engaged", "not engaged"),
}
_PRESENCE_CELL = {"BOTH": (True, True), "ONLY_A": (True, False), "ONLY_B": (False, True), "NEITHER": (False, False)}


def _presence_word(item: Dict, present: bool) -> str:
    tid = item["template_id"]
    if tid == "A6":
        state = item["subject_keys"][-1]
        return state if present else f"not {state}"
    yes, no = _PRESENCE[tid]
    return yes if present else no


def _count_word(v) -> str:
    n = int(v)
    return f"{n} instrument" + ("" if n == 1 else "s")


def per_video_halves(item: Dict) -> Tuple[str, str]:
    """(Video 1 statement, Video 2 statement) in the item's own video order."""
    kind = template_kind(item)
    if kind == "presence":
        a, b = _PRESENCE_CELL[item["cell"]]
        return _presence_word(item, a), _presence_word(item, b)
    if kind == "count":
        return _count_word(item["value_a"]), _count_word(item["value_b"])
    if kind == "attribute":
        f = lambda v: "not visible" if v is None else str(v)  # noqa: E731
        return f(item["value_a"]), f(item["value_b"])
    return f"score {item['score_a']}", f"score {item['score_b']}"  # skill


def answer_text(h1: str, h2: str, letter: str) -> str:
    return f"Video 1: {h1}. Video 2: {h2}. ({letter})"


def make_instance(item: Dict, rng: random.Random, augment: bool = True) -> Dict:
    """One training instance: the item, its video order (``control`` full / swap), its option map and its target."""
    h1, h2 = per_video_halves(item)
    control, gold = "full", item["answer_letter"]
    if augment and rng.random() < 0.5:
        sw = swap_answer(item)
        if sw is not None:
            control, gold, h1, h2 = "swap", sw, h2, h1
    lmap = {k: k for k in LETTERS}  # displayed -> original: options are shown in dataset order
    return {"item": item, "control": control, "lmap": lmap, "answer": answer_text(h1, h2, gold),
            "instruction": PTC_INSTRUCTION, "gold_original": gold}


def build_instances(items: List[Dict], seed: int, augment: bool = True):
    """Training instances in a seeded order. Returns ``(instances, stats)``."""
    rng = random.Random(seed)
    pool = list(items)
    rng.shuffle(pool)
    inst = [make_instance(it, rng, augment) for it in pool]
    stats = {"n_items": len(items), "n_instances": len(inst),
             "controls": dict(Counter(x["control"] for x in inst))}
    return inst, stats


_FINAL = re.compile(r"\(([a-d])\)")


def parse_final_letter(text: str | None) -> str | None:
    """Last ``(x)`` in a generated PtC-CoT answer, or None."""
    m = _FINAL.findall(text or "")
    return m[-1] if m else None
