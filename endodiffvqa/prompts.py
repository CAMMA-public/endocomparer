"""The shared evaluation prompt: two videos, the question stem, four lettered options and one instruction.

Every model in the paper (zero-shot or fine-tuned) sees this prompt. The option order is the dataset order for the
``full`` control; ``perm{k}`` controls shuffle it with a seed derived from the item id; the ``swap`` control exchanges
the two videos.
"""

from __future__ import annotations

import random
from typing import Dict, List

from .dataset import LETTERS

LETTER_INSTRUCTION = "Answer with the letter of the correct option (a, b, c or d) only."
CONTROLS = ("full", "swap")


def letter_map_for(item_id: str, control: str) -> Dict[str, str]:
    """displayed letter -> original letter."""
    if not control.startswith("perm") or control == "perm0":
        return {k: k for k in LETTERS}
    orig = list(LETTERS)
    random.Random(f"{item_id}|{control}").shuffle(orig)
    return dict(zip(LETTERS, orig))


def render_question(item: Dict, lmap: Dict[str, str]) -> str:
    lines = [item["question"]] + [f"({d}) {item['options'][lmap[d]]}" for d in LETTERS]
    return "\n".join(lines)


def video_order(item: Dict, control: str) -> List[str]:
    if control == "swap":
        return [item["video2"], item["video1"]]
    return [item["video1"], item["video2"]]


def build_messages(item: Dict, control: str, lmap: Dict[str, str], instruction: str, video_eles: list) -> list:
    """One user turn: ``Video 1:`` <video> ``Video 2:`` <video> question + options + instruction."""
    content = []
    for i, ele in enumerate(video_eles, 1):
        content.append({"type": "text", "text": f"Video {i}:"})
        content.append(ele)
    text = render_question(item, lmap)
    content.append({"type": "text", "text": f"{text}\n\n{instruction}"})
    return [{"role": "user", "content": content}]
