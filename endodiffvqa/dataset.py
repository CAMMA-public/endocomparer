"""Load EndoDiffVQA items

The dataset expects a directory with this layout::

    data/{train,val,test}.parquet     one row per item
    videos/{source}/{file}.mp4        clips
    metadata/videos.json              per-clip native frame count, fps, size

``load_dataset(split, root)`` and ``load_jsonl`` map the raw dataset files to the item dict;

Item fields used downstream::

    id              unique string
    data_source     cholect50_caption | multibypass_caption | chole_score | multibypass_skill (or your own name)
    template_id     A1 … T6, S1, S2 — decides the template kind (how per-video facts render, Strict grouping)
    category        Anatomy | Tool | Action | Skill
    question        the question (without the options)
    options         e.g. {"a": text, "b": text, "c": text, "d": text}
    answer_letter   a | b | c | d
    option_codes    {"a": "correct" | "D1" | "D2" | "D3" | "D4", ...}  (optional; D1 = the video-swap distractor)
    swap_answer_letter   gold letter once the two videos are exchanged, or None (optional; derived from option_codes)
    cell, value_a, value_b, score_a, score_b, subject_keys   per-video facts for Perceive-then-Compare targets
    video1, video2  absolute paths for videos
    frames1, frames2     native frame counts (optional; probed with decord when absent)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

SPLITS = ("train", "val", "test")
LETTERS = ("a", "b", "c", "d")
ALL_SOURCES = ("cholect50_caption", "multibypass_caption", "chole_score", "multibypass_skill")
CAPTION_SOURCES = ("cholect50_caption", "multibypass_caption")
SKILL_SOURCES = ("chole_score", "multibypass_skill")

# Kind of every template: presence (both / only 1 / only 2 / neither), attribute (two observed values), count,
# skill (same standard / clearly or slightly better in one video).
TEMPLATE_KIND = {
    "A1": "presence", "A5": "presence", "A6": "presence", "T1": "presence", "T6": "presence",
    "A2": "attribute", "A3": "attribute", "A4": "attribute", "T3": "attribute",
    "C1": "attribute", "C2": "attribute", "C4": "attribute",
    "T5": "count",
    "S1": "skill", "S2": "skill",
}
KINDS = ("presence", "attribute", "count", "skill")

REQUIRED_FIELDS = ("id", "data_source", "template_id", "question", "options", "video1", "video2")


def template_kind(item: Dict) -> str:
    """Kind of the item's template (decides the per-video fact rendering and the Strict grouping)."""
    tid = item["template_id"]
    if tid in TEMPLATE_KIND:
        return TEMPLATE_KIND[tid]
    kind = item.get("template_kind")
    if kind in KINDS:
        return kind
    raise KeyError(f"item {item.get('id')}: unknown template_id {tid!r} and no 'template_kind' field "
                   f"(one of {', '.join(KINDS)})")


def is_skill(item: Dict) -> bool:
    """Skill items (phase-level clips of many minutes) get the larger frame budget."""
    return item["data_source"] in SKILL_SOURCES or item.get("category") == "Skill"


# --------------------------------------------------------------------------------------------------------------
# Dataset directory
# --------------------------------------------------------------------------------------------------------------
def dataset_root(root: Optional[str | Path] = None) -> Path:
    """``root`` argument, else ``$ENDODIFFVQA_ROOT``, else ``./data/EndoDiffVQA``."""
    r = Path(root or os.environ.get("ENDODIFFVQA_ROOT") or "data/EndoDiffVQA")
    if not (r / "data").is_dir():
        raise FileNotFoundError(
            f"{r} does not look like the EndoDiffVQA dataset directory (no data/ directory). Pass it with "
            "--dataset-root or set ENDODIFFVQA_ROOT (the dataset is not public yet; see README).")
    return r


def normalize_row(row: Dict, root: Path) -> Dict:
    it = dict(row)
    it["data_source"] = it.pop("source")
    opts, codes = it["options"], it.pop("distractor_codes")
    assert len(opts) == 4 and len(codes) == 4, it["id"]
    it["options"] = dict(zip(LETTERS, opts))
    it["option_codes"] = dict(zip(LETTERS, codes))
    it["question"] = it.pop("question_stem")
    it["video1"] = str(root / it["video1"])
    it["video2"] = str(root / it["video2"])
    it["videos"] = [it["video1"], it["video2"]]
    return it


def load_dataset(split: str, root: Optional[str | Path] = None) -> List[Dict]:
    """Items of one split, in dataset order. Needs ``pyarrow``."""
    import pyarrow.parquet as pq

    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    r = dataset_root(root)
    rows = pq.read_table(r / "data" / f"{split}.parquet").to_pylist()
    return [normalize_row(x, r) for x in rows]


def load_dataset_index(splits: Iterable[str] = SPLITS, root: Optional[str | Path] = None) -> Dict[str, Dict]:
    """id -> item over the given splits (ids are unique across splits)."""
    out: Dict[str, Dict] = {}
    for s in splits:
        for it in load_dataset(s, root):
            assert it["id"] not in out, it["id"]
            out[it["id"]] = it
    return out


def clip_frame_counts(root: Optional[str | Path] = None) -> Dict[str, int]:
    """absolute clip path -> native frame count, from the dataset's ``metadata/videos.json``."""
    r = dataset_root(root)
    with open(r / "metadata" / "videos.json") as f:
        return {str(r / m["path"]): int(m["frames"]) for m in json.load(f)}


# --------------------------------------------------------------------------------------------------------------
# JSONL (your own items, or the dataset converted by endocomparer/data/from_dataset.py)
# --------------------------------------------------------------------------------------------------------------
def normalize_jsonl_item(it: Dict, video_root: Optional[Path] = None) -> Dict:
    """Validate one JSONL record and fill the derived fields."""
    it = dict(it)
    missing = [k for k in REQUIRED_FIELDS if k not in it]
    if missing:
        raise ValueError(f"item {it.get('id')}: missing fields {missing}")
    if isinstance(it["options"], list):
        it["options"] = dict(zip(LETTERS, it["options"]))
    if set(it["options"]) != set(LETTERS):
        raise ValueError(f"item {it['id']}: options must be keyed a-d, got {sorted(it['options'])}")
    if "option_codes" in it and isinstance(it["option_codes"], list):
        it["option_codes"] = dict(zip(LETTERS, it["option_codes"]))
    if it.get("answer_letter") is not None and it["answer_letter"] not in LETTERS:
        raise ValueError(f"item {it['id']}: answer_letter {it['answer_letter']!r}")
    for k in ("video1", "video2"):
        p = Path(it[k])
        if not p.is_absolute() and video_root is not None:
            p = video_root / p
        it[k] = str(p)
    it["videos"] = [it["video1"], it["video2"]]
    it.setdefault("category", "Skill" if it["data_source"] in SKILL_SOURCES else None)
    it.setdefault("subject_keys", [])
    for k in ("cell", "value_a", "value_b", "score_a", "score_b"):
        it.setdefault(k, None)
    template_kind(it)  # raises on an unknown template without a template_kind
    return it


def load_jsonl(path: str | Path, video_root: Optional[str | Path] = None) -> List[Dict]:
    """Items of a JSONL file (one JSON object per line). Relative video paths resolve against ``video_root``
    (default: the file's directory)."""
    path = Path(path)
    vr = Path(video_root) if video_root else path.resolve().parent
    items = []
    with open(path) as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(normalize_jsonl_item(json.loads(line), vr))
            except Exception as e:  # noqa: BLE001
                raise ValueError(f"{path}:{n}: {e}") from e
    ids = [it["id"] for it in items]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate ids")
    return items


def frame_counts_of(items: Iterable[Dict]) -> Dict[str, int]:
    """Native frame counts carried by the items (``frames1`` / ``frames2``), keyed by path."""
    out = {}
    for it in items:
        for k, fk in (("video1", "frames1"), ("video2", "frames2")):
            if it.get(fk):
                out[it[k]] = int(it[fk])
    return out
