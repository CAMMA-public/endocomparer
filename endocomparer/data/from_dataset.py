"""Convert EndoDiffVQA dataset splits to the training / inference JSONL schema (``endocomparer/data/README.md``).

The dataset is not public yet (coming soon); with a dataset directory (``endodiffvqa/README.md``)::

    python -m endocomparer.data.from_dataset --dataset-root data/EndoDiffVQA --splits train,val \\
        --out data/endodiffvqa_trainval.jsonl

Rows are written in dataset order (train, then val); ``endocomparer.train`` shuffles them with its seed, so the paper
run's instances are reproduced exactly from this file. Video paths are absolute by default; ``--relative-to DIR``
writes them relative to ``DIR`` (then pass ``--video-root DIR`` to the consumers).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from endodiffvqa.metrics import swap_answer
from endodiffvqa.dataset import clip_frame_counts, load_dataset

FIELDS = ("id", "data_source", "template_id", "category", "attribute", "question", "options", "answer_letter",
          "option_codes", "swap_answer_letter", "cell", "value_a", "value_b", "score_a", "score_b", "subject_keys",
          "video1", "video2", "frames1", "frames2", "pair_id")


def to_record(it: dict, counts: dict, relative_to: Path | None = None) -> dict:
    rec = {k: it.get(k) for k in FIELDS}
    rec["swap_answer_letter"] = swap_answer(it)
    rec["frames1"], rec["frames2"] = counts.get(it["video1"]), counts.get(it["video2"])
    if relative_to is not None:
        rec["video1"] = str(Path(it["video1"]).relative_to(relative_to))
        rec["video2"] = str(Path(it["video2"]).relative_to(relative_to))
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", default=None)
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--out", required=True)
    ap.add_argument("--relative-to", default=None, help="write video paths relative to this directory")
    ap.add_argument("--limit", type=int, default=None, help="first N rows (smoke tests)")
    args = ap.parse_args()

    counts = clip_frame_counts(args.dataset_root)
    rel = Path(args.relative_to).resolve() if args.relative_to else None
    items = [it for s in args.splits.split(",") for it in load_dataset(s, args.dataset_root)]
    if args.limit:
        items = items[: args.limit]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for it in items:
            f.write(json.dumps(to_record(it, counts, rel), ensure_ascii=False) + "\n")
    print(f"wrote {len(items)} rows -> {out}")


if __name__ == "__main__":
    main()
