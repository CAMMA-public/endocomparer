"""The shared frame protocol: how a clip becomes the frames a model sees.

Sampling is ``qwen_vl_utils.process_vision_info`` (the Qwen reference): ``nframes`` frames uniformly over the whole
clip (``linspace(0, total - 1, nframes)``), each frame capped at ``max_pixels`` with the aspect ratio kept and the
sides rounded to multiples of 32 (``image_patch_size=16``), and ``return_video_metadata=True`` so the true fps and
``frames_indices`` travel with the frames: the model gets correct timestamps and nothing is resampled downstream
(``do_sample_frames=False``).

Budget per video: caption clips 16 frames, skill clips 64 (``is_skill``), capped at the clip's even native frame
count (a 3-frame clip gives 2). Per-frame pixels ≤ 784 × 448.

Decoding runs in spawned worker processes (decord is not thread-safe) and every result is cached on disk as uint8
``(T, C, H, W)`` plus a metadata JSON::

    {cache_root}/{parent dir name}/{stem}__n{nframes}_p{max_pixels}.npy  (+ .json)

Build the cache for a dataset split once with ``python -m endodiffvqa.frames --dataset-root R --splits test``;
inference and training decode on the fly whatever is missing.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np

# qwen_vl_utils prefers torchcodec when importable, whose FFmpeg libraries many clusters lack; decord works everywhere.
os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")

from .dataset import is_skill  # noqa: E402

DEFAULT_MAX_PIXELS = 784 * 448
DEFAULT_CACHE = Path(os.environ.get("ENDODIFFVQA_FRAME_CACHE", "cache/frames"))


def probe_frames(path: str) -> int:
    """Native frame count of a clip (decord)."""
    import decord

    return len(decord.VideoReader(path, num_threads=1))


def video_element(path: str, skill: bool, native_frames: int, frames_caption: int = 16, frames_skill: int = 64,
                  max_pixels: int = DEFAULT_MAX_PIXELS) -> dict:
    """The qwen_vl_utils video element for one clip under the protocol."""
    want = frames_skill if skill else frames_caption
    n = max(2, min(want, native_frames // 2 * 2))
    return {"type": "video", "video": f"file://{path}", "nframes": n, "max_pixels": max_pixels,
            "total_pixels": n * max_pixels}


def cache_path(ele: dict, cache_root: Path) -> Path:
    p = Path(ele["video"].replace("file://", ""))
    return Path(cache_root) / p.parent.name / f"{p.stem}__n{ele['nframes']}_p{ele['max_pixels']}.npy"


def decode(ele: dict):
    """qwen_vl_utils -> (uint8 array (T, C, H, W), metadata dict)."""
    import torch
    from qwen_vl_utils import process_vision_info

    _, videos, kwargs = process_vision_info([{"role": "user", "content": [ele]}], image_patch_size=16,
                                            return_video_kwargs=True, return_video_metadata=True)
    assert kwargs.get("do_sample_frames") is False, kwargs
    frames, meta = videos[0]
    assert len(meta["frames_indices"]) == frames.shape[0] == ele["nframes"], (ele, tuple(frames.shape))
    arr = frames.round().clamp(0, 255).to(dtype=torch.uint8).numpy()
    meta = {k: (list(map(int, v)) if k == "frames_indices" else v) for k, v in meta.items()}
    return arr, meta


def _decode_to_disk(ele: dict, cache_root: str) -> str:
    out = cache_path(ele, Path(cache_root))
    if out.exists() and out.with_suffix(".json").exists():
        return str(out)
    arr, meta = decode(ele)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + f".tmp{os.getpid()}.npy")
    np.save(tmp, arr)
    os.replace(tmp, out)
    with open(out.with_suffix(".json.tmp"), "w") as f:
        json.dump(meta, f)
    os.replace(out.with_suffix(".json.tmp"), out.with_suffix(".json"))
    return str(out)


def _decode_task(args):
    """Pool task: never raises, so one bad clip cannot take the pool down."""
    ele, cache_root = args
    try:
        return ele["video"], _decode_to_disk(ele, cache_root), None
    except Exception as ex:  # noqa: BLE001
        return ele["video"], None, repr(ex)


def _spawn_pool(workers: int):
    return mp.get_context("spawn").Pool(workers, maxtasksperchild=16)


def load_cached(ele: dict, cache_root: Path):
    out = cache_path(ele, cache_root)
    with open(out.with_suffix(".json")) as f:
        meta = json.load(f)
    return np.load(out), meta


class FrameSource:
    """Frames for request building: disk cache first, else decode in a spawned process pool (and cache)."""

    def __init__(self, cache_root: Path = DEFAULT_CACHE, frames_caption: int = 16, frames_skill: int = 64,
                 max_pixels: int = DEFAULT_MAX_PIXELS, workers: int = 8, clip_frames: Optional[Dict[str, int]] = None):
        self.cache_root = Path(cache_root)
        self.fc, self.fs, self.max_pixels = frames_caption, frames_skill, max_pixels
        self.clip_frames: Dict[str, int] = dict(clip_frames or {})
        self.workers = workers
        self._pool = None
        self.mem = {}

    def native_frames(self, path: str) -> int:
        if path not in self.clip_frames:
            self.clip_frames[path] = probe_frames(path)
        return self.clip_frames[path]

    def element(self, path: str, item: dict) -> dict:
        return video_element(path, is_skill(item), self.native_frames(path), self.fc, self.fs, self.max_pixels)

    def elements(self, item: dict, paths: Iterable[str]) -> list:
        return [self.element(p, item) for p in paths]

    def pool(self):
        if self._pool is None:
            self._pool = _spawn_pool(self.workers)
        return self._pool

    def prefetch(self, eles):
        """Make sure every element is on disk (decoding the missing ones in parallel)."""
        todo = {cache_path(e, self.cache_root): e for e in eles}
        todo = [e for p, e in todo.items() if not (p.exists() and p.with_suffix(".json").exists())]
        if todo:
            for video, _, err in self.pool().imap_unordered(_decode_task, [(e, str(self.cache_root)) for e in todo]):
                if err:
                    raise RuntimeError(f"decode failed for {video}: {err}")

    def load(self, ele: dict):
        key = (ele["video"], ele["nframes"], ele["max_pixels"])
        if key not in self.mem:
            self.prefetch([ele])
            self.mem[key] = load_cached(ele, self.cache_root)
        return self.mem[key]

    def clear(self):
        self.mem.clear()


def main():
    from .dataset import clip_frame_counts, frame_counts_of, load_jsonl, load_dataset

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", default=None)
    ap.add_argument("--splits", default="test")
    ap.add_argument("--jsonl", default=None, help="instead of the dataset: items in the JSONL schema")
    ap.add_argument("--frames-caption", type=int, default=16)
    ap.add_argument("--frames-skill", type=int, default=64)
    ap.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS)
    ap.add_argument("--cache-root", default=str(DEFAULT_CACHE))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) - 2))
    args = ap.parse_args()

    if args.jsonl:
        items = load_jsonl(args.jsonl)
        counts = frame_counts_of(items)
    else:
        items = [it for s in args.splits.split(",") for it in load_dataset(s, args.dataset_root)]
        counts = clip_frame_counts(args.dataset_root)
    fs = FrameSource(args.cache_root, args.frames_caption, args.frames_skill, args.max_pixels, args.workers, counts)
    eles = {}
    for it in items:
        for p in (it["video1"], it["video2"]):
            e = fs.element(p, it)
            eles[cache_path(e, Path(args.cache_root))] = e
    todo = [e for p, e in eles.items() if not (p.exists() and p.with_suffix(".json").exists())]
    todo.sort(key=lambda e: -e["nframes"])
    print(f"{len(eles)} cache entries, {len(todo)} to decode, {args.workers} workers", flush=True)
    t0, fails = time.time(), []
    pool = _spawn_pool(args.workers)
    for i, (video, _, err) in enumerate(pool.imap_unordered(_decode_task, [(e, args.cache_root) for e in todo]), 1):
        if err:
            fails.append((video, err))
        if i % 100 == 0 or i == len(todo):
            print(f"{i}/{len(todo)} decoded | {time.time() - t0:.0f}s | {len(fails)} failed", flush=True)
    pool.close()
    pool.join()
    for v, ex in fails:
        print("FAIL", v, ex)
    if fails:
        sys.exit(f"{len(fails)} clips failed to decode")
    print("frame cache complete:", args.cache_root)


if __name__ == "__main__":
    main()
