#!/usr/bin/env python3
"""Run EndoComparer on two-video questions and write a results file that ``endodiffvqa.metrics`` scores.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from endodiffvqa.frames import FrameSource
from endodiffvqa.prompts import build_messages, letter_map_for, video_order
from endodiffvqa.dataset import clip_frame_counts, frame_counts_of, load_jsonl, load_dataset
from .model import DEFAULT_BASE, DEFAULT_WEIGHTS, NO_THINK, load_endocomparer
from .psvl import SLOT_MAPS
from .ptc import PTC_INSTRUCTION, parse_final_letter


def parse_args(argv=None):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None, help="YAML with defaults (configs/eval_default.yaml)")
    pre_args, _ = pre.parse_known_args(argv)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 parents=[pre])
    ap.add_argument("--model", default=DEFAULT_BASE)
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--out", required=True, help="results JSON path")
    ap.add_argument("--dataset-root", default=None)
    ap.add_argument("--split", default="test", choices=("train", "val", "test"))
    ap.add_argument("--jsonl", default=None, help="items in the JSONL schema instead of a dataset split")
    ap.add_argument("--video-root", default=None)
    ap.add_argument("--controls", default="full")
    ap.add_argument("--slot-map", default="12", choices=SLOT_MAPS)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--frames-caption", type=int, default=16)
    ap.add_argument("--frames-skill", type=int, default=64)
    ap.add_argument("--max-pixels", type=int, default=784 * 448)
    ap.add_argument("--frame-cache", default="cache/frames")
    ap.add_argument("--load-workers", type=int, default=4, help="decode processes for clips missing from the cache")
    ap.add_argument("--chunk", type=int, default=16, help="items per prefetch / flush")
    ap.add_argument("--limit", type=int, default=None, help="first N items")
    ap.add_argument("--per-source", type=int, default=None, help="first N items of each data_source")
    ap.add_argument("--sources", default=None, help="comma list of data_source to keep")
    ap.add_argument("--ids", default=None, help="file with one item id per line: keep these items only")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0, help="this job's shard (items[shard::num_shards])")
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--no-score", action="store_true")
    if pre_args.config:
        import yaml

        cfg = yaml.safe_load(Path(pre_args.config).read_text()) or {}
        known = {a.dest for a in ap._actions}
        unknown = sorted(set(cfg) - known)
        if unknown:
            raise SystemExit(f"{pre_args.config}: unknown keys {unknown}")
        ap.set_defaults(**cfg)
    return ap.parse_args(argv)


def select_items(args):
    if args.jsonl:
        items, counts = load_jsonl(args.jsonl, args.video_root), None
        counts = frame_counts_of(items)
    else:
        items, counts = load_dataset(args.split, args.dataset_root), clip_frame_counts(args.dataset_root)
    if args.sources:
        keep = set(args.sources.split(","))
        items = [it for it in items if it["data_source"] in keep]
    if args.ids:
        keep = {ln.strip() for ln in Path(args.ids).read_text().splitlines() if ln.strip()}
        items = [it for it in items if it["id"] in keep]
    if args.per_source:
        seen = {}
        items = [it for it in items if len(seen.setdefault(it["data_source"], [])) < args.per_source
                 and not seen[it["data_source"]].append(1)]
    if args.limit:
        items = items[: args.limit]
    if args.num_shards > 1:
        assert 0 <= args.shard < args.num_shards
        items = items[args.shard::args.num_shards]
    return items, counts


@torch.inference_mode()
def generate(model, processor, text, videos, max_new_tokens):
    kw = {}
    if videos:
        kw = {"videos": [torch.from_numpy(a) for a, _ in videos], "video_metadata": [m for _, m in videos],
              "do_sample_frames": False}
    enc = processor(text=[text], return_tensors="pt", **kw)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    n_prompt = enc["input_ids"].shape[1]
    eos = model.generation_config.eos_token_id
    gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None,
                         top_k=None, eos_token_id=eos, pad_token_id=processor.tokenizer.pad_token_id)
    new = gen[0, n_prompt:].tolist()
    eos_set = set(eos if isinstance(eos, (list, tuple)) else [eos])
    cut = next((i for i, t in enumerate(new) if t in eos_set), len(new))
    return processor.tokenizer.decode(new[:cut], skip_special_tokens=True)


def main(argv=None):
    args = parse_args(argv)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tag = f".shard{args.shard}of{args.num_shards}" if args.num_shards > 1 else ""
    partial = out.with_name(out.stem + f".partial{tag}.jsonl")
    controls = [c for c in args.controls.split(",") if c]
    items, counts = select_items(args)
    print(f"[items] {len(items)} items, controls {controls}", flush=True)

    done = {}
    if partial.exists():
        for line in partial.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[(r["id"], r["control"])] = r
        print(f"[resume] {len(done)} records already done", flush=True)

    processor, model, info = load_endocomparer(args.model, args.weights, args.slot_map, args.attn)
    frames = FrameSource(args.frame_cache, args.frames_caption, args.frames_skill, args.max_pixels,
                         args.load_workers, counts)
    t0, n_new = time.time(), 0
    with open(partial, "a") as pf:
        for c0 in range(0, len(items), args.chunk):
            chunk = items[c0: c0 + args.chunk]
            jobs = [(it, c) for it in chunk for c in controls if (it["id"], c) not in done]
            frames.prefetch([e for it, c in jobs for e in frames.elements(it, video_order(it, c))])
            for it, c in jobs:
                lmap = letter_map_for(it["id"], c)
                eles = frames.elements(it, video_order(it, c))
                msgs = build_messages(it, c, lmap, PTC_INSTRUCTION, eles)
                text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                     enable_thinking=False)
                assert text.endswith(NO_THINK), text[-80:]
                raw = generate(model, processor, text, [frames.load(e) for e in eles], args.max_new_tokens)
                letter = parse_final_letter(raw)
                rec = {"id": it["id"], "control": c, "letter_map": lmap,
                       "prediction": f"({letter})" if letter else raw, "raw_prediction": raw}
                pf.write(json.dumps(rec) + "\n")
                done[(it["id"], c)] = rec
                n_new += 1
            pf.flush()
            frames.clear()
            el = time.time() - t0
            print(f"[infer] {min(c0 + args.chunk, len(items))}/{len(items)} items | {n_new} new | {el:.0f} s | "
                  f"{el / max(n_new, 1):.2f} s/record", flush=True)

    want = {(it["id"], c) for it in items for c in controls}
    res = sorted((r for k, r in done.items() if k in want), key=lambda r: (r["control"], r["id"]))
    out_tag = out.with_name(out.stem + tag + out.suffix)
    out_tag.write_text(json.dumps(res))
    n_fail = sum(1 for r in res if not r["prediction"].startswith("("))
    cfg = {**vars(args), "n_items": len(items), "n_records": len(res), "no_final_letter": n_fail,
           "instruction": PTC_INSTRUCTION, "model_info": info, "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    out_tag.with_name(out_tag.stem + ".config.json").write_text(json.dumps(cfg, indent=1))
    print(f"[done] wrote {out_tag} ({len(res)} records, {n_fail} without a final letter)", flush=True)

    if not args.no_score and args.num_shards == 1 and all(it.get("answer_letter") for it in items):
        from endodiffvqa.metrics import score, summary_md

        m = score(res, {it["id"]: it for it in items})
        out.with_name("metrics.json").write_text(json.dumps(m, indent=1))
        md = summary_md(m, f"EndoDiffVQA metrics — {out.name}")
        out.with_name("metrics.md").write_text(md)
        print(md, flush=True)


if __name__ == "__main__":
    main()
