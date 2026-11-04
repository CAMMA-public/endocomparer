#!/usr/bin/env python3
"""Train EndoComparer: language-model LoRA + Pseudo-Siamese Visual LoRA on Perceive-then-Compare targets.

    torchrun --standalone --nproc_per_node=2 -m endocomparer.train --config configs/endocomparer_qwen35_9b.yaml \\
        --data data/endodiffvqa_trainval.jsonl --out runs/endocomparer

Items come from a JSONL file in the schema of ``endocomparer/data/README.md`` (``--data``), or straight from the
dataset (``--from-dataset --dataset-root R --splits train,val``). They are shuffled with ``--seed``, turned into
PtC-CoT instances (order swap p ½; ``endocomparer.ptc.build_instances``) and
trained for one epoch with the loss on the response tokens only. Thinking is off: the chat template's empty think
block is part of the prompt.

Checkpoints every ``--save-steps`` optimizer steps hold the trainer state, the language-model adapter and the PSVL
weights (``dual_vision_lora.pt``); a rerun with the same ``--out`` resumes from the latest complete checkpoint, and
exits at once when ``final/DONE.json`` says the run finished. The final weights land in ``{out}/final``:
``adapter_config.json``, ``adapter_model.safetensors``, ``dual_vision_lora.pt``, ``psvl_config.json``,
``train_summary.json``.

Every value in the YAML config can be overridden on the command line (``--lr 5e-5``).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from endodiffvqa.frames import FrameSource
from endodiffvqa.dataset import clip_frame_counts, frame_counts_of, load_jsonl, load_dataset
from .data import Collator, PtCDataset
from .model import DEFAULT_BASE, attach_for_training, check_gdn_kernels, load_base
from .psvl import STATE_FILENAME, slot_divergence
from .ptc import build_instances

DONE = "DONE.json"


def latest_checkpoint(out: Path):
    """Latest checkpoint that holds BOTH the trainer state and the slot weights (a job killed between the two writes
    leaves a checkpoint-N without dual_vision_lora.pt; save_total_limit 2 keeps the one before it)."""
    cks = [p for p in out.glob("checkpoint-*") if p.is_dir() and (p / "trainer_state.json").exists()
           and (p / STATE_FILENAME).exists()]
    return max(cks, key=lambda p: int(p.name.split("-")[1])) if cks else None


def parse_args(argv=None):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None, help="YAML with defaults for every option below")
    pre_args, rest = pre.parse_known_args(argv)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 parents=[pre])
    ap.add_argument("--out", required=True, help="run directory (checkpoints, final/, train_config.json)")
    ap.add_argument("--model", default=DEFAULT_BASE, help="base model: local directory or hub id")
    ap.add_argument("--data", default=None, help="training items, JSONL")
    ap.add_argument("--video-root", default=None, help="base for relative video paths in --data")
    ap.add_argument("--from-dataset", action="store_true", help="train on dataset splits instead of --data")
    ap.add_argument("--dataset-root", default=None)
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--frame-cache", default="cache/frames")
    ap.add_argument("--frames-caption", type=int, default=16)
    ap.add_argument("--frames-skill", type=int, default=64)
    ap.add_argument("--max-pixels", type=int, default=784 * 448)
    ap.add_argument("--epochs", type=float, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--grad-accum", type=int, default=8, help="x world size x 1 = effective batch")
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--psvl-r", type=int, default=32)
    ap.add_argument("--psvl-alpha", type=int, default=64)
    ap.add_argument("--psvl-dropout", type=float, default=0.05)
    ap.add_argument("--psvl-mode", choices=("dual", "shared"), default="dual",
                    help="shared = one adapter for both videos (ablation)")
    ap.add_argument("--no-augment", action="store_true", help="no order swap (ablation)")
    ap.add_argument("--save-steps", type=int, default=50)
    ap.add_argument("--save-total-limit", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None, help="first N shuffled items (smoke tests)")
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=20260928)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--attn", default="flash_attention_2")
    if pre_args.config:
        import yaml

        cfg = yaml.safe_load(Path(pre_args.config).read_text()) or {}
        known = {a.dest for a in ap._actions}
        unknown = sorted(set(cfg) - known)
        if unknown:
            raise SystemExit(f"{pre_args.config}: unknown keys {unknown}")
        ap.set_defaults(**cfg)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.out)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    is_main = local_rank == 0

    from transformers import Trainer, TrainerCallback, TrainingArguments
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule  # noqa: F401

    # --- items -> PtC-CoT instances -> dataset ----------------------------------------------------------------
    if args.from_dataset:
        items = [it for sp in args.splits.split(",") for it in load_dataset(sp, args.dataset_root)]
        counts = clip_frame_counts(args.dataset_root)
    else:
        if not args.data:
            raise SystemExit("--data JSONL (or --from-dataset) is required")
        items = load_jsonl(args.data, args.video_root)
        counts = frame_counts_of(items)
    missing = [it["id"] for it in items if it.get("answer_letter") is None or it.get("cell") is None]
    if missing:
        raise SystemExit(f"{len(missing)} items lack answer_letter / cell (needed for the target), e.g. {missing[:3]}")
    random.Random(args.seed).shuffle(items)
    if args.limit:
        items = items[: args.limit]
    instances, stats = build_instances(items, args.seed, augment=not args.no_augment)
    frames = FrameSource(args.frame_cache, args.frames_caption, args.frames_skill, args.max_pixels, workers=1,
                         clip_frames=counts)
    ds = PtCDataset(instances, frames)
    steps_per_epoch = max(1, len(ds) // (world * args.grad_accum))
    total_steps = args.max_steps if args.max_steps and args.max_steps > 0 else \
        max(1, int(round(steps_per_epoch * args.epochs)))
    warmup_steps = max(1, int(round(args.warmup_ratio * total_steps)))
    if is_main:
        print(f"[data] {json.dumps(stats)}", flush=True)
        print(f"[sched] cosine | lr {args.lr} | total_steps {total_steps} | warmup_steps {warmup_steps} | "
              f"steps_per_epoch {steps_per_epoch} | world {world} | eff. batch {world * args.grad_accum}", flush=True)

    done_path = out / "final" / DONE
    if done_path.exists():
        done = json.loads(done_path.read_text())
        if done.get("max_steps", 0) >= total_steps:
            print(f"[resume] {done_path} exists ({done['global_step']} steps): training complete, nothing to do",
                  flush=True)
            return
        print(f"[resume] {done_path} covers {done['global_step']} steps; continuing to {total_steps}", flush=True)

    # --- model ----------------------------------------------------------------------------------------------------
    torch.cuda.set_device(local_rank)
    kernel_state = check_gdn_kernels(verbose=is_main)
    processor, model = load_base(args.model, args.attn, f"cuda:{local_rank}")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    model, dual, info = attach_for_training(model, args.lora_r, args.lora_alpha, args.lora_dropout,
                                            args.psvl_r, args.psvl_alpha, args.psvl_dropout, args.psvl_mode)
    if is_main:
        model.print_trainable_parameters()
        print(f"[lora] {json.dumps(info)}", flush=True)

    resume = latest_checkpoint(out)
    if resume is not None:
        payload = dual.load(resume)  # every rank: the Trainer restores only the language-model adapter
        print(f"[resume] rank {local_rank}: slot weights from {resume} (meta {payload['meta']})", flush=True)

    # --- callbacks ------------------------------------------------------------------------------------------------
    class SlotCheckpoint(TrainerCallback):
        def on_save(self, a, state, control, **kw):
            if a.should_save:  # on_save fires on every rank; one writer only
                ck = Path(a.output_dir) / f"checkpoint-{state.global_step}"
                dual.save(ck, {"global_step": state.global_step})
                print(f"[psvl] saved slot weights -> {ck / STATE_FILENAME}", flush=True)
            return control

    class StepLog(TrainerCallback):
        def __init__(self):
            self.t = time.time()

        def on_step_end(self, a, state, control, **kw):
            now = time.time()
            if is_main:
                print(f"[step] {state.global_step}/{state.max_steps} {now - self.t:.1f} s | max_alloc "
                      f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
            self.t = now
            return control

    targs = TrainingArguments(
        output_dir=str(out), num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=1, gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_steps=warmup_steps, weight_decay=0.0,
        bf16=True, logging_steps=10, save_strategy="steps", save_steps=args.save_steps,
        save_total_limit=args.save_total_limit, save_only_model=False, report_to="none",
        dataloader_num_workers=args.workers, dataloader_persistent_workers=args.workers > 0,
        remove_unused_columns=False, ddp_find_unused_parameters=False, gradient_checkpointing=False,
        seed=args.seed, data_seed=args.seed)
    trainer = Trainer(model=model, args=targs, train_dataset=ds, callbacks=[StepLog(), SlotCheckpoint()],
                      data_collator=Collator(processor, check_first=is_main))
    cfg = {**vars(args), "n_instances": len(ds), "world_size": world, "steps_per_epoch": steps_per_epoch,
           "total_steps": total_steps, "warmup_steps": warmup_steps, "kernels": kernel_state, "thinking": False,
           "instances": stats, **info, "resumed_from": str(resume) if resume else None}
    if is_main:
        out.mkdir(parents=True, exist_ok=True)
        name = "train_config.json" if resume is None else f"train_config.resume_{resume.name}.json"
        (out / name).write_text(json.dumps(cfg, indent=1))

    trainer.train(resume_from_checkpoint=str(resume) if resume else None)

    if is_main:
        final = out / "final"
        model.save_pretrained(final)
        dual.save(final, {"global_step": trainer.state.global_step})
        losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
        summary = {"global_step": trainer.state.global_step, "max_steps": trainer.state.max_steps,
                   "epoch": trainer.state.epoch, "losses": losses, "losses_finite": bool(np.all(np.isfinite(losses)))}
        if args.psvl_mode == "dual":
            div = slot_divergence(dual.state_dict())
            rel = [v["rel_diff"] for v in div.values()]
            ratio = [v["norm2"] / max(v["norm1"], 1e-30) for v in div.values()]
            summary["slot_divergence_rel_diff"] = {"min": min(rel), "mean": float(np.mean(rel)), "max": max(rel)}
            summary["slot_norm_ratio_2_over_1"] = {"min": min(ratio), "mean": float(np.mean(ratio)), "max": max(ratio)}
            summary["slot_cosine_mean"] = float(np.mean([v["cosine"] for v in div.values()]))
        (out / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=1))
        (final / "train_summary.json").write_text(json.dumps(summary, indent=1))
        print(f"[final] {json.dumps({k: v for k, v in summary.items() if k != 'losses'})}", flush=True)
        if summary["global_step"] >= summary["max_steps"] and summary["losses_finite"]:
            (final / DONE).write_text(json.dumps({"finished": time.strftime("%Y-%m-%d %H:%M:%S"),
                                                  "global_step": summary["global_step"],
                                                  "max_steps": summary["max_steps"]}))


if __name__ == "__main__":
    np.random.seed(0)
    main()
