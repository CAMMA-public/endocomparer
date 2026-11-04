"""Training dataset and collator for EndoComparer.

``PtCDataset`` turns PtC-CoT instances (``endocomparer.ptc.build_instances``) into samples: the two videos in the
instance's order (full / swap), their frames under the shared protocol, the displayed option order and the target
text. ``Collator`` renders the chat (thinking off), tokenises prompt + answer and masks everything up to and
including the empty think block, so the loss covers exactly ``Video 1: … Video 2: … (x)<|im_end|>``.
"""

from __future__ import annotations

import torch

from endodiffvqa.frames import FrameSource
from endodiffvqa.prompts import build_messages, video_order

NO_THINK = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


class PtCDataset(torch.utils.data.Dataset):
    def __init__(self, instances, frames: FrameSource):
        self.instances, self.frames = instances, frames

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, i):
        x = self.instances[i]
        it = x["item"]
        eles = self.frames.elements(it, video_order(it, x["control"]))
        videos = [self.frames.load(e) for e in eles]
        self.frames.clear()  # frames are read once per step; keep the worker's memory flat
        return {"item": it, "control": x["control"], "eles": eles, "videos": videos, "lmap": x["lmap"],
                "instruction": x["instruction"], "answer": x["answer"]}


class Collator:
    """Per-device batch 1 (variable video lengths); both videos must be present (the PSVL slots)."""

    def __init__(self, processor, check_first: bool = True, require_two_videos: bool = True):
        self.p = processor
        self.tok = processor.tokenizer
        self.think_end = self.tok.convert_tokens_to_ids("</think>")
        self.check = check_first
        self.two = require_two_videos

    def __call__(self, batch):
        assert len(batch) == 1, "per-device batch 1 (variable video lengths)"
        b = batch[0]
        msgs = build_messages(b["item"], b["control"], b["lmap"], b["instruction"], b["eles"])
        answer = b["answer"]
        msgs = msgs + [{"role": "assistant", "content": [{"type": "text", "text": answer}]}]
        text = self.p.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False, enable_thinking=False)
        assert text.endswith(f"{NO_THINK}{answer}<|im_end|>\n"), text[-120:]
        kw = {}
        if b["videos"]:
            kw = {"videos": [torch.from_numpy(a) for a, _ in b["videos"]],
                  "video_metadata": [m for _, m in b["videos"]], "do_sample_frames": False}
        enc = self.p(text=[text], return_tensors="pt", **kw)
        if self.two:
            assert enc["video_grid_thw"].shape[0] == 2, enc["video_grid_thw"]
        ids = enc["input_ids"][0]
        k = (ids == self.think_end).nonzero()[-1].item()
        start = k + 1
        while self.tok.decode(ids[start:start + 1]).strip() == "":  # the "\n\n" after </think>
            start += 1
        labels = ids.clone()
        labels[:start] = -100
        if self.check:
            self.check = False
            tgt = self.tok.decode(ids[start:])
            print(f"[collator] loss target {tgt!r}", flush=True)
            assert tgt.strip() == f"{answer}<|im_end|>", tgt
        enc["labels"] = labels[None]
        return dict(enc)
