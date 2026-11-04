"""Pseudo-Siamese Visual LoRA (PSVL): one low-rank adapter per video slot on the Qwen3.5 vision tower and merger.

Video 1 and Video 2 of an item go through the vision tower as ONE packed ``(seq_len, hidden)`` tensor in
``grid_thw`` row order (all of Video 1's patch tokens, then all of Video 2's). Attention runs inside per-frame
``cu_seqlens`` segments, so no block ever mixes the two videos, and the merger is two Linears on
``(seq_len / 4, 4 * hidden)`` in the same order. A LoRA whose weights are chosen per token ROW (slot 1's on the first
``n1`` rows, slot 2's on the rest) is therefore the same function as running the tower twice with two adapters —
two branches that share the frozen backbone but not the adaptation, in a single encoder pass.

PEFT cannot express this (it sums every active adapter over all tokens), hence this module:

- ``SlotLoRALinear``   wraps one frozen ``nn.Linear``: ``base(x) + (alpha / r) * B_k(A_k(dropout(x)))`` on slot k's
                       rows — PEFT's vanilla-LoRA forward (same dropout placement, scaling and init), routed per row.
- ``SlotContext``      the current item's row boundaries, ``cumsum(grid_thw.prod(-1))``, and the ``slot_map``.
- ``attach_psvl``      replaces the 108 ViT-block + 2 merger Linears of Qwen3.5-9B and reassigns the inner model's
                       class so that ``get_image_features`` (which ``get_video_features`` delegates to) sets the
                       context before the tower runs. The context is never cleared: non-reentrant gradient
                       checkpointing re-runs the blocks during backward and must still see that item's boundaries.
- state lives in ONE file, ``dual_vision_lora.pt`` (never an adapter-shaped subdirectory: the Hugging Face Trainer
  would load it as the language-model adapter on resume), with a readable ``psvl_config.json`` next to it.
  Parameter names are ``slot_A`` / ``slot_B`` — deliberately without ``lora_`` so PEFT's ``save_pretrained`` never
  copies them into the language-model adapter file.

Attach AFTER ``get_peft_model`` (its freeze pass would freeze the slot weights), on a host whose language-model LoRA
regex matches ``language_model`` only.

Slot maps (inference diagnostics; training always uses ``12``): ``12`` slot k on Video k, ``21`` crossed, ``11`` /
``22`` one slot's weights on both videos.
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any

import torch
from torch import nn

STATE_FILENAME = "dual_vision_lora.pt"
CONFIG_FILENAME = "psvl_config.json"
STATE_FORMAT = 1
SLOTS = ("slot1", "slot2")
MODES = ("dual", "shared")
SLOT_MAPS = ("12", "21", "11", "22")
# Every Linear of the 27 ViT blocks + both merger projections. patch_embed.proj (a Conv3d run as a matmul) and
# pos_embed (an Embedding) stay frozen.
VISION_TARGET_RE = re.compile(r"(?:^|\.)visual\.(blocks\.\d+\.(attn\.(qkv|proj)|mlp\.(linear_fc1|linear_fc2))"
                              r"|merger\.(linear_fc1|linear_fc2))$")
EXPECTED_BLOCK_MODULES = 27 * 4
EXPECTED_MERGER_MODULES = 2


class SlotContext:
    """Row boundaries of the item currently in the tower, in patch tokens, and which slot each row uses."""

    def __init__(self, merge_unit: int = 4):
        self.merge_unit = merge_unit
        self.bounds: list[int] | None = None  # [0, n1, n1 + n2]
        self.slot_map = "12"

    def set_slot_map(self, slot_map: str) -> None:
        if slot_map not in SLOT_MAPS:
            raise ValueError(f"slot_map {slot_map!r} not in {SLOT_MAPS}")
        self.slot_map = slot_map

    def set(self, grid_thw: torch.Tensor) -> None:
        tokens = [int(t) for t in grid_thw.prod(-1).tolist()]
        if not 1 <= len(tokens) <= len(SLOTS):
            raise ValueError(f"PSVL handles 1 or 2 grid_thw rows (videos), got {len(tokens)}")
        if any(t % self.merge_unit for t in tokens):
            raise ValueError(f"row token counts {tokens} not divisible by the merge unit {self.merge_unit}")
        bounds = [0]
        for t in tokens:
            bounds.append(bounds[-1] + t)
        self.bounds = bounds

    def segments(self, n_rows: int, divisor: int) -> list[tuple[int, int, str]]:
        """``[(start, end, slot)]`` for a layer input of ``n_rows`` rows; ``divisor`` is 1 in the blocks and the
        merge unit in the merger. Adjacent rows on the same slot are fused (slot map 11 / 22: one segment)."""
        if self.bounds is None:
            raise RuntimeError("SlotContext is unset: the tower ran without going through get_image_features")
        if n_rows * divisor != self.bounds[-1]:
            raise RuntimeError(f"layer input has {n_rows} rows (x{divisor}) but the context holds "
                               f"{self.bounds[-1]} patch tokens: stale or foreign context")
        segs: list[tuple[int, int, str]] = []
        for i in range(len(self.bounds) - 1):
            slot = SLOTS[int(self.slot_map[i]) - 1]
            a, b = self.bounds[i] // divisor, self.bounds[i + 1] // divisor
            if segs and segs[-1][2] == slot:
                segs[-1] = (segs[-1][0], b, slot)
            else:
                segs.append((a, b, slot))
        return segs


class SlotLoRALinear(nn.Module):
    """A frozen ``nn.Linear`` plus one LoRA per slot, applied to that slot's rows of a 2-D ``(rows, in)`` input."""

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float, mode: str, ctx: SlotContext,
                 divisor: int):
        super().__init__()
        assert isinstance(base, nn.Linear), type(base)
        assert mode in MODES, mode
        self.base_layer = base
        self.r, self.alpha, self.scaling, self.mode, self.divisor = r, alpha, alpha / r, mode, divisor
        self.ctx = ctx  # plain object shared by every layer of one host; not a Module, not in the state dict
        slots = SLOTS if mode == "dual" else SLOTS[:1]
        self.slot_A = nn.ModuleDict({s: nn.Linear(base.in_features, r, bias=False) for s in slots})
        self.slot_B = nn.ModuleDict({s: nn.Linear(r, base.out_features, bias=False) for s in slots})
        self.slot_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # PEFT's default init: A kaiming-uniform (as nn.Linear), B zero -> the model equals the base model at step 0.
        # Slot 2's A is a COPY of slot 1's: both branches start from the backbone and differ only through the
        # gradients they receive.
        nn.init.kaiming_uniform_(self.slot_A[SLOTS[0]].weight, a=math.sqrt(5))
        for s in slots:
            nn.init.zeros_(self.slot_B[s].weight)
            if s != SLOTS[0]:
                with torch.no_grad():
                    self.slot_A[s].weight.copy_(self.slot_A[SLOTS[0]].weight)
        # As PEFT (autocast_adapter_dtype=True): on a bf16 / fp16 base the LoRA weights are float32 master weights
        # — lr-1e-4 AdamW updates would be rounded away in bf16 — and the forward casts x up, adds in float32 and
        # casts the sum back to the base dtype.
        w = base.weight
        dtype = torch.float32 if w.dtype in (torch.bfloat16, torch.float16) else w.dtype
        self.slot_A.to(device=w.device, dtype=dtype)
        self.slot_B.to(device=w.device, dtype=dtype)

    @property
    def weight(self) -> torch.Tensor:  # so `module.weight.dtype/device` probes keep working
        return self.base_layer.weight

    @property
    def bias(self):
        return self.base_layer.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base_layer(x)
        a0 = self.slot_A[SLOTS[0]].weight
        xd = self.slot_dropout(x if x.dtype == a0.dtype else x.to(a0.dtype))
        if self.mode == "shared":
            delta = self.slot_B[SLOTS[0]](self.slot_A[SLOTS[0]](xd))
        else:
            if x.dim() != 2:
                raise RuntimeError(f"slot routing needs a packed (rows, features) input, got {tuple(x.shape)}")
            segs = self.ctx.segments(x.shape[0], self.divisor)
            parts = [self.slot_B[s](self.slot_A[s](xd[a:b] if len(segs) > 1 else xd)) for a, b, s in segs]
            delta = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
            if self.training:
                # An item that uses one slot only (slot map 11 / 22, or a single video) leaves the other slot out
                # of the graph, and DDP with find_unused_parameters=False then stalls. An exact-zero term keeps it
                # in, with a zero gradient and no numerical change.
                unused = [s for s in self.slot_A if all(s != t for _, _, t in segs)]
                for s in unused:
                    delta = delta + 0.0 * (self.slot_A[s].weight.sum() + self.slot_B[s].weight.sum())
        return (result + delta * self.scaling).to(result.dtype)

    def delta_weight(self, slot: str) -> torch.Tensor:
        """``(alpha / r) * B @ A`` in float32."""
        return self.scaling * (self.slot_B[slot].weight.float() @ self.slot_A[slot].weight.float())


_SLOT_MODEL_CLASS_CACHE: dict[type, type] = {}


def _make_slot_model_class(base_cls: type) -> type:
    if base_cls in _SLOT_MODEL_CLASS_CACHE:
        return _SLOT_MODEL_CLASS_CACHE[base_cls]

    def get_image_features(self, pixel_values, image_grid_thw=None, **kwargs):  # noqa: ANN001, ANN202
        self._slot_ctx.set(image_grid_thw)  # never cleared: checkpointed blocks re-run during backward
        return base_cls.get_image_features(self, pixel_values, image_grid_thw, **kwargs)

    cls = type(f"{base_cls.__name__}SlotLoRA", (base_cls,), {"get_image_features": get_image_features})
    _SLOT_MODEL_CLASS_CACHE[base_cls] = cls
    return cls


class DualVisionLoRA:
    """Handle on an attached host: the wrapped layers (in attach order), the context, save / load."""

    def __init__(self, layers: dict[str, SlotLoRALinear], ctx: SlotContext, config: dict):
        self.layers, self.ctx, self.config = layers, ctx, config

    def module_names(self) -> list[str]:
        return list(self.layers)

    def parameters(self):
        for layer in self.layers.values():
            yield from layer.slot_A.parameters()
            yield from layer.slot_B.parameters()

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def set_slot_map(self, slot_map: str) -> None:
        if self.config["mode"] != "dual" and slot_map != "12":
            raise ValueError("slot_map only applies to mode 'dual'")
        self.ctx.set_slot_map(slot_map)

    def state_dict(self) -> dict[str, torch.Tensor]:
        sd = {}
        for name, layer in self.layers.items():
            for s in layer.slot_A:
                sd[f"{name}.slot_A.{s}.weight"] = layer.slot_A[s].weight.detach().cpu().clone()
                sd[f"{name}.slot_B.{s}.weight"] = layer.slot_B[s].weight.detach().cpu().clone()
        return sd

    def state_dict_keys(self) -> list[str]:
        return [f"{name}.{kind}.{s}.weight" for name, layer in self.layers.items() for s in layer.slot_A
                for kind in ("slot_A", "slot_B")]

    def load_state_dict(self, sd: dict[str, torch.Tensor]) -> None:
        want = set(self.state_dict_keys())
        if set(sd) != want:
            missing, extra = sorted(want - set(sd)), sorted(set(sd) - want)
            raise RuntimeError(f"PSVL state mismatch: missing {missing[:3]} (+{len(missing)}), "
                               f"unexpected {extra[:3]} (+{len(extra)})")
        with torch.no_grad():
            for name, layer in self.layers.items():
                for s in layer.slot_A:
                    for kind, mod in (("slot_A", layer.slot_A[s]), ("slot_B", layer.slot_B[s])):
                        t = sd[f"{name}.{kind}.{s}.weight"]
                        if t.shape != mod.weight.shape:
                            raise RuntimeError(f"{name}.{kind}.{s}: shape {tuple(t.shape)} != {tuple(mod.weight.shape)}")
                        mod.weight.copy_(t.to(device=mod.weight.device, dtype=mod.weight.dtype))

    def save(self, directory: str | Path, meta: dict | None = None) -> Path:
        """Write ``dual_vision_lora.pt`` (atomically) and ``psvl_config.json`` into ``directory``."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / STATE_FILENAME
        tmp = directory / f".{STATE_FILENAME}.tmp{os.getpid()}"
        torch.save({"format": STATE_FORMAT, "config": self.config, "state": self.state_dict(), "meta": meta or {}},
                   tmp)
        os.replace(tmp, path)  # atomic: a killed job never leaves a truncated file a resume would load
        write_config(directory, self.config, meta)
        return path

    def load(self, directory: str | Path) -> dict:
        payload = read_state(directory)
        for k in ("mode", "r", "alpha", "modules"):
            if payload["config"][k] != self.config[k]:
                raise RuntimeError(f"{STATE_FILENAME} config {k} differs from the attached model's "
                                   f"({str(payload['config'][k])[:80]} != {str(self.config[k])[:80]})")
        self.load_state_dict(payload["state"])
        return payload


def write_config(directory: str | Path, config: dict, meta: dict | None = None) -> Path:
    """A readable copy of the PSVL configuration, so nobody has to unpickle the state file to see what it holds."""
    path = Path(directory) / CONFIG_FILENAME
    path.write_text(json.dumps({"format": STATE_FORMAT, "state_file": STATE_FILENAME, **config,
                                "meta": meta or {}}, indent=1))
    return path


def read_state(directory: str | Path) -> dict:
    path = Path(directory) / STATE_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing: refusing to run a PSVL model without its slot weights "
                                "(it would silently be the base vision tower)")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != STATE_FORMAT:
        raise RuntimeError(f"{path}: unknown format {payload.get('format')}")
    return payload


def attach_psvl(host: Any, r: int = 32, alpha: float = 64, dropout: float = 0.05, mode: str = "dual",
                expect_counts: bool = True) -> DualVisionLoRA:
    """Replace the vision-tower / merger Linears of ``host`` (the bare ``Qwen3_5ForConditionalGeneration``, i.e.
    ``peft_model.get_base_model()``) with ``SlotLoRALinear`` and install the context hook on ``host.model``.
    Modules are visited in ``named_modules()`` order, so the parameter order — which optimizer-state restore on
    resume depends on — is the same on every attach."""
    if mode not in MODES:
        raise ValueError(f"mode {mode!r} not in {MODES}")
    inner = host.model
    if hasattr(inner, "_slot_ctx"):
        raise RuntimeError("PSVL is already attached to this host")
    merge_unit = int(inner.visual.spatial_merge_size) ** 2
    ctx = SlotContext(merge_unit)
    targets = [(n, m) for n, m in host.named_modules() if VISION_TARGET_RE.search(n)]
    layers: dict[str, SlotLoRALinear] = {}
    for name, mod in targets:
        if not isinstance(mod, nn.Linear):
            raise RuntimeError(f"{name} is {type(mod).__name__}, not nn.Linear: already wrapped (the PEFT regex "
                               "must match language_model only) or an unexpected architecture")
        parent = host.get_submodule(name.rsplit(".", 1)[0])
        layer = SlotLoRALinear(mod, r, alpha, dropout, mode, ctx, merge_unit if ".merger." in name else 1)
        setattr(parent, name.rsplit(".", 1)[1], layer)
        layers[name] = layer
    n_blk = sum(".blocks." in n for n in layers)
    n_mrg = sum(".merger." in n for n in layers)
    if expect_counts and (n_blk, n_mrg) != (EXPECTED_BLOCK_MODULES, EXPECTED_MERGER_MODULES):
        raise RuntimeError(f"expected {EXPECTED_BLOCK_MODULES} block + {EXPECTED_MERGER_MODULES} merger Linears, "
                           f"got {n_blk} + {n_mrg}")
    if any("patch_embed" in n or "pos_embed" in n for n in layers):
        raise RuntimeError("patch_embed / pos_embed must stay frozen")
    inner._slot_ctx = ctx
    inner.__class__ = _make_slot_model_class(type(inner))
    config = {"mode": mode, "r": r, "alpha": alpha, "dropout": dropout, "slots": list(SLOTS if mode == "dual"
              else SLOTS[:1]), "modules": list(layers), "n_block_modules": n_blk, "n_merger_modules": n_mrg,
              "init": "slot2.A = copy of slot1.A; B = 0", "merge_unit": merge_unit}
    return DualVisionLoRA(layers, ctx, config)


def slot_divergence(state: dict[str, torch.Tensor], scaling: float = 1.0) -> dict[str, dict[str, float]]:
    """Per module: ``||dW1 - dW2||_F / ||dW1||_F`` with ``dW = B @ A`` (float32), plus the two norms and cosine."""
    out = {}
    for name in sorted({k.split(".slot_")[0] for k in state}):
        d = [scaling * (state[f"{name}.slot_B.{s}.weight"].float() @ state[f"{name}.slot_A.{s}.weight"].float())
             for s in SLOTS]
        n1, n2 = d[0].norm().item(), d[1].norm().item()
        out[name] = {"rel_diff": (d[0] - d[1]).norm().item() / max(n1, 1e-30), "norm1": n1, "norm2": n2,
                     "cosine": (d[0] * d[1]).sum().item() / max(n1 * n2, 1e-30)}
    return out
