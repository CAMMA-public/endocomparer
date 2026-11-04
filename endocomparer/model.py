"""Building and loading EndoComparer: Qwen3.5-9B + language-model LoRA (PEFT) + Pseudo-Siamese Visual LoRA.

Two details of the backbone matter for anyone who changes this file:

* **Conv3d patch embedding as a matmul.** ``visual.patch_embed.proj`` is a Conv3d whose kernel equals its stride, so
  it is exactly one linear map per patch. bf16 Conv3d through cuDNN on torch 2.9 is pathologically slow (a training
  rank can sit in that layer for minutes while the others wait in NCCL); ``patch_embed_as_linear`` replaces its
  forward with the equivalent ``F.linear``. Numerically identical; applied at training and inference alike.
* **Gated-DeltaNet kernels.** Three of every four Qwen3.5 layers are linear-attention (Gated DeltaNet) layers whose
  forward and backward run on the ``flash-linear-attention`` Triton kernels. ``check_gdn_kernels`` fails fast on two
  states that would otherwise surface only deep in a long run: triton in [3.4.0, 3.7.1) on Hopper with TileLang off
  (fla refuses the backward: wrong gradients, issue #640), and TileLang on without a usable nvcc (a JIT failure in
  the backward with an unrelated-looking error). The pinned ``requirements.txt`` (triton 3.7.1) passes.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import torch

from .psvl import attach_psvl, read_state

DEFAULT_BASE = "Qwen/Qwen3.5-9B"
DEFAULT_WEIGHTS = "ethanshili/EndoComparer"
NO_THINK = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
# Attention + MLP projections of the full-attention layers and the Gated-DeltaNet projections (in_proj_qkv,
# in_proj_z, out_proj) of the linear-attention layers: 200 modules on Qwen3.5-9B. The per-head gates in_proj_a /
# in_proj_b (32-dim) stay frozen. The regex matches language_model only, so PEFT never touches a vision Linear.
LM_LORA_TARGETS = (r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj"
                   r"|in_proj_qkv|in_proj_z|out_proj)$")
WEIGHT_FILES = ("adapter_config.json", "adapter_model.safetensors", "dual_vision_lora.pt", "psvl_config.json",
                "train_summary.json", "README.md")


def patch_embed_as_linear(model) -> None:
    """Run the vision patch embedding (Conv3d, kernel == stride) as the equivalent matmul."""
    import torch.nn.functional as F

    def forward(self, hidden_states):
        w = self.proj.weight.view(self.embed_dim, -1)
        return F.linear(hidden_states.reshape(-1, w.shape[1]).to(w.dtype), w, self.proj.bias)

    n = 0
    for m in model.modules():
        if type(m).__name__ == "Qwen3_5VisionPatchEmbed":
            assert list(m.proj.kernel_size) == list(m.proj.stride), (m.proj.kernel_size, m.proj.stride)
            m.forward = forward.__get__(m)
            n += 1
    assert n == 1, n


def check_gdn_kernels(verbose: bool = True) -> dict:
    """Fail fast if the Gated-DeltaNet backward would be wrong or unavailable (see the module docstring)."""
    import triton
    from packaging import version as _v
    import fla
    from fla.utils import IS_NVIDIA_HOPPER, has_usable_nvcc
    from fla.ops.common.backends.tilelang import TileLangBackend

    tl_enabled, tl_available = TileLangBackend.is_enabled(), TileLangBackend.is_available()
    tri = triton.__version__
    bad_triton = _v.parse("3.4.0") <= _v.parse(tri) < _v.parse("3.7.1")
    state = {"triton": tri, "fla": fla.__version__, "hopper": bool(IS_NVIDIA_HOPPER),
             "tilelang_enabled": bool(tl_enabled), "tilelang_available": bool(tl_available),
             "usable_nvcc": bool(has_usable_nvcc()),
             "gdn_backward": "tilelang" if (tl_enabled and tl_available) else "triton"}
    if verbose:
        print(f"[kernels] {state}", flush=True)
    if IS_NVIDIA_HOPPER and bad_triton and not (tl_enabled and tl_available):
        raise SystemExit(f"triton {tri} is in [3.4.0, 3.7.1) on a Hopper GPU and TileLang is not usable: "
                         "flash-linear-attention refuses the gated DeltaNet backward (issue #640, incorrect "
                         "gradients). Install triton>=3.7.1 (requirements.txt) or provide a usable nvcc for TileLang.")
    if tl_enabled and not tl_available:
        raise SystemExit("FLA_TILELANG is on but the TileLang backend is not available (usually no usable nvcc). "
                         "Set FLA_TILELANG=0 and use triton>=3.7.1.")
    return state


def load_base(model_path: str = DEFAULT_BASE, attn: str = "flash_attention_2", device: Optional[str] = None):
    """Processor + ``Qwen3_5ForConditionalGeneration`` in bf16 with the patch-embed fix, on ``device``."""
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    processor = AutoProcessor.from_pretrained(model_path)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(model_path, dtype=torch.bfloat16, attn_implementation=attn)
    patch_embed_as_linear(model)
    if device:
        model.to(device)
    return processor, model


def resolve_weights(weights: str = DEFAULT_WEIGHTS) -> Path:
    """A local directory holding the adapter + PSVL files, or a Hugging Face model id to download."""
    p = Path(weights)
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=weights, allow_patterns=list(WEIGHT_FILES)))


def attach_for_training(model, lora_r: int = 32, lora_alpha: int = 64, lora_dropout: float = 0.05,
                        psvl_r: int = 32, psvl_alpha: int = 64, psvl_dropout: float = 0.05, psvl_mode: str = "dual"):
    """Freeze the backbone, wrap the language model with a PEFT LoRA, attach PSVL. Returns (peft_model, dual, info).

    Order matters: PSVL is attached AFTER ``get_peft_model`` (PEFT's freeze pass would freeze the slot weights)."""
    from peft import LoraConfig, get_peft_model

    for p in model.parameters():
        p.requires_grad_(False)
    model = get_peft_model(model, LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                                             bias="none", task_type="CAUSAL_LM", target_modules=LM_LORA_TARGETS))
    lm_names = [n for n, p in model.named_parameters() if p.requires_grad]
    lm_modules = sorted({n.split(".lora_")[0] for n in lm_names})
    assert lm_names and all("language_model" in n for n in lm_names), [n for n in lm_names if "language_model" not in n][:3]
    lm_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    dual = attach_psvl(model.get_base_model(), r=psvl_r, alpha=psvl_alpha, dropout=psvl_dropout, mode=psvl_mode)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total == lm_trainable + dual.num_parameters(), (total, lm_trainable, dual.num_parameters())
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    bad = [n for n in trainable if "lora_" not in n and ".slot_" not in n]
    assert not bad, f"base weights marked trainable: {bad[:3]}"
    assert not any(".visual." in n and "lora_" in n for n in trainable), "PEFT wrapped a vision module"
    info = {"lm_lora_modules": len(lm_modules), "lm_lora_tensors": len(lm_names), "lm_lora_params": lm_trainable,
            "lm_lora_targets": LM_LORA_TARGETS, "psvl_modules": len(dual.layers), "psvl_slots": dual.config["slots"],
            "psvl_params": dual.num_parameters(), "psvl_init": dual.config["init"], "trainable_params": total,
            "host_inner_class": type(model.get_base_model().model).__name__}
    return model, dual, info


def load_endocomparer(model_path: str = DEFAULT_BASE, weights: str = DEFAULT_WEIGHTS, slot_map: str = "12",
                      attn: str = "flash_attention_2", device: str = "cuda"):
    """Inference model: base + language-model LoRA + PSVL weights, eval mode. Returns (processor, model, info)."""
    from peft import PeftModel

    t0 = time.time()
    wdir = resolve_weights(weights)
    payload = read_state(wdir)  # refuses a missing slot file before anything is loaded
    processor, model = load_base(model_path, attn, device)
    model = PeftModel.from_pretrained(model, str(wdir))
    n_lora = sum(1 for k, _ in model.named_parameters() if "lora_" in k)
    dcfg = payload["config"]
    dual = attach_psvl(model.get_base_model(), r=dcfg["r"], alpha=dcfg["alpha"], dropout=0.0, mode=dcfg["mode"])
    dual.load(wdir)
    dual.set_slot_map(slot_map)
    model.eval()
    info = {"base": model_path, "weights": str(wdir), "lm_lora_tensors": n_lora, "psvl_modules": len(dual.layers),
            "psvl_tensors": len(payload["state"]), "psvl_r": dcfg["r"], "psvl_alpha": dcfg["alpha"],
            "psvl_meta": payload.get("meta"), "slot_map": slot_map, "dtype": "bfloat16", "attn": attn,
            "patch_embed": "linear", "load_seconds": round(time.time() - t0, 1)}
    print(f"[model] {json.dumps(info)}", flush=True)
    return processor, model, info
