"""Merge FSDP-saved LoRA into a full LTX-2 checkpoint (safetensors).

Designed for checkpoints produced by train_nft_ltx_v2_fsdp.py.
Uses v2 load_transformer + PeftModel.from_pretrained + merge_and_unload.

Usage:
    python scripts/merge_lora_fsdp.py \
        --checkpoint-path /path/to/base_model.safetensors \
        --lora-dir /path/to/checkpoints/checkpoint-XXX/lora \
        --output-path /path/to/merged.safetensors \
        --dtype bf16
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import safetensors
import torch
from peft import PeftModel
from safetensors.torch import save_file

_SCRIPT_DIR = Path(__file__).resolve().parent
_LTX_V2_ROOT = _SCRIPT_DIR.parent / "ltx_v2"
for _pkg in ("ltx-core", "ltx-pipelines", "ltx-trainer"):
    _src = str(_LTX_V2_ROOT / _pkg / "src")
    if _src not in sys.path:
        sys.path.insert(0, _src)

from ltx_trainer.model_loader import load_transformer


def parse_args():
    p = argparse.ArgumentParser("Merge LoRA (FSDP-trained) into full LTX-2 checkpoint")
    p.add_argument("--checkpoint-path", type=str, required=True,
                   help="Base full checkpoint (.safetensors) used at training time")
    p.add_argument("--lora-dir", type=str, required=True,
                   help="PEFT LoRA dir (adapter_model.safetensors + adapter_config.json)")
    p.add_argument("--output-path", type=str, required=True,
                   help="Output merged .safetensors path")
    p.add_argument("--dtype", type=str, default="bf16",
                   choices=["keep", "fp16", "bf16", "fp32"])
    p.add_argument("--strict", action="store_true",
                   help="Fail if any merged tensor cannot be mapped back to full_state")
    return p.parse_args()


_DTYPE_MAP = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def load_full_safetensors(path: str):
    state, meta = {}, {}
    with safetensors.safe_open(path, framework="pt", device="cpu") as f:
        md = f.metadata()
        if md is not None:
            meta = dict(md)
        for k in f.keys():
            state[k] = f.get_tensor(k)
    return state, meta


@torch.no_grad()
def main():
    args = parse_args()
    target_dtype = _DTYPE_MAP.get(args.dtype)

    print(f"[1/4] Loading full checkpoint: {args.checkpoint_path}")
    full_state, metadata = load_full_safetensors(args.checkpoint_path)

    print(f"[2/4] Loading base transformer via load_transformer ...")
    base_transformer = load_transformer(args.checkpoint_path, device="cpu", dtype=torch.float32)
    base_transformer.eval()

    print(f"[3/4] Loading LoRA from {args.lora_dir} and merging ...")
    peft_model = PeftModel.from_pretrained(base_transformer, args.lora_dir, is_trainable=False)
    lora_count = sum(1 for n, _ in peft_model.named_parameters() if "lora_" in n)
    print(f"    {lora_count} LoRA parameter tensors attached")
    if lora_count == 0:
        raise RuntimeError("No LoRA parameters attached. Key mismatch between adapter and base transformer.")

    merged_model = peft_model.merge_and_unload()
    merged_sd = merged_model.state_dict()

    PREFIX = "model.diffusion_model."
    replaced, not_found, shape_mismatch = 0, [], []

    for bare_key, v in merged_sd.items():
        if not torch.is_tensor(v):
            continue
        target_k = PREFIX + bare_key
        if target_k not in full_state:
            not_found.append((bare_key, target_k))
            continue
        if full_state[target_k].shape != v.shape:
            shape_mismatch.append((bare_key, target_k, tuple(v.shape), tuple(full_state[target_k].shape)))
            continue
        out_v = v.detach().cpu().contiguous()
        out_v = out_v.to(target_dtype) if target_dtype is not None else out_v.to(full_state[target_k].dtype)
        full_state[target_k] = out_v
        replaced += 1

    print(f"    Replaced {replaced} tensors in full checkpoint")
    if not_found:
        print(f"    Not found: {len(not_found)}, first 10:")
        for a, b in not_found[:10]:
            print(f"      {a} -> {b}")
    if shape_mismatch:
        print(f"    Shape mismatch: {len(shape_mismatch)}, first 10:")
        for a, b, s1, s2 in shape_mismatch[:10]:
            print(f"      {a} -> {b}  {s1} vs {s2}")
    if args.strict and (not_found or shape_mismatch):
        raise RuntimeError("Strict mode: some weights could not be mapped back.")
    if replaced == 0:
        raise RuntimeError("0 tensors replaced in full checkpoint. Aborting.")

    print(f"[4/4] Saving to {args.output_path}")
    metadata = {str(k): str(v) for k, v in metadata.items()}
    metadata.update({
        "lora_merged_fsdp": "true",
        "source_checkpoint": str(args.checkpoint_path),
        "source_lora_dir": str(args.lora_dir),
        "dtype_policy": args.dtype,
        "replaced_tensors": str(replaced),
    })
    out = Path(args.output_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(full_state, str(out), metadata=metadata)
    print(f"[OK] {out}  ({len(full_state)} tensors, {replaced} replaced)")


if __name__ == "__main__":
    main()
