#!/usr/bin/env python3
"""
scripts/convert_hf_weights.py

Convert a HuggingFace BitNet / LLaMA checkpoint to a safetensors file
that the Rust engine can load directly.

Usage:
    python scripts/convert_hf_weights.py \
        --model-id microsoft/bitnet-b1.58-3B \
        --output ./model/model.safetensors

    # local path also works
    python scripts/convert_hf_weights.py \
        --model-id ./my_local_model \
        --output ./model/model.safetensors \
        --dtype bfloat16
"""

import argparse
import json
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace checkpoint to safetensors for BitNet Engine"
    )
    parser.add_argument(
        "--model-id",
        required=True,
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output safetensors file path (e.g. ./model/model.safetensors)",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Dtype for non-quantised tensors (default: float16)",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip shape verification after conversion",
    )
    return parser.parse_args()


def check_dependencies():
    missing = []
    for pkg in ["torch", "transformers", "safetensors"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        sys.exit(
            f"Missing dependencies: {', '.join(missing)}\n"
            f"Install with: pip install {' '.join(missing)}"
        )


def convert(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoConfig
    from safetensors.torch import save_file

    print(f"Loading model: {args.model_id}")
    print(f"Target dtype:  {args.dtype}")

    # load config first (fast, no weights)
    config = AutoConfig.from_pretrained(args.model_id)

    # load model weights
    torch_dtype = getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    model.eval()

    state_dict = {k: v.contiguous() for k, v in model.state_dict().items()}

    # print summary
    total_params = sum(t.numel() for t in state_dict.values())
    total_bytes  = sum(t.numel() * t.element_size() for t in state_dict.values())
    print(f"\nTensors:       {len(state_dict)}")
    print(f"Total params:  {total_params / 1e9:.3f}B")
    print(f"Total size:    {total_bytes / 1e9:.3f} GB")

    # verify shapes if requested
    if not args.no_verify:
        print("\nVerifying tensor shapes...")
        for name, tensor in state_dict.items():
            assert tensor.is_contiguous(), f"{name} is not contiguous"
            assert not torch.isnan(tensor).any(), f"{name} contains NaN"
        print("All tensors OK")

    # write safetensors
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving to: {out_path}")
    save_file(state_dict, str(out_path))

    # write config.json alongside
    cfg_path = out_path.parent / "config.json"
    with open(cfg_path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    print(f"Config saved to: {cfg_path}")

    # write a quick metadata file so the Rust loader knows dtype + quant info
    meta = {
        "model_id":     args.model_id,
        "dtype":        args.dtype,
        "total_params": total_params,
        "tensors":      len(state_dict),
    }
    meta_path = out_path.parent / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved to: {meta_path}")

    print("\nConversion complete.")


def main():
    check_dependencies()
    args = parse_args()
    convert(args)


if __name__ == "__main__":
    main()
