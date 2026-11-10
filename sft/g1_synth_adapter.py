#!/usr/bin/env python3
"""CPU-only: build a RANDOM (untrained) wide adapter in mlx_lm format with the exact module set and shapes G1 trains, to test conversion and the GGUF audit before any training.
usage: g1_synth_adapter.py OUT_DIR [--layers 8-34] [--rank 16]   Shapes come from the safetensors headers of the MLX base (no weights are loaded)."""
import argparse, json, pathlib, numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file
MODEL = pathlib.Path("/Users/USER/Documents/GitHub/Thesis/sft/models/gemma-4-e2b-it-mlx")
ATTN = ["q_proj", "o_proj", "k_proj", "v_proj"]; MLP = ["gate_proj", "up_proj", "down_proj"]
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--layers", default="8-34"); ap.add_argument("--rank", type=int, default=16); a = ap.parse_args()
idx = json.load(open(MODEL / "model.safetensors.index.json"))["weight_map"]
lo, hi = (int(x) for x in a.layers.split("-"))
rng = np.random.default_rng(0); t = {}; handles = {}
for i in range(lo, hi + 1):
    for grp, names in (("self_attn", ATTN), ("mlp", MLP)):
        for n in names:
            key = f"language_model.model.layers.{i}.{grp}.{n}.weight"
            if key not in idx:
                continue
            fn = idx[key]
            if fn not in handles:
                handles[fn] = safe_open(str(MODEL / fn), "np")
            out_f, in_f = handles[fn].get_slice(key).get_shape()
            p = key[: -len(".weight")]
            t[p + ".lora_a"] = (rng.standard_normal((in_f, a.rank)) * 0.02).astype(np.float32)
            t[p + ".lora_b"] = (rng.standard_normal((a.rank, out_f)) * 0.02).astype(np.float32)
out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
save_file(t, str(out / "adapters.safetensors"))
json.dump({"fine_tune_type": "lora", "num_layers": hi - lo + 1, "lora_parameters": {"rank": a.rank, "scale": 2.0, "dropout": 0.0, "keys": sorted(set(ATTN + MLP))}}, open(out / "adapter_config.json", "w"))
print("modules", len(t) // 2, "tensors", len(t))
