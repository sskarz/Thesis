#!/usr/bin/env python3
"""Convert an mlx-lm LoRA adapter to a llama.cpp GGUF LoRA so it can be applied at RUNTIME on the
exact bar GGUF (bartowski Q4_K_M): no fuse, no re-quantization, base weights bit-identical to Run 69.

mlx:  <prefix>.lora_a [in, r], <prefix>.lora_b [r, out], output = base(x) + scale * (x @ a) @ b
PEFT: lora_A.weight [r, in] = a.T, lora_B.weight [out, r] = b.T, scaling = lora_alpha / r  => alpha = scale * r

usage: mlx_adapter_to_gguf_lora.py <adapter_dir> <out.gguf>
"""
import glob
import json
import pathlib
import subprocess
import sys
import tempfile

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

ROOT = pathlib.Path("/Users/USER/Documents/GitHub/Thesis")
HF = glob.glob(str(pathlib.Path.home() / ".cache/huggingface/hub/models--google--gemma-4-e2b-it/snapshots/*"))[0]


def main(adapter_dir: str, out: str) -> None:
    cfg = json.load(open(f"{adapter_dir}/adapter_config.json"))["lora_parameters"]
    r, scale = int(cfg["rank"]), float(cfg["scale"])
    tensors = {}
    with safe_open(f"{adapter_dir}/adapters.safetensors", "np") as f:
        for k in f.keys():
            assert k.startswith("language_model.model."), k
            base = "base_model.model.model." + k[len("language_model.model."):]
            if k.endswith(".lora_a"):
                tensors[base[: -len(".lora_a")] + ".lora_A.weight"] = np.ascontiguousarray(f.get_tensor(k).T.astype(np.float32))
            elif k.endswith(".lora_b"):
                tensors[base[: -len(".lora_b")] + ".lora_B.weight"] = np.ascontiguousarray(f.get_tensor(k).T.astype(np.float32))
            else:
                raise SystemExit(f"unexpected key {k}")
    targets = sorted({k.split(".")[-3] for k in tensors})
    with tempfile.TemporaryDirectory() as d:
        save_file(tensors, f"{d}/adapter_model.safetensors")
        json.dump({"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": r, "lora_alpha": scale * r,
                   "lora_dropout": 0.0, "bias": "none", "target_modules": targets, "use_rslora": False,
                   "base_model_name_or_path": HF, "inference_mode": True}, open(f"{d}/adapter_config.json", "w"))
        subprocess.run([sys.executable, str(ROOT / "llama.cpp/convert_lora_to_gguf.py"), "--base", HF,
                        "--outfile", out, "--outtype", "f16", d], check=True)
    print("wrote", out, "rank", r, "scale", scale, "tensors", len(tensors), "targets", targets)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
