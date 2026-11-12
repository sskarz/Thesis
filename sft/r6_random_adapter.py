#!/usr/bin/env python3
"""Round 6 D1: build a spectrum-preserving RANDOM-DIRECTION LoRA adapter from a trained mlx-lm adapter (no training).

For every LoRA pair (<prefix>.lora_a [in, r], <prefix>.lora_b [r, out]) in the source adapter, the weight update is
P = A @ B (times the unchanged scale hyper-parameter). We take the top-r singular values S of P and rebuild
    A' = Qu * sqrt(S),   B' = (Qv * sqrt(S))^T
where Qu [in, r] and Qv [out, r] are random orthonormal matrices (QR of Gaussian draws, fixed seed). Then
P' = A' @ B' has the same singular values (up to float32 rounding, checked) and so the same Frobenius norm and spectrum as P, at the same layers and
modules, with random input/output directions, i.e. no learned content. The factors are balanced (||A'|| = ||B'||).
Keys, shapes and dtype (float32) are identical to the source.

usage: r6_random_adapter.py <src_adapter_dir> <out_dir> [--seed 6061]
Writes <out_dir>/adapters.safetensors, a copy of adapter_config.json, and <out_dir>/random_adapter_report.json
(per-pair product norms, factor norms, and the product-norm error after a float16 round trip, since the GGUF LoRA is fp16).
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file


def _orthonormal(rng: np.random.Generator, n: int, r: int) -> np.ndarray:
    q, _ = np.linalg.qr(rng.standard_normal((n, r)))
    return q


def build_random(src_tensors: dict[str, np.ndarray], seed: int) -> tuple[dict[str, np.ndarray], list[dict]]:
    rng = np.random.default_rng(seed)
    prefixes = sorted({k[: -len(".lora_a")] for k in src_tensors if k.endswith(".lora_a")})
    for p in prefixes:
        if p + ".lora_b" not in src_tensors:
            raise ValueError(f"missing lora_b for {p}")
    extra = [k for k in src_tensors if not (k.endswith(".lora_a") or k.endswith(".lora_b"))]
    if extra:
        raise ValueError(f"unexpected keys: {extra[:3]}")
    out: dict[str, np.ndarray] = {}
    report: list[dict] = []
    for p in prefixes:
        a_s = src_tensors[p + ".lora_a"].astype(np.float64)
        b_s = src_tensors[p + ".lora_b"].astype(np.float64)
        inn, r = a_s.shape
        r2, outd = b_s.shape
        if r != r2:
            raise ValueError(f"rank mismatch for {p}: {a_s.shape} vs {b_s.shape}")
        prod = a_s @ b_s
        s = np.linalg.svd(prod, compute_uv=False)[:r]
        sq = np.sqrt(s)
        a_r = _orthonormal(rng, inn, r) * sq[None, :]
        b_r = (_orthonormal(rng, outd, r) * sq[None, :]).T
        a32 = np.ascontiguousarray(a_r.astype(np.float32))
        b32 = np.ascontiguousarray(b_r.astype(np.float32))
        out[p + ".lora_a"], out[p + ".lora_b"] = a32, b32
        target = float(np.linalg.norm(prod))
        got = float(np.linalg.norm(a32.astype(np.float64) @ b32.astype(np.float64)))
        got16 = float(np.linalg.norm(a32.astype(np.float16).astype(np.float64) @ b32.astype(np.float16).astype(np.float64)))
        rel = lambda x: 0.0 if target == 0.0 else abs(x - target) / target
        report.append({"prefix": p, "target_product_norm": target, "random_product_norm": got, "rel_err": rel(got),
                       "rel_err_after_fp16": rel(got16),
                       "src_a_norm": float(np.linalg.norm(a_s)), "rand_a_norm": float(np.linalg.norm(a32)),
                       "src_b_norm": float(np.linalg.norm(b_s)), "rand_b_norm": float(np.linalg.norm(b32))})
    return out, report


def audit_saved(src: dict[str, np.ndarray], saved_path: pathlib.Path) -> None:
    """Re-read the SAVED artifact: same keys, shapes and dtype as the source, all values finite, per-pair singular values equal to the
    source's (rel 1e-4), and the float16 matrix error ||P16 - P32||_F / ||P32||_F below 1e-2. Registered tolerances: spectrum 1e-4 (float32 factors), product norm 1e-5, fp16 matrix 1e-2."""
    with safe_open(str(saved_path), "np") as f:
        saved = {k: f.get_tensor(k) for k in f.keys()}
    if sorted(saved) != sorted(src):
        raise SystemExit("saved keys differ from source keys")
    for k, v in saved.items():
        if v.shape != src[k].shape or v.dtype != np.float32 or not np.isfinite(v).all():
            raise SystemExit(f"bad tensor {k}: {v.shape} {v.dtype}")
    for p in sorted({k[: -len(".lora_a")] for k in saved if k.endswith(".lora_a")}):
        ps = src[p + ".lora_a"].astype(np.float64) @ src[p + ".lora_b"].astype(np.float64)
        p32 = saved[p + ".lora_a"].astype(np.float64) @ saved[p + ".lora_b"].astype(np.float64)
        p16 = saved[p + ".lora_a"].astype(np.float16).astype(np.float64) @ saved[p + ".lora_b"].astype(np.float16).astype(np.float64)
        r = src[p + ".lora_a"].shape[1]
        s_src, s_new = np.linalg.svd(ps, compute_uv=False)[:r], np.linalg.svd(p32, compute_uv=False)[:r]
        if not np.allclose(s_src, s_new, rtol=1e-4, atol=1e-9):
            raise SystemExit(f"singular values differ for {p}")
        if np.linalg.norm(ps) > 0 and np.linalg.norm(p16 - p32) / np.linalg.norm(p32) > 1e-2:
            raise SystemExit(f"fp16 matrix error too large for {p}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--seed", type=int, default=6061)
    a = ap.parse_args()
    src = pathlib.Path(a.src_dir)
    out = pathlib.Path(a.out_dir)
    if src.resolve() == out.resolve():
        raise SystemExit("refusing to write into the source adapter dir")
    out.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, np.ndarray] = {}
    with safe_open(str(src / "adapters.safetensors"), "np") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    rand, report = build_random(tensors, a.seed)
    save_file(rand, str(out / "adapters.safetensors"))
    cfg = json.load(open(src / "adapter_config.json"))
    cfg["adapter_path"] = str(out)               # no stale output path from the source adapter
    json.dump(cfg, open(out / "adapter_config.json", "w"), indent=4)
    audit_saved(tensors, out / "adapters.safetensors")
    worst = max(r["rel_err"] for r in report)
    worst16 = max(r["rel_err_after_fp16"] for r in report)
    json.dump({"seed": a.seed, "src": str(src), "n_pairs": len(report), "max_rel_err": worst, "max_rel_err_after_fp16": worst16,
               "pairs": report}, open(out / "random_adapter_report.json", "w"), indent=1)
    print(json.dumps({"pairs": len(report), "max_rel_err": worst, "max_rel_err_after_fp16": worst16, "seed": a.seed}))
    if worst > 1e-5 or worst16 > 1e-2:
        raise SystemExit(f"norm match failed: max rel err {worst}, after fp16 {worst16}")


if __name__ == "__main__":
    main()
