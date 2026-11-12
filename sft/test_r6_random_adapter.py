#!/usr/bin/env python3
"""Offline self-test for r6_random_adapter.py (no model, no server)."""
import json
import pathlib
import subprocess
import sys
import tempfile

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

HERE = pathlib.Path(__file__).resolve().parent


def prod(f, p):
    return f.get_tensor(p + ".lora_a").astype(np.float64) @ f.get_tensor(p + ".lora_b").astype(np.float64)


def main() -> None:
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as d:
        d = pathlib.Path(d)
        src = d / "src"
        src.mkdir()
        t = {}
        for i, (inn, out) in enumerate([(32, 16), (32, 48), (32, 16)]):
            p = f"language_model.model.layers.{19 + i}.self_attn.q_proj"
            t[p + ".lora_a"] = (rng.standard_normal((inn, 8)) * 0.1).astype(np.float32)
            t[p + ".lora_b"] = (rng.standard_normal((8, out)) * (0.01 * (i + 1))).astype(np.float32)
        save_file(t, str(src / "adapters.safetensors"))
        (src / "adapter_config.json").write_text(json.dumps({"lora_parameters": {"rank": 8, "scale": 2.0}}))
        res = []
        for name in ("o1", "o2"):
            r = subprocess.run([sys.executable, str(HERE / "r6_random_adapter.py"), str(src), str(d / name), "--seed", "6061"],
                               capture_output=True, text=True)
            assert r.returncode == 0, r.stderr
            res.append(d / name / "adapters.safetensors")
        assert subprocess.run([sys.executable, str(HERE / "r6_random_adapter.py"), str(src), str(src)], capture_output=True, text=True).returncode != 0
        with safe_open(str(src / "adapters.safetensors"), "np") as fs, safe_open(str(res[0]), "np") as f1, safe_open(str(res[1]), "np") as f2:
            assert sorted(fs.keys()) == sorted(f1.keys()) == sorted(f2.keys())
            prefixes = sorted({k[: -len(".lora_a")] for k in fs.keys() if k.endswith(".lora_a")})
            for p in prefixes:
                for suf in (".lora_a", ".lora_b"):
                    s, x, y = fs.get_tensor(p + suf), f1.get_tensor(p + suf), f2.get_tensor(p + suf)
                    assert s.shape == x.shape and x.dtype == np.float32
                    assert np.array_equal(x, y), "not deterministic for a fixed seed"
                    assert not np.allclose(s, x), "random adapter equals source"
                ps, px = prod(fs, p), prod(f1, p)
                ss, sx = np.linalg.svd(ps, compute_uv=False)[:8], np.linalg.svd(px, compute_uv=False)[:8]
                assert np.allclose(ss, sx, rtol=1e-4), ("spectrum differs", p, ss, sx)
                assert abs(np.linalg.norm(ps) - np.linalg.norm(px)) / np.linalg.norm(ps) < 1e-5
                # directions are random: product is far from the source product in the sense of low cosine similarity
                cos = abs(np.sum(ps * px)) / (np.linalg.norm(ps) * np.linalg.norm(px))
                assert cos < 0.5, ("directions too similar", p, cos)
                a, b = f1.get_tensor(p + ".lora_a"), f1.get_tensor(p + ".lora_b")
                assert abs(np.linalg.norm(a) - np.linalg.norm(b)) / np.linalg.norm(a) < 1e-4, "factors not balanced"
        rep = json.load(open(d / "o1" / "random_adapter_report.json"))
        assert rep["max_rel_err"] < 1e-5 and rep["max_rel_err_after_fp16"] < 1e-2 and rep["n_pairs"] == 3
        r = subprocess.run([sys.executable, str(HERE / "r6_random_adapter.py"), str(src), str(d / "o3"), "--seed", "1"], capture_output=True, text=True)
        assert r.returncode == 0
        with safe_open(str(d / "o3" / "adapters.safetensors"), "np") as f3, safe_open(str(res[0]), "np") as f1:
            k = sorted(f3.keys())[0]
            assert not np.array_equal(f3.get_tensor(k), f1.get_tensor(k))
        cfg = json.load(open(d / "o1" / "adapter_config.json")); assert cfg["adapter_path"] == str(d / "o1"), cfg
    print("test_r6_random_adapter: OK")


if __name__ == "__main__":
    main()
