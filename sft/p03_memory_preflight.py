#!/usr/bin/env python3
"""P0.3 memory preflight: one forward+backward pass (several timed repeats) with the registered 'wide' LoRA adapter on the longest training rows.

usage: p03_memory_preflight.py --layers 0-34 --rank 16 --len 9500 --out result.json [--row-file F --row-index I]
Wide adapter spec (training plan 2026-11-09, P0 and 'Wide adapter spec'): LoRA on q_proj, o_proj, gate/up/down MLP projections on every adapted layer, plus
k_proj and v_proj where they exist (layers 0-14; layers 15-34 share KV and have none), rank R, scale 2.0, dropout 0, gradient checkpointing, batch 1.
The model is the same MLX base as Round 5 (sft/models/gemma-4-e2b-it-mlx). Loss = stock mlx_lm default_loss with a prompt mask. No weights are saved. $0.
"""
import argparse, json, sys, time, gc
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm.utils import load
from mlx_lm.tuner.lora import LoRALinear
from mlx_lm.tuner.trainer import default_loss, grad_checkpoint

MODEL = "/Users/USER/Documents/GitHub/Thesis/sft/models/gemma-4-e2b-it-mlx"
ATTN = ["q_proj", "o_proj", "k_proj", "v_proj"]
MLP = ["gate_proj", "up_proj", "down_proj"]


def parse_range(s):
    a, b = s.split("-")
    return range(int(a), int(b) + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="0-34")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--len", type=int, required=True, help="sequence length in tokens (rows are tiled/truncated to this)")
    ap.add_argument("--row-file")
    ap.add_argument("--row-index", type=int, default=0)
    ap.add_argument("--qonly", action="store_true", help="Round 5 baseline config: q_proj only")
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    model, tok = load(MODEL)
    model.freeze()
    layers = model.language_model.model.layers
    n_lora = 0
    for i in parse_range(a.layers):
        L = layers[i]
        for grp, names in (((L.self_attn, ["q_proj"]),) if a.qonly else ((L.self_attn, ATTN), (L.mlp, MLP))):
            for n in names:
                if hasattr(grp, n):
                    setattr(grp, n, LoRALinear.from_base(getattr(grp, n), r=a.rank, scale=2.0, dropout=0.0))
                    n_lora += 1
    trainable = sum(v.size for _, v in nn.utils.tree_flatten(model.trainable_parameters()))
    grad_checkpoint(layers[0])
    if a.row_file:
        row = [json.loads(l) for l in open(a.row_file)][a.row_index]
        ids = tok.apply_chat_template(row["messages"], tokenize=True)
        ids = list(ids["input_ids"] if isinstance(ids, dict) else ids)
    else:
        ids = []
    real_len = len(ids)
    if not ids:
        ids = list(range(1000, 1000 + a.len + 1))
    while len(ids) < a.len + 1:                      # tile a shorter real row up to the requested stress length (memory depends on length, not content)
        ids = ids + ids
    ids = ids[: a.len + 1]
    batch = mx.array([ids])
    offset = max(1, a.len - 400)                      # loss on the last ~400 tokens (prompt masked), as with mask_prompt
    lengths = mx.array([[offset, a.len + 1]])
    opt = optim.AdamW(learning_rate=1e-4)
    loss_and_grad = nn.value_and_grad(model, default_loss)
    times, peaks, losses = [], [], []
    mx.reset_peak_memory()
    for r in range(a.repeats):
        t0 = time.time()
        (loss, ntoks), grads = loss_and_grad(model, batch, lengths)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state, loss)
        dt = time.time() - t0
        times.append(round(dt, 2)); losses.append(round(float(loss), 4)); peaks.append(round(mx.get_peak_memory() / 2**30, 2))
    res = {"qonly": a.qonly, "layers": a.layers, "rank": a.rank, "seq_len": a.len, "row_real_tokens": real_len, "lora_modules": n_lora, "trainable_params": trainable,
           "seconds_per_step": times, "peak_gib_after_each_step": peaks, "loss": losses, "mlx_active_gib_end": round(mx.get_active_memory() / 2**30, 2)}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
