#!/usr/bin/env python3
"""G3 trainer (copy of g1_train.py): registered 'wide' LoRA (q/o/gate/up/down on every adapted layer, k/v where they exist), prompt mask; --think off = arm A (no think marker, JSON call list target), --think on = arm B (exactly one <|think|> marker in the system turn, target = thought channel with the teacher sentences, then the call list).
Extra vs G1: rows longer than --max-tokens (default 6999, the memory limit measured in P0.3) are dropped and counted (dropped_rows.json); a checkpoint at the end of pass 1 (adapters_pass1/); validation loss on --valid rows at the end of each pass (logged only, never used to select anything); default 2 passes and a 30 h cap.

usage: g1_train.py --rows g1_rows.jsonl --out-dir OUT [--layers 8-34] [--rank 16] [--lr 1e-4] [--passes 4] [--max-micro 1000] [--accum 4]
                   [--seed 61] [--max-seconds 36000] [--smoke N] [--dry-tokenize]
Differences from mlx_lm.lora, on purpose: (1) rows are rendered with enable_thinking=False so the training prompt has no <|think|> marker and matches the thinking-off
server (mlx_lm.lora cannot pass that flag); (2) per-layer module choice (layers 15-34 have no k/v). Loss = stock mlx_lm default_loss (mean over unmasked tokens).
Schedule: linear warm-up over 5% of updates, then linear decay to zero. Optimizer: Adam (mlx_lm default). Gradient accumulation = --accum rows per update (batch 1).
Exit codes: 0 done; 4 non-finite loss; 5 diverged (mean loss of updates 76-100 above mean of updates 1-10); 6 time cap; 2 usage/data error.
"""
import argparse, json, math, pathlib, random, sys, time

MODEL = "/Users/sanskarthapa/Documents/GitHub/Thesis/sft/models/gemma-4-e2b-it-mlx"
ATTN = ["q_proj", "o_proj", "k_proj", "v_proj"]
MLP = ["gate_proj", "up_proj", "down_proj"]


THINK = False


def render_ids(tok, messages, add_generation_prompt=False):
    out = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=add_generation_prompt, enable_thinking=THINK, return_dict=False)
    return list(out["input_ids"]) if isinstance(out, dict) else list(out)


def prepare(tok, rows):
    """-> list of (token_ids, offset, task_id). Asserts: prefix parity, last message is the assistant target, no think marker anywhere."""
    out = []
    think_id = tok.convert_tokens_to_ids("<|think|>")
    for r in rows:
        m = r["messages"]
        assert m[-1]["role"] == "assistant", "last message must be the assistant target"
        ids = render_ids(tok, m)
        pre = render_ids(tok, m[:-1], add_generation_prompt=True)
        assert ids[: len(pre)] == pre, f"prefix mismatch for {r['meta']['task_id']}"
        assert len(ids) > len(pre), "empty target"
        n_think = sum(1 for x in ids if x == think_id)
        assert n_think == (1 if THINK else 0), f"expected {1 if THINK else 0} <|think|> markers, found {n_think} for {r['meta']['task_id']}"
        if THINK:
            assert "reasoning_content" in m[-1] and m[-1]["reasoning_content"], f"arm B row without sentences: {r['meta']['task_id']}"
        else:
            assert not m[-1].get("reasoning_content"), f"arm A row with reasoning: {r['meta']['task_id']}"
        bos = tok.bos_token_id
        assert bos is None or (ids[0] == bos and ids[1] != bos), f"BOS must appear exactly once at the start ({r['meta']['task_id']})"
        out.append((ids, len(pre), r["meta"]["task_id"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--layers", default="8-34"); ap.add_argument("--rank", type=int, default=16); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--passes", type=int, default=2); ap.add_argument("--max-micro", type=int, default=100000); ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--seed", type=int, default=61); ap.add_argument("--max-seconds", type=float, default=108000)
    ap.add_argument("--smoke", type=int, default=0, help="run only this many micro-batches (still writes the adapter, for a conversion test)")
    ap.add_argument("--dry-tokenize", action="store_true")
    ap.add_argument("--think", choices=["on", "off"], required=True); ap.add_argument("--valid", default=None); ap.add_argument("--max-tokens", type=int, default=6999)
    a = ap.parse_args()
    global THINK
    THINK = a.think == "on"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    rows = [json.loads(l) for l in open(a.rows, encoding="utf-8") if l.strip()]
    data_all = prepare(tok, rows)
    dropped = [(d[2], len(d[0])) for d in data_all if len(d[0]) > a.max_tokens]
    data = [d for d in data_all if len(d[0]) <= a.max_tokens]
    out = pathlib.Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    json.dump({"max_tokens": a.max_tokens, "rows_in": len(data_all), "rows_kept": len(data), "rows_dropped": len(dropped), "tasks_losing_rows": len({t for t, _ in dropped}), "tasks_total": len({d[2] for d in data_all}), "dropped": dropped}, open(out / "dropped_rows.json", "w"))
    print(json.dumps({"rows_in": len(data_all), "rows_kept": len(data), "rows_dropped": len(dropped)}))
    lens = sorted(len(d[0]) for d in data)
    print(json.dumps({"rows": len(data), "tokens_max": lens[-1], "tokens_median": lens[len(lens) // 2], "target_tokens_total": sum(len(d[0]) - d[1] for d in data)}))
    if a.dry_tokenize:
        return 0
    import mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_map
    from mlx_lm.utils import load
    from mlx_lm.tuner.lora import LoRALinear
    from mlx_lm.tuner.trainer import default_loss, grad_checkpoint
    mx.random.seed(a.seed); rng = random.Random(a.seed)
    model, _ = load(MODEL); model.freeze()
    layers = model.language_model.model.layers
    lo, hi = (int(x) for x in a.layers.split("-"))
    n_lora = 0
    for i in range(lo, hi + 1):
        L = layers[i]
        for grp, names in ((L.self_attn, ATTN), (L.mlp, MLP)):
            for n in names:
                if hasattr(grp, n):
                    setattr(grp, n, LoRALinear.from_base(getattr(grp, n), r=a.rank, scale=2.0, dropout=0.0)); n_lora += 1
    grad_checkpoint(layers[0])
    total_micro = min(a.max_micro, a.passes * len(data)) if not a.smoke else a.smoke
    total_updates = max(1, total_micro // a.accum)
    warm = max(1, int(0.05 * total_updates))
    sched = optim.join_schedules([optim.linear_schedule(0.0, a.lr, warm), optim.linear_schedule(a.lr, 0.0, max(1, total_updates - warm))], [warm])
    opt = optim.Adam(learning_rate=sched)
    order = []
    while len(order) < total_micro:
        p = list(range(len(data))); rng.shuffle(p); order += p
    order = order[:total_micro]
    loss_and_grad = nn.value_and_grad(model, default_loss)
    cfg = {"fine_tune_type": "lora", "num_layers": hi - lo + 1, "lora_parameters": {"rank": a.rank, "scale": 2.0, "dropout": 0.0,
           "keys": sorted(set(ATTN + MLP))}, "g1": {"layers": a.layers, "lr": a.lr, "accum": a.accum, "seed": a.seed, "micro": total_micro,
           "updates": total_updates, "warmup_updates": warm, "rows": len(data), "lora_modules": n_lora, "render": f"enable_thinking={THINK}", "think_arm": a.think, "passes": a.passes, "rows_dropped": len(dropped)}}
    vrows = [d for d in prepare(tok, [json.loads(l) for l in open(a.valid, encoding="utf-8") if l.strip()]) if len(d[0]) <= a.max_tokens] if a.valid else []
    print(json.dumps(cfg["g1"]))
    t0 = time.time(); acc = None; losses = []; log = open(out / "train_log.jsonl", "a")
    for k, idx in enumerate(order, 1):
        ids, off, tid = data[idx]
        batch = mx.array([ids]); lengths = mx.array([[off, len(ids)]])
        t1 = time.time()
        (loss, ntoks), grads = loss_and_grad(model, batch, lengths)
        grads = tree_map(lambda g: g / a.accum, grads)
        acc = grads if acc is None else tree_map(lambda x, y: x + y, acc, grads)
        mx.eval(loss, acc)
        lv = float(loss); losses.append(lv)
        log.write(json.dumps({"micro": k, "task": tid, "tokens": len(ids), "target_tokens": len(ids) - off, "loss": round(lv, 5), "s": round(time.time() - t1, 2)}) + "\n"); log.flush()
        if not math.isfinite(lv):
            print("ABORT: non-finite loss at micro", k); return 4
        if k % a.accum == 0:
            opt.update(model, acc); mx.eval(model.parameters(), opt.state); acc = None
            u = k // a.accum
            if u == 100 and len(losses) >= 400:                      # update 100 = micro 400: divergence rule
                early = sum(losses[: 10 * a.accum]) / (10 * a.accum); late = sum(losses[75 * a.accum: 100 * a.accum]) / (25 * a.accum)
                print(f"divergence check at update 100: mean(u1-10)={early:.4f} mean(u76-100)={late:.4f}")
                if late > early:
                    print("ABORT: diverged"); return 5
            if u % 10 == 0:
                print(f"update {u}/{total_updates} micro {k} loss(last {a.accum})={sum(losses[-a.accum:]) / a.accum:.4f} elapsed {time.time() - t0:.0f}s", flush=True)
            if not a.smoke and u % 50 == 0:
                mx.save_safetensors(str(out / "adapters_ckpt.safetensors"), dict(tree_flatten(model.trainable_parameters())))
        if not a.smoke and k % len(data) == 0 and k // len(data) <= a.passes:
            pn = k // len(data)
            if pn == 1 and a.passes > 1:
                p1 = out / "adapters_pass1"; p1.mkdir(exist_ok=True)
                mx.save_safetensors(str(p1 / "adapters.safetensors"), dict(tree_flatten(model.trainable_parameters()))); json.dump(cfg, open(p1 / "adapter_config.json", "w"), indent=1)
            if a.valid:
                vl = []
                for r in vrows:
                    if time.time() - t0 > a.max_seconds:
                        break                      # the training deadline also binds during validation
                    ids_v, off_v, _ = r
                    l_v, _n = default_loss(model, mx.array([ids_v]), mx.array([[off_v, len(ids_v)]])); mx.eval(l_v); vl.append(float(l_v))
                if vl:
                    log.write(json.dumps({"pass_end": pn, "micro": k, "valid_rows": len(vl), "valid_loss_mean": round(sum(vl) / len(vl), 5)}) + "\n"); log.flush()
        if time.time() - t0 > a.max_seconds:
            print("ABORT: time cap"); return 6
    if a.smoke:
        print("smoke done", round(time.time() - t0, 1), "s", "peak GiB", round(mx.get_peak_memory() / 2**30, 2))
    ad = out / "adapters"; ad.mkdir(exist_ok=True)
    mx.save_safetensors(str(ad / "adapters.safetensors"), dict(tree_flatten(model.trainable_parameters())))
    json.dump(cfg, open(ad / "adapter_config.json", "w"), indent=1)
    print(json.dumps({"done": True, "micro": total_micro, "updates": total_updates, "seconds": round(time.time() - t0, 1),
                      "loss_first10_mean": round(sum(losses[:10]) / 10, 4), "loss_last40_mean": round(sum(losses[-40:]) / len(losses[-40:]), 4),
                      "peak_gib": round(mx.get_peak_memory() / 2**30, 2)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
