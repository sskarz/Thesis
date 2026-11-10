"""Round 5 B4: sequence-sum DPO and the matched chosen-only SFT control on same-state preference pairs. MLX, LoRA via mlx_lm layers.
Pair file: JSONL {"pair_id","messages":[prompt messages incl. <|think|> system],"chosen":{"content","reasoning_content"},"rejected":{...},"meta":{"task_id",...}}
Objective dpo: L = -log sigmoid(beta * ((lp_c - ref_c) - (lp_r - ref_r))), lp = SUM of token log-probs over the assistant continuation
  (reasoning + separator + answer + <turn|>; trailing newline and pad excluded), no length normalization anywhere.
Objective sft_chosen: L = -mean token log-prob of the chosen continuation (same tokens, same shuffled order, same optimizer, same adapter).
Reference = the bf16 base: LoRA B = 0 at init so policy == base before the first update; reference scores are materialized before any update
(preflight compares them with the true base without LoRA layers on BOTH branches). Both branches are differentiated jointly (equal to the
sequential-branch gradient in the preflight); the vocabulary projection is sliced to the continuation positions; fp32 log-softmax; no truncation
(pairs longer than --max-seq are rejected). Gradient checkpointing on the decoder blocks. The adapter is q-only in effect (layers 19-34 share KV so
the configured v_proj keys have no effect; 524,288 trainable parameters), identical to the SFT arms.
usage: r5_dpo.py --pairs train.jsonl --val val.jsonl --out ADAPTER_DIR --objective {dpo,sft_chosen} [--beta 0.1 --lr 1e-5 --updates 80 --accum 4 --seed 61 --max-seq 9500] [--preflight]"""
import argparse, json, math, pathlib, random, sys, time
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map, tree_unflatten
from mlx_lm import load
from mlx_lm.tuner.trainer import grad_checkpoint
from mlx_lm.tuner.utils import linear_to_lora_layers

MODEL = "/Users/sanskarthapa/Documents/GitHub/Thesis/sft/models/gemma-4-e2b-it-mlx"
LORA = {"rank": 8, "scale": 2.0, "dropout": 0.0, "keys": ["self_attn.q_proj", "self_attn.v_proj"]}
NUM_LAYERS = 16
MIN_PAIRS, MIN_TASKS = 150, 50   # registered pair floor (applied to the post-filter TRAIN set)
END_THINK = None


def ids(x):
    return x["input_ids"] if hasattr(x, "keys") else x


def encode(tok, messages, resp):
    full = ids(tok.apply_chat_template(messages + [{"role": "assistant", **resp}], tokenize=True))
    prompt = ids(tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True))
    assert full[:len(prompt)] == prompt, "prompt is not a prefix of the full render"
    assert tok.convert_ids_to_tokens(full[-2:]) == ["<turn|>", "\n"], tok.convert_ids_to_tokens(full[-3:])
    assert tok.convert_ids_to_tokens(full[len(prompt): len(prompt) + 3]) == ["<|channel>", "thought", "\n"], "continuation must start with the thought channel"
    return full, len(prompt)


def softcap(c, x):
    return mx.tanh(x / c) * c


def sep_index(tokens, off):
    """number of continuation tokens before the first <channel|> separator (= start of the answer segment)"""
    for j in range(off, len(tokens) - 1):
        if tokens[j] == END_THINK:
            return j - off
    return len(tokens) - 1 - off


def cont_logps(model, tokens, offset):
    """per-token log-probs of tokens[offset : n-1] (continuation up to and including <turn|>), float32, shape [T]"""
    lm = model.language_model
    x = mx.array(tokens[:-1])[None]
    h = lm.model(x)
    hs = h[:, offset - 1: len(tokens) - 2]
    assert lm.tie_word_embeddings
    logits = lm.model.embed_tokens.as_linear(hs)
    if lm.final_logit_softcapping is not None:
        logits = softcap(lm.final_logit_softcapping, logits)
    logits = logits.astype(mx.float32)
    tgt = mx.array(tokens[offset: len(tokens) - 1])[None]
    lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    return mx.take_along_axis(lp, tgt[..., None], axis=-1)[0, :, 0]


def seg_sums(tokens, off, lp):
    k = sep_index(tokens, off)
    return lp[:k].sum(), lp[k:].sum()


def load_pairs(tok, path, max_seq):
    out, rej = [], 0
    for l in open(path):
        if not l.strip():
            continue
        p = json.loads(l)
        c, oc = encode(tok, p["messages"], p["chosen"]); r, orr = encode(tok, p["messages"], p["rejected"])
        assert oc == orr and c[:oc] == r[:orr], "chosen/rejected prompts differ"
        if max(len(c), len(r)) > max_seq:
            rej += 1; continue
        out.append({"id": p["pair_id"], "c": c, "r": r, "off": oc, "meta": p.get("meta", {})})
    return out, rej


def main():
    global END_THINK
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True); ap.add_argument("--val", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--objective", choices=["dpo", "sft_chosen"], default="dpo")
    ap.add_argument("--beta", type=float, default=0.1); ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--updates", type=int, default=80); ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--seed", type=int, default=61); ap.add_argument("--max-seq", type=int, default=9500)
    ap.add_argument("--preflight", action="store_true")
    a = ap.parse_args()
    mx.random.seed(a.seed)
    model, tok = load(MODEL)
    END_THINK = tok.convert_tokens_to_ids("<channel|>")
    train, rej_t = load_pairs(tok, a.pairs, a.max_seq); val, rej_v = load_pairs(tok, a.val, a.max_seq)
    print(f"pairs train {len(train)} (rejected overlength {rej_t}) val {len(val)} (rejected {rej_v})", flush=True)
    assert train and val, "empty train or validation pair set after filtering"
    ntasks = len({p["meta"].get("task_id") for p in train})
    if not a.preflight:
        assert len(train) >= MIN_PAIRS and ntasks >= MIN_TASKS, f"pair floor not met: {len(train)} pairs / {ntasks} tasks (need {MIN_PAIRS}/{MIN_TASKS})"
        assert not ({p["meta"].get("task_id") for p in train} & {p["meta"].get("task_id") for p in val}), "train/val tasks overlap"
    model.freeze()
    base_tok = {}
    p0 = train[0]
    for nm in ("c", "r"):                                     # true base, no LoRA layers, both branches
        base_tok[nm] = cont_logps(model, p0[nm], p0["off"]); mx.eval(base_tok[nm])
    linear_to_lora_layers(model, NUM_LAYERS, LORA)
    grad_checkpoint(model.layers[0])
    tparams = dict(tree_flatten(model.trainable_parameters()))
    assert tparams and all("lora_" in k for k in tparams), "trainable parameters must be LoRA only"
    w_init = {k: mx.array(v) for k, v in tparams.items()}
    print("trainable params", sum(v.size for v in tparams.values()), flush=True)
    for nm in ("c", "r"):
        z0 = cont_logps(model, p0[nm], p0["off"]); mx.eval(z0)
        d = float(mx.abs(z0 - base_tok[nm]).max())
        print(f"init check {nm}: max abs per-token logp diff base vs LoRA(B=0): {d}", flush=True)
        assert d < 1e-3, "reference != policy at init"

    def score(p):
        out = []
        for nm in ("c", "r"):
            lp = cont_logps(model, p[nm], p["off"]); re, an = seg_sums(p[nm], p["off"], lp)
            mx.eval(lp.sum(), re, an); out.append((float(lp.sum()), float(re), float(an), int(lp.shape[0])))
            mx.clear_cache()
        return out

    t0 = time.time()
    ref_tr = [score(p) for p in train] if a.objective == "dpo" else None
    ref_va = [score(p) for p in val] if a.objective == "dpo" else None
    print(f"reference scores in {time.time()-t0:.0f}s", flush=True)

    def joint_dpo(m, p, ref):
        lc = cont_logps(m, p["c"], p["off"]); lr = cont_logps(m, p["r"], p["off"])
        lpc, lpr = lc.sum(), lr.sum()
        z = a.beta * ((lpc - ref[0][0]) - (lpr - ref[1][0]))
        rc, ac = seg_sums(p["c"], p["off"], lc); rr, ar = seg_sums(p["r"], p["off"], lr)
        return mx.logaddexp(0.0, -z), lpc, lpr, z, rc, ac, rr, ar

    def joint_sft(m, p, ref):
        lc = cont_logps(m, p["c"], p["off"])
        return -lc.mean(), lc.sum()

    fn = joint_dpo if a.objective == "dpo" else joint_sft
    vg = nn.value_and_grad(model, lambda m, p, ref: fn(m, p, ref))

    def pair_step(p, ref):
        out, g = vg(model, p, ref); mx.eval(out, g)
        if a.objective == "sft_chosen":
            return {"loss": float(out[0]), "lpc": float(out[1])}, g
        loss, lpc, lpr, z, rc, ac, rr, ar = [float(x) for x in out]
        return {"loss": loss, "z": z, "dc": lpc - ref[0][0], "dr": lpr - ref[1][0], "lpc": lpc, "lpr": lpr,
                "dc_reason": rc - ref[0][1], "dc_answer": ac - ref[0][2], "dr_reason": rr - ref[1][1], "dr_answer": ar - ref[1][2]}, g

    if a.preflight:
        i = max(range(len(train)), key=lambda k: max(len(train[k]["c"]), len(train[k]["r"])))
        print("longest pair length", max(len(train[i]["c"]), len(train[i]["r"])), flush=True)
        mx.reset_peak_memory()
        ref_i = ref_tr[i] if ref_tr else None
        t2 = time.time(); info, g = pair_step(train[i], ref_i); print("longest-pair joint step seconds", round(time.time() - t2, 1), "info", info, flush=True)
        if a.objective == "dpo":
            assert abs(info["loss"] - math.log(2)) < 1e-2 and abs(info["z"]) < 1e-2, "loss at init must be ln 2"
        gflat = dict(tree_flatten(g)); assert gflat and all("lora_" in k for k in gflat) and all(bool(mx.all(mx.isfinite(v)).item()) for v in gflat.values())
        # sliced head == full logits (finite and within tolerance)
        p = train[0]; full = model(mx.array(p["c"][:-1])[None]).astype(mx.float32)
        lpf = full - mx.logsumexp(full, axis=-1, keepdims=True)
        ref_tokv = mx.take_along_axis(lpf[:, p["off"] - 1: len(p["c"]) - 2], mx.array(p["c"][p["off"]: len(p["c"]) - 1])[None, :, None], axis=-1)[0, :, 0]
        sl = cont_logps(model, p["c"], p["off"]); dmax = float(mx.abs(ref_tokv - sl).max())
        print("sliced vs full-logits continuation logp max abs diff", dmax, flush=True); assert math.isfinite(dmax) and dmax < 1e-3
        # causality: changing the LAST continuation token must not change earlier per-token log-probs
        t2_ = list(p["c"]); t2_[-2] = (t2_[-2] + 1) % 262144
        sl2 = cont_logps(model, t2_, p["off"]); diff_early = float(mx.abs(sl2[:-1] - sl[:-1]).max())
        print("causality check (early positions unchanged by a later-token change):", diff_early, flush=True); assert diff_early < 1e-3
        # boundaries: continuation = thought channel ... <turn|>, targets exclude the trailing newline and the pad
        print("continuation head", tok.convert_ids_to_tokens(p["c"][p["off"]: p["off"] + 3]), "tail", tok.convert_ids_to_tokens(p["c"][-3:]), "n_targets", int(sl.shape[0]), flush=True)
        assert int(sl.shape[0]) == len(p["c"]) - 1 - p["off"]
        print("separator index within continuation (chosen/rejected)", sep_index(p["c"], p["off"]), sep_index(p["r"], p["off"]), flush=True)
        # gradient checks at a non-zero LoRA state on the shortest pair
        j = min(range(len(train)), key=lambda k: max(len(train[k]["c"]), len(train[k]["r"]))); q = train[j]
        w0 = dict(tree_flatten(model.trainable_parameters()))
        model.update(tree_unflatten([(k, v + 0.02 * mx.random.normal(v.shape, key=mx.random.key(100 + n))) for n, (k, v) in enumerate(w0.items())]))
        w1 = dict(tree_flatten(model.trainable_parameters()))
        refj = ref_tr[j] if ref_tr else None
        if a.objective == "dpo":
            def seq_pair(pp, ref):                             # sequential-branch reference implementation
                lpc = float(cont_logps(model, pp["c"], pp["off"]).sum()); lpr = float(cont_logps(model, pp["r"], pp["off"]).sum())
                z = a.beta * ((lpc - ref[0][0]) - (lpr - ref[1][0])); coef = a.beta * (1.0 / (1.0 + math.exp(z)))
                vgb = nn.value_and_grad(model, lambda m, tk, off, c: c * cont_logps(m, tk, off).sum())
                _, gc = vgb(model, pp["c"], pp["off"], -coef); mx.eval(gc)
                _, gr = vgb(model, pp["r"], pp["off"], +coef)
                gg = tree_map(lambda x, y: x + y, gc, gr); mx.eval(gg); return gg
            gs = dict(tree_flatten(seq_pair(q, refj)))
            _, gjn = pair_step(q, refj); gj = dict(tree_flatten(gjn))
            rel = max(float(mx.abs(gs[k] - gj[k]).max()) / (float(mx.abs(gj[k]).max()) + 1e-12) for k in gj)
            print("sequential vs joint gradient max relative diff", rel, flush=True); assert rel < 5e-2
        else:
            _, gjn = pair_step(q, refj); gj = dict(tree_flatten(gjn))
        info0, _ = pair_step(q, refj)
        gn = float(mx.sqrt(sum((v * v).sum() for v in gj.values())))
        model.update(tree_unflatten([(k, w1[k] - (0.5 / max(gn, 1e-12)) * gj[k]) for k in w1]))
        info1, _ = pair_step(q, refj)
        print("descent check: loss before", info0["loss"], "after small step", info1["loss"], flush=True); assert info1["loss"] < info0["loss"]
        print("full preflight peak memory GB", mx.get_peak_memory() / 1e9, flush=True)
        print("PREFLIGHT DONE", flush=True)
        return

    rng = random.Random(a.seed)
    order, n_micro = [], a.updates * a.accum
    while len(order) < n_micro:
        idx = list(range(len(train))); rng.shuffle(idx); order += idx
    order = order[:n_micro]
    opt = optim.Adam(learning_rate=a.lr)
    outd = pathlib.Path(a.out); outd.mkdir(parents=True, exist_ok=True)
    log = open(outd / "train_log.jsonl", "w")

    def evaluate(tag):
        if a.objective == "dpo":
            zs, ls = [], []
            for p, ref in zip(val, ref_va):
                s = score(p)
                z = a.beta * ((s[0][0] - ref[0][0]) - (s[1][0] - ref[1][0])); zs.append(z); ls.append(math.log1p(math.exp(-z)))
            rec = {"eval": tag, "val_pref_acc": sum(z > 0 for z in zs) / len(zs), "val_mean_margin_z": sum(zs) / len(zs), "val_loss": sum(ls) / len(ls), "n": len(zs)}
        else:
            nl = []
            for p in val:
                lc = cont_logps(model, p["c"], p["off"]); mx.eval(lc); nl.append(-float(lc.mean())); mx.clear_cache()
            rec = {"eval": tag, "val_chosen_mean_nll": sum(nl) / len(nl), "n": len(nl)}
        print(rec, flush=True); log.write(json.dumps(rec) + "\n"); log.flush()

    evaluate("init")
    t0 = time.time()
    for u in range(a.updates):
        acc = None; recs = []
        for m in range(a.accum):
            k = order[u * a.accum + m]
            info, g = pair_step(train[k], ref_tr[k] if ref_tr else None)
            acc = g if acc is None else tree_map(lambda x, y: x + y, acc, g)
            mx.eval(acc); recs.append({"pair_id": train[k]["id"], **info}); mx.clear_cache()
        acc = tree_map(lambda x: x / a.accum, acc)
        opt.update(model, acc); mx.eval(model.parameters(), opt.state)
        cur = dict(tree_flatten(model.trainable_parameters()))
        dn = float(mx.sqrt(sum(((cur[k] - w_init[k]) ** 2).sum() for k in cur)))
        dnb = float(mx.sqrt(sum(((cur[k] - w_init[k]) ** 2).sum() for k in cur if k.endswith("lora_b"))))
        rec = {"update": u + 1, "mean_loss": sum(r["loss"] for r in recs) / len(recs), "adapter_delta_norm": dn, "adapter_delta_norm_B": dnb,
               "pairs": recs, "elapsed_s": round(time.time() - t0)}
        log.write(json.dumps(rec) + "\n"); log.flush()
        if (u + 1) % 5 == 0:
            print(f"update {u+1}/{a.updates} loss {rec['mean_loss']:.4f} delta {dn:.4f} elapsed {rec['elapsed_s']}s", flush=True)
        if (u + 1) in (a.updates // 2, a.updates):
            evaluate(f"update_{u+1}")
    mx.save_safetensors(str(outd / "adapters.safetensors"), dict(tree_flatten(model.trainable_parameters())))
    cfg = json.load(open("sft/adapters-r3-think-AB_lr2e5/adapter_config.json"))
    cfg.update({"adapter_path": a.out, "learning_rate": a.lr, "iters": a.updates * a.accum, "seed": a.seed, "grad_accumulation_steps": a.accum,
                "config": f"sft/r5_dpo.py objective={a.objective} beta={a.beta} max_seq={a.max_seq}", "data": a.pairs, "max_seq_length": a.max_seq})
    json.dump(cfg, open(outd / "adapter_config.json", "w"), indent=4)
    print("saved", a.out, flush=True)


if __name__ == "__main__":
    main()
