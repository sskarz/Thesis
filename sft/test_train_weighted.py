"""Self-test for sft/train_mlx_weighted.py. Part 1 (no model): synthetic boundary cases for seg_masks. Part 2 (needs the MLX base): masks and equal-weight reproduction of the default loss on real rows.
Exit code 1 on any failure."""
import json, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import mlx.core as mx
import sft.train_mlx_weighted as W

M = "/Users/USER/Documents/GitHub/Thesis/sft/models/gemma-4-e2b-it-mlx"
fail = []


def check(name, cond):
    print(("ok  " if cond else "FAIL"), name)
    if not cond:
        fail.append(name)


def synthetic():
    W.END_THINK = 7
    # batch of tokens; lengths = (offset, length) as in mlx_lm: supervised targets at steps offset..length (1-indexed over targets)
    def seg(tokens, off, ln):
        b = mx.array([tokens]); l = mx.array([[off, ln]])
        _, tg, mask, ans, rea = W.seg_masks(b, l)
        return mask.tolist()[0], ans.tolist()[0], rea.tolist()[0], tg.tolist()[0]
    # prompt 3 tokens, reasoning 3 tokens, separator, answer 2 tokens, turn-end, newline, pad
    toks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0]
    m, a, r, tg = seg(toks, 3, len(toks) - 1)
    check("reasoning targets are exactly toks[3:6]", [i for i, x in enumerate(r) if x] == [2, 3, 4])
    check("answer segment is separator..last supervised target", [i for i, x in enumerate(a) if x] == [5, 6, 7, 8, 9, 10])
    check("separator is first answer-segment target", tg[a.index(True)] == 7)
    check("segments partition the supervised span", [x or y for x, y in zip(a, r)] == m and not any(x and y for x, y in zip(a, r)))
    # no reasoning: separator is the first supervised target
    toks2 = [1, 2, 3, 7, 8, 9, 10, 11, 0]
    m, a, r, tg = seg(toks2, 3, len(toks2) - 1)
    check("no-reasoning row has an empty reasoning segment", not any(r) and any(a))
    # no separator: everything is reasoning
    toks3 = [1, 2, 3, 4, 5, 6, 8, 9, 0]
    m, a, r, tg = seg(toks3, 3, len(toks3) - 1)
    check("no-separator row is all reasoning", not any(a) and any(r))
    # separator only in the prompt region must not count (masked out)
    toks4 = [1, 7, 3, 4, 5, 6, 8, 9, 0]
    m, a, r, tg = seg(toks4, 3, len(toks4) - 1)
    check("separator inside the prompt is ignored", not any(a))
    # two separators: the first supervised one splits
    toks5 = [1, 2, 3, 4, 7, 5, 7, 8, 9, 0]
    m, a, r, tg = seg(toks5, 3, len(toks5) - 1)
    first = a.index(True)
    check("with two separators the first splits", tg[first] == 7 and a[first:] == [x for x in m[first:]])


def real():
    from mlx_lm import load
    from mlx_lm.tuner.datasets import ChatDataset, CacheDataset
    from mlx_lm.tuner.trainer import default_loss, iterate_batches
    model, tok = load(M)
    W.END_THINK = tok.convert_tokens_to_ids("<channel|>")
    rows = [json.loads(l) for l in open("sft/data/distill/r5/mlx-AB-think-clean/train.jsonl")][:6]
    ds = CacheDataset(ChatDataset(rows, tok, mask_prompt=True))
    for k, (batch, lengths) in zip(range(6), iterate_batches(ds, 1, 9500, False)):
        inputs, targets, mask, ans, rea = W.seg_masks(batch, lengths)
        check(f"row {k}: segments partition", int((ans & rea).sum()) == 0 and int((ans | rea).sum()) == int(mask.sum()))
        p = int(mx.argmax(ans[0].astype(mx.int32)).item())
        check(f"row {k}: answer segment starts at <channel|>", int(targets[0, p].item()) == W.END_THINK)
        d, _ = default_loss(model, batch, lengths)
        na, nr = int(ans.sum()), int(rea.sum())
        wl, _ = W.weighted_loss(model, batch, lengths, w_ans=na / (na + nr))
        check(f"row {k}: size-proportional weights reproduce the default loss ({float(d):.6f} vs {float(wl):.6f})", abs(float(d) - float(wl)) < 1e-4)


if __name__ == "__main__":
    synthetic()
    if "--no-model" not in sys.argv:
        real()
    print("SELFTEST", "FAIL" if fail else "PASS", fail)
    sys.exit(1 if fail else 0)
