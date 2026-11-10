"""A0.1 (Round 5): supervised-token segments of mask_prompt SFT rows. Tokenizer only (no model, no network).
Segments of the supervised span (everything after the generation prompt): reasoning = tokens before the first <channel|>
(including the opening channel marker the template emits), separator = the <channel|> token, answer = tokens after it up to
but excluding the final turn-end tokens, termination = the template suffix '<turn|>' + '\n' (explicit, asserted). The stock mlx_lm loss mask (steps <= length) additionally supervises ONE <pad> target per row
(stock_mlx_extra_pad_targets), counted separately here and not part of the rendered continuation totals (Codex r29).
usage: token_segments_audit.py <out.json> rows1.jsonl [rows2.jsonl ...]"""
import json, sys, collections, statistics
from transformers import AutoTokenizer

M = "/Users/sanskarthapa/Documents/GitHub/Thesis/sft/models/gemma-4-e2b-it-mlx"
tok = AutoTokenizer.from_pretrained(M)
END_THINK = tok.convert_tokens_to_ids("<channel|>")


def ids(x):
    return x["input_ids"] if hasattr(x, "keys") else x


def seg(r):
    m = r["messages"]
    p = ids(tok.apply_chat_template(m[:-1], add_generation_prompt=True, tokenize=True))
    f = ids(tok.apply_chat_template(m, tokenize=True))
    assert f[:len(p)] == p
    t = f[len(p):]
    k = t.index(END_THINK) if END_THINK in t else len(t)
    reasoning, rest = t[:k], t[k:]
    sep = rest[:1]
    ans = rest[1:]
    # termination: trailing special tokens (ids >= tokenizer special-token set) after the last '}' or ']' token
    # explicit expected boundary (Codex r29): the render ends with '<turn|>' then an ordinary newline token; nothing else is stripped
    toks = tok.convert_ids_to_tokens(ans)
    assert toks[-2:] == ["<turn|>", "\n"], toks[-3:]
    tail = 2
    return {"prompt": len(p), "reasoning": len(reasoning), "separator": len(sep), "answer": len(ans) - tail,
            "termination": tail, "turn_end": 1, "trailing_newline": 1, "total_supervised": len(t), "has_separator": bool(sep),
            "stock_mlx_extra_pad_targets": 1}


def main():
    out = sys.argv[1]
    rep = {}
    for p in sys.argv[2:]:
        rows = [json.loads(l) for l in open(p) if l.strip()]
        per = [seg(r) for r in rows]
        tot = collections.Counter()
        for d in per:
            for k in ("reasoning", "separator", "answer", "termination", "turn_end", "trailing_newline", "stock_mlx_extra_pad_targets", "total_supervised"):
                tot[k] += d[k]
        by_arm = collections.defaultdict(collections.Counter)
        for r, d in zip(rows, per):
            by_arm[r["meta"].get("arm")]["n"] += 1
            for k in ("reasoning", "separator", "answer", "termination", "total_supervised"):
                by_arm[r["meta"].get("arm")][k] += d[k]
        share = [(d["answer"] + d["termination"] + d["separator"]) / d["total_supervised"] for d in per]
        rep[p] = {"n_rows": len(rows), "totals": dict(tot),
                  "answer_plus_sep_plus_term_share": (tot["answer"] + tot["separator"] + tot["termination"]) / tot["total_supervised"],
                  "answer_share": tot["answer"] / tot["total_supervised"], "reasoning_share": tot["reasoning"] / tot["total_supervised"],
                  "per_row_non_reasoning_share": {"mean": statistics.mean(share), "median": statistics.median(share), "min": min(share), "max": max(share)},
                  "rows_without_separator": sum(not d["has_separator"] for d in per),
                  "by_arm": {str(a): dict(c) for a, c in by_arm.items()},
                  "prompt_tokens": {"max": max(d["prompt"] for d in per), "mean": statistics.mean(d["prompt"] for d in per)},
                  "supervised_tokens": {"max": max(d["total_supervised"] for d in per), "mean": statistics.mean(d["total_supervised"] for d in per)}}
    json.dump(rep, open(out, "w"), indent=1)
    print(json.dumps(rep, indent=1))


main()
