"""Round 5 A2: mlx_lm.lora with an optional segment-weighted loss (everything else identical to `python -m mlx_lm lora`).
usage: train_mlx_weighted.py --mode {plain,weighted,selftest} <usual mlx_lm lora args...>
 plain    : mlx_lm default loss (S0 control; identical numerics to stock mlx_lm lora)
 weighted : L = W_ANS * mean(CE over answer tokens incl. the <channel|> separator and turn-end tokens) + (1-W_ANS) * mean(CE over reasoning tokens),
            per sequence (batch_size 1). A segment with no tokens contributes nothing and the other term is renormalized to weight 1.
 selftest : on the first rows, check shifted-token masks and that weighted loss with weights proportional to segment sizes equals the default loss.
Segment boundary = first target token equal to the <channel|> id at/after the supervised offset; reasoning = targets strictly before it."""
import sys, functools, json, argparse
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.tuner import trainer as T
import mlx_lm.lora as L

W_ANS = 0.9
END_THINK = None


def seg_masks(batch, lengths):
    inputs, targets = batch[:, :-1], batch[:, 1:]
    steps = mx.arange(1, targets.shape[1] + 1)
    mask = mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])
    is_sep = mx.logical_and(targets == END_THINK, mask)
    # first separator position per row (steps index); rows without one -> all supervised tokens are 'reasoning'
    pos = mx.where(is_sep, steps[None, :], mx.array(10**9))
    first = pos.min(axis=1, keepdims=True)
    ans = mx.logical_and(mask, steps[None, :] >= first)
    rea = mx.logical_and(mask, steps[None, :] < first)
    return inputs, targets, mask, ans, rea


def weighted_loss(model, batch, lengths, w_ans=None):
    w = W_ANS if w_ans is None else w_ans
    inputs, targets, mask, ans, rea = seg_masks(batch, lengths)
    logits = model(inputs)
    ce = nn.losses.cross_entropy(logits, targets).astype(mx.float32)
    na, nr = ans.sum(), rea.sum()
    ca = (ce * ans).sum() / mx.maximum(na, 1)
    cr = (ce * rea).sum() / mx.maximum(nr, 1)
    wa = mx.where(na > 0, w, 0.0); wr = mx.where(nr > 0, 1.0 - w, 0.0)
    loss = (wa * ca + wr * cr) / (wa + wr)
    return loss, mask.sum()


def main():
    global END_THINK
    wp = argparse.ArgumentParser(add_help=False)
    wp.add_argument("--mode", choices=["plain", "weighted"], required=True)
    wa, rest = wp.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    if wa.mode == "weighted":
        L.train = functools.partial(T.train, loss=weighted_loss)
    from transformers import AutoTokenizer
    if "--model" not in rest:
        raise SystemExit("pass --model explicitly (the wrapper needs the tokenizer to resolve <channel|>)")
    model_path = rest[rest.index("--model") + 1]
    END_THINK = AutoTokenizer.from_pretrained(model_path).convert_tokens_to_ids("<channel|>")
    assert isinstance(END_THINK, int) and END_THINK >= 0, "separator id not resolved"
    L.main()


if __name__ == "__main__":
    main()
