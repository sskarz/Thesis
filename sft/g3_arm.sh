#!/bin/bash
# G3 arm chain (SANS APPROVED 2026-11-10 "Approve for all 3"; plan G3). usage: bash sft/g3_arm.sh smoke A|B   |   bash sft/g3_arm.sh run A|B
# Arm A = direct answer, thinking OFF (rows_clean: JSON call list target). Arm B = short reasoning first, thinking ON (rows_B: teacher sentences as reasoning_content, then the call list).
# Same tasks and the same task-disjoint train/valid split (batch 1, Thesis-wt-data attempt 20261109_212657_g2full.AUhok0), adapter as G1 (layers 8-34, rank 16, lr 1e-4, warm-up + decay, prompt mask),
# 2 passes, rows over 6,999 tokens dropped and counted, checkpoint at the end of pass 1 (kept, not scored), render parity against the server in compact-list mode.
# smoke: 12 micro-batches -> adapter -> GGUF -> audit -> server with the adapter -> identity -> parity (not a result; measures s per micro-batch and peak memory).
# run  : verify frozen rows and the G2/provenance/base-reference preconditions -> base-server parity -> train -> convert + audit -> adapter server -> parity -> FK screen run 1
#        -> (only if run 1 meets the registered threshold) a restarted repeat -> report. GPU only, $0. Whole chain capped at 34 h; training capped at 30 h.
set -u
MODE=${1:-}; ARM=${2:-}
case "$MODE" in smoke|run) ;; *) echo "usage: $0 smoke|run A|B"; exit 2;; esac
case "$ARM" in A) THINK=off; SRC=rows_clean;; B) THINK=on; SRC=rows_B;; *) echo "usage: $0 smoke|run A|B"; exit 2;; esac
cd /Users/sanskarthapa/Documents/GitHub/Thesis-wt-r6 || exit 2
source sft/g3lib.sh
D6=$ABS6/sft/data/distill/r6g3
B1=/Users/sanskarthapa/Documents/GitHub/Thesis-wt-data/sft/data/distill/yield/attempts/20261109_212657_g2full.AUhok0
FROZEN=$D6/final/$ARM
if [ "$MODE" = smoke ]; then CHAIN_CAP=5400; else CHAIN_CAP=122400; fi
T=sft/r6_tools.py
init_attempt "${MODE}_$ARM"
echo "$(date +%T) g3 $MODE arm $ARM start, attempt $ATT"
preflight
date +%s > "$D6/chain_start_epoch"      # each arm chain owns its own clock (the base-reference chain wrote an earlier one)
chain_clock
[ -s "$FROZEN/train.jsonl" ] && [ -s "$FROZEN/valid.jsonl" ] || die "frozen rows for arm $ARM missing in $FROZEN (copy them with sft/g3_freeze.sh first)"
[ "$(shasum -a 256 $FROZEN/train.jsonl | cut -d' ' -f1)" = "$(grep "^$ARM train " sft/g3_expected.sha | cut -d' ' -f3)" ] || die "arm $ARM train.jsonl hash differs from sft/g3_expected.sha"
[ "$(shasum -a 256 $FROZEN/valid.jsonl | cut -d' ' -f1)" = "$(grep "^$ARM valid " sft/g3_expected.sha | cut -d' ' -f3)" ] || die "arm $ARM valid.jsonl hash differs from sft/g3_expected.sha"
# provenance gate on BOTH arms, bound to the frozen row files by hash, before ANY training (smoke included)
$PY -B - $B1 $FROZEN <<'PYEOF' || die "provenance gate: both final provenance files must pass and be bound to the frozen rows"
import json, sys, hashlib, pathlib
b1, fz = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]).parent
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
ok = True
for arm in "AB":
    d = json.load(open(b1 / f"provenance_final_{arm}.json")); ins = set(d["inputs"].values())
    bound = sha(fz / arm / "train.jsonl") in ins and sha(fz / arm / "valid.jsonl") in ins
    print(arm, "pass", d["pass"], "bound to frozen rows", bound); ok &= bool(d["pass"]) and bound
sys.exit(0 if ok else 1)
PYEOF
bt $PY -B sft/g3_train.py --rows $FROZEN/train.jsonl --valid $FROZEN/valid.jsonl --out-dir $ATT/dry --think $THINK --dry-tokenize > $RL/dry_tokenize.log 2>&1 || die "row rendering check failed (see $RL/dry_tokenize.log)"
say "inputs frozen and render-checked: $(tail -2 $RL/dry_tokenize.log | tr '\n' ' ' | cut -c1-300)"

g3_parity() {   # NAME
  bt $PY -B sft/g3_parity.py --rows $FROZEN/train.jsonl --out $ATT/parity_$1.json --think $THINK > $RL/parity_$1.out 2>&1 || die "render parity failed for $1: $(tail -c 400 $RL/parity_$1.out)"
  say "parity ok ($1): $(cat $ATT/parity_$1.json | tr -d '\n ' | cut -c1-240)"
}

if [ "$MODE" = smoke ]; then
  lsof -nP -iTCP:8080 -sTCP:LISTEN >/dev/null 2>&1 && abort "port 8080 busy"
  bt $PY -B sft/g3_train.py --rows $FROZEN/train.jsonl --out-dir $ATT/train --think $THINK --smoke 12 > $RL/train.out 2>&1 || die "smoke training failed: $(tail -c 400 $RL/train.out)"
  say "train: $(tail -2 $RL/train.out | tr '\n' ' ' | cut -c1-300)"
  $PY -B - $ATT/train/train_log.jsonl $FROZEN/train.jsonl <<'PYEOF'
import json, sys
L = [json.loads(l) for l in open(sys.argv[1]) if '"micro"' in l and '"s"' in l]
n_rows = sum(1 for l in open(sys.argv[2]) if l.strip())
mean_s = sum(x["s"] for x in L) / len(L); mean_t = sum(x["tokens"] for x in L) / len(L)
print(json.dumps({"micro_batches": len(L), "mean_s": round(mean_s, 2), "mean_tokens": round(mean_t), "s_per_1k_tokens": round(1000 * sum(x["s"] for x in L) / sum(x["tokens"] for x in L), 2), "losses": [x["loss"] for x in L]}))
print("rows", n_rows)
PYEOF
  bt $PY -B sft/mlx_adapter_to_gguf_lora.py $ATT/train/adapters $ATT/smoke.gguf > $RL/convert.log 2>&1 || die "convert failed"
  bt $PY -B $T gguf-check --adapter-dir $ATT/train/adapters --src-dir $ATT/train/adapters --gguf $ATT/smoke.gguf || die "gguf audit failed"
  start_server smoke$ARM $ATT/smoke.gguf $THINK
  check_identity smoke$ARM 1.0 $ATT/smoke.gguf
  g3_parity smoke
  stop_server
  say "SMOKE COMPLETE (not a result): $ATT"
  exit 0
fi

# ---------------------------------- run ----------------------------------
$PY -B - $B1 $ARM <<'PYEOF' || die "preconditions failed (G2 floors, provenance, or the G3 base reference report)"
import json, sys, glob, pathlib
b1, arm = pathlib.Path(sys.argv[1]), sys.argv[2]
g2 = json.load(open(b1 / "G2_REPORT.json")); prov = json.load(open(b1 / ("provenance_final_A.json" if arm == "A" else "provenance_final_B.json")))
base = sorted(glob.glob("/Users/sanskarthapa/Documents/GitHub/Thesis-wt-r6/sft/data/distill/r6g3/attempts/*_base.*/G3_BASE_REPORT.json"))
ok = bool(g2["floor_met"]) and bool(prov["pass"]) and bool(base) and json.load(open(base[-1]))["valid"]
print(json.dumps({"floor_met": g2["floor_met"], "surviving_tasks": g2["surviving_tasks"], "provenance_pass": prov["pass"], "base_report": base[-1] if base else None}))
sys.exit(0 if ok else 1)
PYEOF
BASEREP=$(ls -d $D6/attempts/*_base.*/G3_BASE_REPORT.json | tail -1)
cp "$BASEREP" "$ATT/G3_BASE_REPORT.copy.json"

start_server base$ARM "" $THINK
check_identity base$ARM 1.0
g3_parity base
stop_server

TRAIN_T0=$(date +%s)
bt timeout -k 60 $(min2 108000 "$(chain_left)") $PY -B sft/g3_train.py --rows $FROZEN/train.jsonl --valid $FROZEN/valid.jsonl --out-dir $ATT/train --think $THINK --max-seconds $(min2 108000 "$(chain_left)") > $RL/train.out 2>&1
RC=$?
[ $RC -eq 0 ] || die "training failed (rc $RC; 4 non-finite loss, 5 diverged, 6 time cap): $(tail -c 300 $RL/train.out)"
[ -f $ATT/train/adapters/adapters.safetensors ] || die "no adapter was written"
say "training done: $(tail -1 $RL/train.out | cut -c1-300)"
bt $PY -B sft/mlx_adapter_to_gguf_lora.py $ATT/train/adapters $ATT/G3$ARM.gguf > $RL/convert.log 2>&1 || die "convert failed"
bt $PY -B $T gguf-check --adapter-dir $ATT/train/adapters --src-dir $ATT/train/adapters --gguf $ATT/G3$ARM.gguf || die "GGUF audit failed"

start_server G3$ARM $ATT/G3$ARM.gguf $THINK
check_identity G3${ARM}run1 1.0 $ATT/G3$ARM.gguf
g3_parity adapter
fk6 G3${ARM}1
stop_server
$PY -B - $ATT G3${ARM}1 $ARM <<'PYEOF'
import json, sys, pathlib, math
a, name, arm = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
base = json.load(open(a / "G3_BASE_REPORT.copy.json")); thr = base["threshold"]["run1_needs_at_least"]; mean = base["threshold"]["mean_thinking_on"]
def comp(n):
    r = 0; parts = {}
    for bank, cnt in (("F", 81), ("K2", 74)):
        d = a / "evals" / f"{bank}_{n}"; v = [json.loads(l) for l in open(d / "verdicts.jsonl")]
        valid = len(v) == cnt and len({x["task_id"] for x in v}) == cnt and not any(x.get("api_error") for x in v)
        dm = json.load(open(d / "report.json"))["decision_mix"]
        parts[bank] = {"passes": sum(bool(x["valid"]) for x in v), "valid": valid, "outputs": dm["assistant_outputs"], "unparsed": dm["unparsed_outputs"]}
    return parts
p = comp(name); c = p["F"]["passes"] + p["K2"]["passes"]; ok = p["F"]["valid"] and p["K2"]["valid"]
out = {"arm": arm, "run1": {"parts": p, "composite": c, "valid": ok}, "threshold_run1": thr, "base_mean_thinking_on": mean, "run1_meets_threshold": bool(ok and c >= thr)}
json.dump(out, open(a / f"G3_{arm}_RUN1.json", "w"), indent=1); print(json.dumps(out))
sys.exit(0 if out["run1_meets_threshold"] else 3)
PYEOF
R1=$?
if [ $R1 -eq 0 ]; then
  say "run 1 MEETS the registered threshold; running the restarted repeat"
  start_server G3${ARM}b $ATT/G3$ARM.gguf $THINK
  check_identity G3${ARM}run2 1.0 $ATT/G3$ARM.gguf
  fk6 G3${ARM}2
  stop_server
  $PY -B - $ATT G3${ARM}2 $ARM <<'PYEOF'
import json, sys, pathlib, math
a, name, arm = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
base = json.load(open(a / "G3_BASE_REPORT.copy.json")); mean = base["threshold"]["mean_thinking_on"]
need = math.ceil(mean + 0.06 * 155)
tot = 0; ok = True
for bank, cnt in (("F", 81), ("K2", 74)):
    v = [json.loads(l) for l in open(a / "evals" / f"{bank}_{name}" / "verdicts.jsonl")]
    ok &= len(v) == cnt and len({x["task_id"] for x in v}) == cnt and not any(x.get("api_error") for x in v)
    tot += sum(bool(x["valid"]) for x in v)
r1 = json.load(open(a / f"G3_{arm}_RUN1.json"))
out = {**r1, "run2": {"composite": tot, "valid": ok}, "repeat_needs_at_least": need, "repeat_meets": bool(ok and tot >= need)}
json.dump(out, open(a / f"G3_{arm}_REPORT.json", "w"), indent=1); print(json.dumps(out))
PYEOF
else
  cp $ATT/G3_${ARM}_RUN1.json $ATT/G3_${ARM}_REPORT.json 2>/dev/null
  say "run 1 did NOT meet the registered threshold (or was invalid); no repeat"
fi
say "G3 ARM $ARM COMPLETE: $ATT"
say "REQUIRED NEXT (Thesis bot, not the chain): report counts vs the thinking-on base, row counts, dropped rows, pass-1 checkpoint path, Codex review of the result"
