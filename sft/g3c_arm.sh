#!/bin/bash
# G3c: combined retrain (batch 1 + batch 2) of the arm with the higher run-1 FK composite (ties to arm A), then the FK screen (run 1, repeat if run 1 meets the threshold). GPU only, $0, nothing on the BFCL 200.
# Approval: ledger A-005 (Sans 2026-11-11: "Approved to all", for "launch the final retrain right after the 200-task test ends, with no pause? it's already approved at up to 36h.") and the G3c pre-registration (docs/DISTILLATION_G3C_COMBINED_PREREGISTRATION.md section 6).
# usage: bash sft/g3c_arm.sh check   (launch conditions and estimate, no GPU, nothing started)   |   G3C_APPROVAL=A-005 bash sft/g3c_arm.sh run
set -u
MODE=${1:-}
case "$MODE" in check|run) ;; *) echo "usage: $0 check|run"; exit 2;; esac
cd /Users/USER/Documents/GitHub/Thesis-wt-training || exit 2
# g3lib.sh pins preflight to the arm chains' worktree (branch thesis/distill-r6-diagnostics); this chain runs from Thesis-wt-training (branch thesis/training, which contains that branch). Same checks, branch name replaced at load time.
if [ "${1:-}" = check ] && [ -n "${G3C_TEST_B_ATT:-}" ]; then      # rehearsal while arm B trains: the "training process / port in use" checks cannot hold now and are dropped, in this mode only
  sed 's#thesis/distill-r6-diagnostics#thesis/training#' sft/g3lib.sh | grep -v 'an MLX training process is running' | grep -v 'port 8080 already in use' > /tmp/g3lib_g3c.$$.sh; source /tmp/g3lib_g3c.$$.sh; rm -f /tmp/g3lib_g3c.$$.sh
else
  sed 's#thesis/distill-r6-diagnostics#thesis/training#' sft/g3lib.sh > /tmp/g3lib_g3c.$$.sh; source /tmp/g3lib_g3c.$$.sh; rm -f /tmp/g3lib_g3c.$$.sh
fi
R6G3=$ABS6/sft/data/distill/r6g3; D6=$ABS6/sft/data/distill/r6g3c
A_ATT=$R6G3/attempts/20261110_062035_run_A.lBKtpA; B_ATT=""; NB=0
for d in $R6G3/attempts/*_run_B.*; do [ -d "$d" ] || continue; if [ -f "$d/G3_B_RUN1.json" ]; then NB=$((NB+1)); B_ATT=$d; elif [ -z "$B_ATT" ]; then B_ATT=$d; fi; done
[ $NB -le 1 ] || { echo "ABORT: two arm B folders hold a run-1 result"; exit 1; }
CHAIN_CAP=140400                         # 39 h = training cap 36 h + conversion, screens and servers
T=sft/r6_tools.py
mkdir -p $D6/attempts
[ "$MODE" = check ] && [ -n "${G3C_TEST_B_ATT:-}" ] && B_ATT=$G3C_TEST_B_ATT      # check mode only: lets the checks run on a stand-in for arm B while it trains
[ "$MODE" = run ] && { [ "${G3C_APPROVAL:-}" = "A-005" ] || abort "G3C_APPROVAL=A-005 is not set"; }
if [ "$MODE" = run ]; then       # nothing new starts after Nov 25 (SOUL.md / G3c prereg section 5)
  $PY -c "import datetime,sys; sys.exit(0 if datetime.datetime.now() < datetime.datetime(2026,11,25,23,59) else 1)" || abort "past Nov 25: no new run starts"
fi
if [ "$MODE" = run ]; then       # launch order (G3c prereg amendment, advisor note 2026-11-11): G4 finished and its outputs verified in a backup first
  [ -n "${G3C_G4_ATT:-}" ] && [ -n "${G3C_BACKUP:-}" ] || abort "G3C_G4_ATT and G3C_BACKUP are not set (this chain starts only from sft/g3c_wait_launch.sh, after G4 COMPLETE and a verified backup)"
  grep -q "G4 COMPLETE" "$G3C_G4_ATT/chain.log" 2>/dev/null || abort "G4 has not completed ($G3C_G4_ATT)"
  [ -f "$G3C_BACKUP/verify.txt" ] && [ ! -s "$G3C_BACKUP/verify.txt" ] && grep -q "G4 COMPLETE" "$G3C_BACKUP/g4_attempt/chain.log" 2>/dev/null || abort "the G4 backup is not verified ($G3C_BACKUP)"
fi
[ -n "$B_ATT" ] && [ -f "$B_ATT/G3_B_RUN1.json" ] && [ -f "$A_ATT/G3_A_RUN1.json" ] || abort "arm A or arm B has no run-1 FK result yet (launch condition 1)"
ARM=$($PY -B - $A_ATT $B_ATT <<'PYEOF'
import json, sys
a = json.load(open(sys.argv[1] + "/G3_A_RUN1.json")); b = json.load(open(sys.argv[2] + "/G3_B_RUN1.json"))
ca = a["run1"]["composite"] if a["run1"]["valid"] else -1; cb = b["run1"]["composite"] if b["run1"]["valid"] else -1
print("B" if cb > ca else "A")
PYEOF
)
[ "$ARM" = A ] || [ "$ARM" = B ] || abort "arm choice failed"
case "$ARM" in A) THINK=off;; B) THINK=on;; esac
FROZEN=$D6/final/$ARM
init_attempt "${MODE}_c_$ARM"
echo "$(date +%T) g3c $MODE arm $ARM start, attempt $ATT"
preflight
date +%s > "$D6/chain_start_epoch"
chain_clock
[ -s "$FROZEN/train.jsonl" ] && [ -s "$FROZEN/valid.jsonl" ] || die "frozen rows for arm $ARM missing in $FROZEN (copy them with sft/g3c_freeze.sh first)"
[ "$(shasum -a 256 $FROZEN/train.jsonl | cut -d' ' -f1)" = "$(grep "^$ARM train " sft/g3c_expected.sha | cut -d' ' -f3)" ] || die "arm $ARM train.jsonl hash differs from sft/g3c_expected.sha"
[ "$(shasum -a 256 $FROZEN/valid.jsonl | cut -d' ' -f1)" = "$(grep "^$ARM valid " sft/g3c_expected.sha | cut -d' ' -f3)" ] || die "arm $ARM valid.jsonl hash differs from sft/g3c_expected.sha"
# provenance gate on the COMBINED rows of the chosen arm, bound to the frozen row files by hash, before ANY training
$PY -B - $D6/final $ARM <<'PYEOF' || die "provenance gate: the combined provenance file must pass and be bound to the frozen rows"
import json, sys, hashlib, pathlib
fz, arm = pathlib.Path(sys.argv[1]), sys.argv[2]
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
d = json.load(open(fz / f"provenance_combined_{arm}.json")); ins = set(d["inputs"].values())
bound = sha(fz / arm / "train.jsonl") in ins and sha(fz / arm / "valid.jsonl") in ins
print(arm, "pass", d["pass"], "bound to frozen rows", bound, "rows", d["n_rows"], "train tasks", d["n_train_tasks"])
sys.exit(0 if d["pass"] and bound else 1)
PYEOF
bt $PY -B sft/g3_train.py --rows $FROZEN/train.jsonl --valid $FROZEN/valid.jsonl --out-dir $ATT/dry --think $THINK --dry-tokenize > $RL/dry_tokenize.log 2>&1 || die "row rendering check failed (see $RL/dry_tokenize.log)"
say "inputs frozen and render-checked: $(tail -2 $RL/dry_tokenize.log | tr '\n' ' ' | cut -c1-300)"

g3_parity() {   # NAME
  bt $PY -B sft/g3_parity.py --rows $FROZEN/train.jsonl --out $ATT/parity_$1.json --think $THINK > $RL/parity_$1.out 2>&1 || die "render parity failed for $1: $(tail -c 400 $RL/parity_$1.out)"
  say "parity ok ($1): $(cat $ATT/parity_$1.json | tr -d '\n ' | cut -c1-240)"
}

# ---------------------------------- run ----------------------------------
BASEREP=$(ls -d $R6G3/attempts/*_base.*/G3_BASE_REPORT.json | tail -1)
cp "$BASEREP" "$ATT/G3_BASE_REPORT.copy.json"
$PY -B - $BASEREP $A_ATT $B_ATT $ARM $ATT $FROZEN/train.jsonl <<'PYEOF' || die "preconditions failed (base reference, arm results, or the estimate is not under 36 h)"
import json, sys, datetime
base, a_att, b_att, arm, att, train = sys.argv[1:7]
assert json.load(open(base))["valid"], "G3 base reference invalid"
ch = a_att if arm == "A" else b_att
rows = sum(1 for l in open(train) if l.strip())
log = [json.loads(l) for l in open(ch + "/train/train_log.jsonl") if '"micro"' in l and '"s"' in l]
mean_s = sum(x["s"] for x in log) / len(log)
est_h = rows * 2 * mean_s / 3600            # 2 passes, one micro-batch per row, mean seconds per micro-batch of the chosen arm's complete first run
rec = {"arm": arm, "date": datetime.datetime.now().isoformat(timespec="seconds"), "run1_A": json.load(open(a_att + "/G3_A_RUN1.json"))["run1"]["composite"], "run1_B": json.load(open(b_att + "/G3_B_RUN1.json"))["run1"]["composite"],
       "combined_train_rows": rows, "micro_batches_in_chosen_arm_log": len(log), "mean_s_per_micro_first_run": round(mean_s, 2), "estimated_training_hours": round(est_h, 2), "limit_hours": 36}
json.dump(rec, open(att + "/G3C_ARM_CHOICE.json", "w"), indent=1); print(json.dumps(rec))
sys.exit(0 if est_h < 36 else 1)
PYEOF
if [ "$MODE" = check ]; then say "CHECK COMPLETE: all conditions that need no GPU hold; arm $ARM; nothing started"; exit 0; fi
lsof -nP -iTCP:8080 -sTCP:LISTEN >/dev/null 2>&1 && abort "port 8080 busy (GPU not free)"
pgrep -f "g3_train.py|mlx_lm|llama-server" >/dev/null && abort "a training or server process is running (GPU not free)"

start_server base$ARM "" $THINK
check_identity base$ARM 1.0
g3_parity base
stop_server

TRAIN_T0=$(date +%s)
bt timeout -k 60 $(min2 129600 "$(chain_left)") $PY -B sft/g3_train.py --rows $FROZEN/train.jsonl --valid $FROZEN/valid.jsonl --out-dir $ATT/train --think $THINK --max-seconds $(min2 129600 "$(chain_left)") > $RL/train.out 2>&1
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
say "G3C ARM $ARM COMPLETE: $ATT"
say "REQUIRED NEXT (Thesis bot, not the chain): report counts vs the thinking-on base, row counts, dropped rows, pass-1 checkpoint path, Codex review of the result"
