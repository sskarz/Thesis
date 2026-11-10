#!/bin/bash
# P0.2: which serving setting renders zero <|think|> markers and yields replies without a thought channel? (read-only probe, a few minutes of GPU, $0)
# usage: bash sft/p02_thinking_off.sh OUTDIR     Starts and stops its OWN llama-server on :8080; aborts if the port is busy.
set -u
OUT=${1:?outdir}; mkdir -p "$OUT"
R5=/Users/USER/Documents/GitHub/Thesis-wt-distill
BAR=/Users/USER/.cache/huggingface/hub/models--bartowski--google_gemma-4-E2B-it-GGUF/snapshots/81012ba3538e061d5ee003f11f25335b17f82e2d/google_gemma-4-E2B-it-Q4_K_M.gguf
LLAMA=/Users/USER/Documents/GitHub/Thesis/llama.cpp/build/bin/llama-server
PY=/Users/USER/Documents/GitHub/Thesis/.venv/bin/python
if lsof -nP -iTCP:8080 -sTCP:LISTEN > /dev/null 2>&1; then echo "port 8080 busy; not touching it"; exit 1; fi
PID=""
cleanup() { [ -n "$PID" ] && kill "$PID" 2>/dev/null; sleep 2; [ -n "$PID" ] && kill -9 "$PID" 2>/dev/null; true; }
trap cleanup EXIT
for variant in default kwargs budget0; do
  case $variant in
    default) EXTRA="" ;;
    kwargs)  EXTRA="--chat-template-kwargs {\"enable_thinking\":false}" ;;
    budget0) EXTRA="--reasoning-budget 0" ;;
  esac
  nohup $LLAMA --jinja -fa on -m "$BAR" $EXTRA --host 127.0.0.1 --port 8080 -ngl 99 -c 65536 --temp 0.0 -np 4 -cb -ub 1024 > "$OUT/server_$variant.log" 2>&1 &
  PID=$!
  for _ in $(seq 1 90); do kill -0 $PID 2>/dev/null || break; curl -s --max-time 5 http://127.0.0.1:8080/health | grep -q '"ok"' && break; sleep 2; done
  if ! curl -s --max-time 5 http://127.0.0.1:8080/health | grep -q '"ok"'; then echo "$variant: server did not start (flag rejected?)"; tail -3 "$OUT/server_$variant.log"; cleanup; PID=""; sleep 2; continue; fi
  $PY -B - "$variant" "$OUT" "$R5" <<'PYEOF'
import json, sys, requests
variant, out, r5 = sys.argv[1:4]
rows = [json.loads(l) for l in open(f"{r5}/sft/data/distill/r5/mlx-AB-think-clean/valid.jsonl")][:5]
res = {"variant": variant, "props_chat_template_kwargs": None, "requests": []}
props = requests.get("http://127.0.0.1:8080/props", timeout=10).json()
res["build"] = props.get("build_info")
for r in rows:
    msgs = r["messages"][:-1]
    m0 = dict(msgs[0]); m0["content"] = m0["content"].replace("<|think|>\n", "", 1)       # same strip as the Round 5 probe
    msgs = [m0] + msgs[1:]
    ap = requests.post("http://127.0.0.1:8080/apply-template", json={"messages": msgs}, timeout=30).json()["prompt"]
    c = requests.post("http://127.0.0.1:8080/v1/chat/completions", json={"messages": msgs, "temperature": 0, "max_tokens": 600}, timeout=300).json()["choices"][0]
    msg = c["message"]
    res["requests"].append({"think_markers_in_rendered_prompt": ap.count("<|think|>"), "finish_reason": c.get("finish_reason"),
                            "has_reasoning_content": bool((msg.get("reasoning_content") or "").strip()),
                            "content_has_thought_channel": "<|channel>thought" in (msg.get("content") or ""),
                            "content_chars": len(msg.get("content") or ""), "reasoning_chars": len(msg.get("reasoning_content") or ""),
                            "content_is_bare_json_list": (msg.get("content") or "").strip().startswith("[")})
json.dump(res, open(f"{out}/result_{variant}.json", "w"), indent=1)
q = res["requests"]
print(variant, {"markers": [x["think_markers_in_rendered_prompt"] for x in q], "reasoning": [x["has_reasoning_content"] for x in q],
                "thought_channel_in_content": [x["content_has_thought_channel"] for x in q], "bare_list": [x["content_is_bare_json_list"] for x in q],
                "finish": [x["finish_reason"] for x in q]})
PYEOF
  cleanup; PID=""; sleep 3
done
echo "p02 done"
