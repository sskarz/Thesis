#!/usr/bin/env python3
"""Offline self-test for r6_self_targets.py using a fake llama-server (no model)."""
import json
import pathlib
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

HERE = pathlib.Path(__file__).resolve().parent
STATE = {"adapters": [], "chat": []}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj):
        b = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/lora-adapters":
            self._send(STATE["adapters"])
        elif self.path == "/props":
            self._send({"model_path": "/x/google_gemma-4-E2B-it-Q4_K_M.gguf", "build_info": "fake"})
        else:
            self.send_error(404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/apply-template":
            sysmsg = body["messages"][0]["content"]
            self._send({"prompt": "<|think|>\n" + sysmsg})  # template adds exactly one marker
        elif self.path == "/v1/chat/completions":
            STATE["chat"].append(body)
            last = body["messages"][-1]["content"]
            if "EMPTY" in last:
                msg, fin = {"content": "", "reasoning_content": "r"}, "stop"
            elif "LONG" in last:
                msg, fin = {"content": "[]", "reasoning_content": "r"}, "length"
            else:
                msg, fin = {"content": ' [{"function_name": "ls", "parameters": {}}] ', "reasoning_content": " thinking about " + last[:10] + " "}, "stop"
            self._send({"choices": [{"message": msg, "finish_reason": fin}]})
        else:
            self.send_error(404)


def row(user, tgt="old"):
    return {"messages": [{"role": "system", "content": "<|think|>\nSYS"}, {"role": "user", "content": user},
                         {"role": "assistant", "content": tgt, "reasoning_content": "old reasoning"}], "meta": {"id": user}}


def main() -> None:
    srv = HTTPServer(("127.0.0.1", 0), H)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    with tempfile.TemporaryDirectory() as d:
        d = pathlib.Path(d)
        src = d / "src"
        src.mkdir()
        (src / "train.jsonl").write_text("\n".join(json.dumps(r) for r in [row("alpha"), row("EMPTY x"), row("LONG y"), row("beta")]) + "\n")
        (src / "valid.jsonl").write_text(json.dumps(row("gamma")) + "\n")
        cmd = [sys.executable, str(HERE / "r6_self_targets.py"), "--rows-dir", str(src), "--out-dir", str(d / "out"), "--url", f"http://127.0.0.1:{port}"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr + r.stdout
        tr = [json.loads(l) for l in open(d / "out" / "train.jsonl")]
        assert len(tr) == 2, len(tr)
        for x in tr:
            assert x["messages"][0]["content"] == "<|think|>\nSYS", "marker not restored"
            assert x["messages"][-1]["content"].startswith("[{") and "old" not in x["messages"][-1]["reasoning_content"]
            assert x["meta"]["r6_self_target"] is True
        for body in STATE["chat"]:
            assert body["temperature"] == 0.0 and "max_tokens" not in body
            assert not body["messages"][0]["content"].startswith("<|think|>"), "marker not stripped for the request"
        m = json.load(open(d / "out" / "manifest.json"))
        assert m["stats"]["train"]["dropped_not_stop"] == 1 and m["stats"]["train"]["dropped_empty_content"] == 1
        assert m["stats"]["train"]["kept"] == 2 and m["stats"]["valid"]["kept"] == 1
        assert m["inputs"]["train"]["rows"] == 4
        # refuses to write into the source dir, and refuses a server with adapters loaded
        assert subprocess.run(cmd[:-4] + ["--out-dir", str(src), "--url", f"http://127.0.0.1:{port}"], capture_output=True, text=True).returncode != 0
        STATE["adapters"] = [{"id": 0, "path": "x.gguf", "scale": 1.0}]
        r2 = subprocess.run(cmd[:-2] + ["--out-dir", str(d / "out2"), "--url", f"http://127.0.0.1:{port}"], capture_output=True, text=True)
        assert r2.returncode != 0 and "adapters loaded" in (r2.stderr + r2.stdout)
    print("test_r6_self_targets: OK")


if __name__ == "__main__":
    main()
