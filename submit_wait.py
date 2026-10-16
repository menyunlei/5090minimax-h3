# Submit job JSONs to ComfyUI /prompt, then wait until all expected mp4s exist.
# Usage: python submit_wait.py job1.json [job2.json ...]
import json, sys, time, urllib.request
from pathlib import Path

OUT_DIR = Path(r"C:\Users\yunlei\AppData\Local\Comfy-Desktop\ComfyUI-Shared\output")
API = "http://127.0.0.1:8188/prompt"

def post(p):
    data = p.read_bytes()
    req = urllib.request.Request(API, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())

files = [Path(a) for a in sys.argv[1:]]
expected = {}
for fp in files:
    wf = json.loads(fp.read_text())
    prefix = None
    for node in wf["prompt"].values():
        if node.get("class_type") == "SaveVideo":
            prefix = node["inputs"]["filename_prefix"]
    assert prefix, fp
    expected[fp.name] = OUT_DIR / f"{prefix}_00001_.mp4"

for fp in files:
    resp = post(fp)
    print("submitted", fp.name, "->", resp.get("prompt_id", "?"), flush=True)

deadline = time.time() + 2400
pending = dict(expected)
while pending and time.time() < deadline:
    for name, out in list(pending.items()):
        if out.exists() and out.stat().st_size > 100000:
            print("DONE", out.name, f"{out.stat().st_size/1e6:.1f}MB", flush=True)
            del pending[name]
    if pending:
        time.sleep(8)
if pending:
    print("TIMEOUT waiting for:", ", ".join(pending))
    sys.exit(1)
print("ALL COMPLETE")
