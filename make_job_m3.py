"""Build one chained M2 hop job JSON by cloning jobs_yym2/h3m2_s2.json.
Usage: python make_job_m2.py <seg N> <anchor_png_name_in_shared_input> <spec_file> [seed]
Writes jobs_yym2/h3m2_sN.json. POV preamble is preserved verbatim from the S2 template."""
import json, sys
from pathlib import Path

BASE = Path(r"C:\Users\yunlei\.zcode\workspace\default\h3_map_baseline")
TEMPLATE = BASE / "jobs_yym2" / "h3m2_s2.json"

seg = int(sys.argv[1])
anchor = sys.argv[2]           # e.g. m2_s2_last.png (must already be in Shared input)
spec_file = BASE / sys.argv[3] # e.g. hop_specs_m2/s3.txt
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 42

j = json.load(open(TEMPLATE))
g = j["prompt"]

template_prompt = g["17"]["inputs"]["prompt"]
preamble = template_prompt.split("The ride continues")[0]  # POV lock block, exact
spec = spec_file.read_text(encoding="utf-8").strip()

g["16"]["inputs"]["image"] = anchor
g["17"]["inputs"]["prompt"] = preamble + spec
g["15"]["inputs"]["filename_prefix"] = f"h3m3_s{seg}"

# keep every seed knob identical to template unless overridden (K_sampler seed lives in unet sampler node)
def set_seeds(node, val):
    for n in g.values():
        inp = n.get("inputs", {})
        for k in ("seed", "noise_seed"):
            if k in inp and isinstance(inp[k], int):
                inp[k] = val
set_seeds(None, seed)

out = BASE / "jobs_yym2" / f"h3m3_s{seg}.json"
json.dump(j, open(out, "w"), ensure_ascii=False, indent=1)
print("wrote", out, "| anchor:", anchor, "| spec:", spec_file.name, "| seed:", seed)
print("prefix:", g["15"]["inputs"]["filename_prefix"])
