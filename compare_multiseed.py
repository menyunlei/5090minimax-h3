# Multiseed frame-diff analysis for the no-legend H3 map->video ablation.
# Computes: per case, mean orig-vs-edit diff across 3 seeds (+std);
# per video, diff vs the no-ref baseline (legend _00001_ control).
import av, numpy as np, os, glob, re

OUT = r"C:\Users\yunlei\AppData\Local\Comfy-Desktop\ComfyUI-Shared\output"
CASES = ["C1", "C2", "C3", "C4"]
SEEDS = ["s1", "s2", "s3"]
IDXS = {0, 60, 120}


def frames(path):
    c = av.open(path); s = c.streams.video[0]; out = {}
    for i, f in enumerate(c.decode(s)):
        if i in IDXS: out[i] = f.to_ndarray(format="rgb24").astype(np.float32)
        if i > max(IDXS): break
    c.close(); return out


def find(prefix, tag):
    matches = glob.glob(os.path.join(OUT, f"{prefix}_{tag}_*.mp4"))
    if not matches: return None
    return sorted(matches)[-1]  # latest version


# no-ref baseline (from the wiring-bug era; legend _00001_ control)
base_path = sorted(glob.glob(os.path.join(OUT, "h3map_C1_orig_00001_.mp4")))[-1]
base = frames(base_path)
print(f"no-ref baseline: {os.path.basename(base_path)}\n")

# load all 24 videos
data = {}
for c in CASES:
    for v in ["orig", "edit"]:
        for s in SEEDS:
            tag = f"{c}_{v}_{s}"
            p = find("h3mapNL", tag)
            if p is None:
                print(f"MISSING: {tag}"); continue
            data[tag] = frames(p)

# per-case orig-vs-edit across seeds
print("=== orig vs edit, per case x seed (mean abs px diff /255) ===")
print(f"{'case':>5} {'seed':>4} {'diff':>7}")
for c in CASES:
    diffs = []
    for s in SEEDS:
        o, e = f"{c}_orig_{s}", f"{c}_edit_{s}"
        if o in data and e in data:
            d = np.mean([np.mean(np.abs(data[o][i] - data[e][i])) for i in IDXS])
            diffs.append(d)
            print(f"{c:>5} {s:>4} {d:>7.1f}")
    if diffs:
        print(f"{c:>5} {'avg':>4} {np.mean(diffs):>7.1f}  (std {np.std(diffs):.1f})")
    print()

# per-video diff vs no-ref baseline (proves reference reached the model)
print("=== each video vs no-ref baseline (mean abs px diff /255) ===")
for c in CASES:
    row = f"{c}:"
    for v in ["orig", "edit"]:
        for s in SEEDS:
            tag = f"{c}_{v}_{s}"
            if tag in data:
                d = np.mean([np.mean(np.abs(data[tag][i] - base[i])) for i in IDXS])
                row += f"  {v}_{s}={d:.0f}"
    print(row)
