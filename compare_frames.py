# Frame-level comparison of the H3 map-baseline videos.
import glob, av, numpy as np, os

OUT = r"C:\Users\yunlei\AppData\Local\Comfy-Desktop\ComfyUI-Shared\output"
files = sorted(glob.glob(os.path.join(OUT, "h3map_*.mp4")))

def frames(path, idxs):
    c = av.open(path)
    s = c.streams.video[0]
    out = {}
    for i, frame in enumerate(c.decode(s)):
        if i in idxs:
            out[i] = frame.to_ndarray(format="rgb24").astype(np.float32)
        if i > max(idxs):
            break
    c.close()
    return out

idxs = {0, 60, 120}
data = {}
for f in files:
    name = os.path.basename(f).replace("h3map_", "").replace("_00001_.mp4", "")
    data[name] = frames(f, idxs)

names = list(data.keys())
print("pairs: mean abs pixel diff (0-255), frames 0/60/120")
hdr = " " * 10 + "".join(f"{n[:9]:>10}" for n in names)
print(hdr)
for a in names:
    row = f"{a[:9]:>10}"
    for b in names:
        if a == b:
            row += f"{'-':>10}"
        else:
            diffs = [np.mean(np.abs(data[a][i] - data[b][i])) for i in idxs]
            row += f"{np.mean(diffs):>10.1f}"
    print(row)
