# Continuity analysis of the compiled 12-segment ride.
# 1) interframe abs-diff at the 11 joins vs the global median
# 2) contact sheet: 1 frame before + 1 frame after each join (halved scale)
import av
import numpy as np
from PIL import Image, ImageDraw

VID = r"C:\Users\yunlei\.zcode\workspace\default\h3_map_baseline\continuous_yym_60s.mp4"
DST = r"C:\Users\yunlei\.zcode\workspace\default\h3_map_baseline\yym2_diag"

lens = [124] + [123] * 11
bounds = []
acc = 0
for L in lens[:-1]:
    acc += L
    bounds.append(acc - 1)  # last frame of seg k in output indexing

c = av.open(VID)
prev, diffs, keep = None, [], {}
want = set()
for b in bounds:
    want.update([b, b + 1])
for i, f in enumerate(c.decode(video=0)):
    a = f.to_ndarray(format="rgb24")
    if prev is not None:
        diffs.append(float(np.abs(a.astype(np.int16) - prev).mean()))
    if i in want:
        keep[i] = Image.fromarray(a).resize((560, 320), Image.LANCZOS)
    prev = a.astype(np.int16)
    if i > max(want):
        break
c.close()

med = float(np.median(diffs))
print(f"global median interframe diff: {med:.2f}/255  (n={len(diffs)})")
worst = 0.0
for k, b in enumerate(bounds, 1):
    d = diffs[b] if b < len(diffs) else float('nan')
    worst = max(worst, d)
    tag = "JOIN" if d > med * 2.5 else "ok  "
    print(f"join {k:2d} (out frame {b}->{b+1}): diff {d:.2f}/255  [{tag}]")
print(f"worst join diff: {worst:.2f}/255")

# contact sheet around joins
n = len(bounds)
sheet = Image.new("RGB", (560 * 2, 320 * n), (8, 8, 8))
dr = ImageDraw.Draw(sheet)
for k, b in enumerate(bounds):
    a, bb = keep.get(b), keep.get(b + 1)
    if a: sheet.paste(a, (0, 320 * k))
    if bb: sheet.paste(bb, (560, 320 * k))
    dr.text((8, 320 * k + 8), f"join {k+1} BEFORE", fill=(255, 255, 0))
    dr.text((560 + 8, 320 * k + 8), f"join {k+1} AFTER", fill=(120, 255, 140))
sheet.save(DST + r"\continuous_joins_sheet.jpg", quality=88)
print("saved joins sheet")
