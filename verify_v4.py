"""Verify v4 joins numerically + straddling sheet. Joins (0-index) at 243, 486, 729, 972
(assuming 243 frames per segment with anchor-drop)."""
import av
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path

VID = Path(r"C:\Users\yunlei\.zcode\workspace\default\h3_map_baseline\continuous_yym_v4_mapeveryhop.mp4")
OUT = Path(r"C:\Users\yunlei\.zcode\workspace\default\h3_map_baseline\yym3_diag\v4_joins_sheet.png")
JOINS = [243, 485, 727, 969]

need = set()
for j in JOINS:
    need.update([j - 2, j - 1, j, j + 1])
keep = {}
prev = None
global_diffs = []
with av.open(str(VID)) as c:
    idx = -1
    for f in c.decode(c.streams.video[0]):
        idx += 1
        if idx in need:
            keep[idx] = f.to_ndarray(format="rgb24")
        arr = keep.get(idx)
        if idx % 37 == 25 and prev is not None and arr is not None:
            global_diffs.append(float(np.abs(arr.astype(np.int16) - prev.astype(np.int16)).mean()))
        if arr is not None:
            prev = arr

if len(keep) < len(need):
    print("WARN: fewer frames than expected", len(keep), "of", len(need))
base = float(np.median(global_diffs)) if global_diffs else float("nan")
print(f"baseline median interframe diff (sampled): {base:.2f}/255")
for j in JOINS:
    if j - 1 not in keep or j not in keep:
        print(f"join {j}: frame missing"); continue
    d = float(np.abs(keep[j].astype(np.int16) - keep[j - 1].astype(np.int16)).mean())
    print(f"join at frame {j}: diff {d:.2f}/255  ({'OK' if d < base * 0.8 else 'LARGER THAN MOTION'})")

rows = []
for j in JOINS:
    if j not in keep:
        continue
    tiles = []
    for k in [j - 2, j - 1, j, j + 1]:
        im = Image.fromarray(keep[k]).resize((448, 256), Image.LANCZOS)
        dd = ImageDraw.Draw(im)
        dd.rectangle([0, 0, 96, 16], fill=(0, 0, 0))
        dd.text((4, 2), f"f{k}", fill=(255, 230, 80))
        tiles.append(im)
    row = Image.new("RGB", (448 * 4 + 10, 256), (30, 30, 30))
    for i, t in enumerate(tiles):
        row.paste(t, (i * 450, 0))
    rows.append(row)
if rows:
    sheet = Image.new("RGB", (448 * 4 + 10, 256 * len(rows) + 8 * len(rows)), (20, 20, 20))
    for i, r in enumerate(rows):
        sheet.paste(r, (0, i * 264))
    sheet.save(OUT)
    print("saved", OUT)
