"""Verify v3 joins numerically + build a straddling-frame sheet around each join.
Joins land at cum frames 243, 485, 727, 969, 1211 (0-indexed: first frame of next seg)."""
import av
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path

VID = Path(r"C:\Users\yunlei\.zcode\workspace\default\h3_map_baseline\continuous_m3_v1.mp4")
OUT = Path(r"C:\Users\yunlei\.zcode\workspace\default\h3_map_baseline\yym3_diag\m3_joins_sheet.png")
JOINS = [243, 485, 727, 969, 1211]  # first frame index AFTER each dropped anchor

# decode all frames index->arr (1453 frames @1344x768 is heavy; keep only needed ones)
need = set()
for j in JOINS:
    need.update([j - 2, j - 1, j, j + 1])
keep = {}
prev = None
global_diffs = []  # baseline interframe diffs away from joins
with av.open(str(VID)) as c:
    idx = -1
    for f in c.decode(c.streams.video[0]):
        idx += 1
        arr = None
        if idx in need:
            arr = f.to_ndarray(format="rgb24")
            keep[idx] = arr
        # baseline: sample diffs every 37 frames mid-segment
        if idx % 37 == 25 and prev is not None:
            global_diffs.append(float(np.abs(arr.astype(np.int16) - prev.astype(np.int16)).mean()) if arr is not None else None)
        if arr is not None:
            prev = arr

global_diffs = [d for d in global_diffs if d is not None]
base = float(np.median(global_diffs)) if global_diffs else float("nan")
print(f"baseline median interframe diff (sampled): {base:.2f}/255")
stats = []
for j in JOINS:
    a, b = keep[j - 1].astype(np.int16), keep[j].astype(np.int16)
    d = float(np.abs(b - a).mean())
    stats.append((j, d))
    print(f"join at frame {j}: diff {d:.2f}/255  ({'OK' if d < base * 0.8 else 'CHECK'})")

# sheet: one row per join, 4 columns (j-2, j-1, j, j+1)
rows = []
for j in JOINS:
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
sheet = Image.new("RGB", (448 * 4 + 10, 256 * 5 + 8 * 5), (20, 20, 20))
for i, r in enumerate(rows):
    sheet.paste(r, (0, i * 264))
sheet.save(OUT)
print("saved", OUT)
