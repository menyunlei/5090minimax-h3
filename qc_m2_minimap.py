# Minimap-dot QC for continuous_m2_v1.mp4: crop the minimap corner at the
# six segment boundaries and check the yellow dot sits on the route.
import av
from PIL import Image, ImageDraw
from pathlib import Path

VID = Path(r"C:\Users\yunlei\.zcode\workspace\default\h3_map_baseline\continuous_m2_v1.mp4")
OUT = Path(r"C:\Users\yunlei\.zcode\workspace\default\h3_map_baseline\yym3_diag\m2_minimap_qc.png")
PTS = [0, 242, 484, 726, 968, 1210, 1452]
LABELS = ["start", "S1->S2", "S2->S3", "S3->S4", "S4->S5", "S5->S6", "end"]

need = set(PTS)
keep = {}
with av.open(str(VID)) as c:
    idx = -1
    for f in c.decode(c.streams.video[0]):
        idx += 1
        if idx in need:
            keep[idx] = f.to_image().convert("RGB")
        if idx > max(PTS):
            break

tiles = []
for p, lab in zip(PTS, LABELS):
    im = keep[p]
    crop = im.crop((1344 - 320, 0, 1344, 560))  # minimap corner region
    dd = ImageDraw.Draw(crop)
    dd.rectangle([0, 0, 130, 20], fill=(0, 0, 0))
    dd.text((4, 3), f"f{p} {lab}", fill=(255, 230, 80))
    tiles.append(crop)

W = max(t.width for t in tiles)
sheet = Image.new("RGB", (W * 2 + 6, 560 * 4 + 6 * 4), (20, 20, 20))
for i, t in enumerate(tiles):
    sheet.paste(t, ((i % 2) * (W + 6), (i // 2) * 566))
sheet.save(OUT)
print("saved", OUT)
