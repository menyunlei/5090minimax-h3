"""Full-video QC dense strip: sample every 24 frames over the whole 60s compile."""
import av
from PIL import Image, ImageDraw
from pathlib import Path

VID = Path(r"C:\Users\yunlei\.zcode\workspace\default\h3_map_baseline\continuous_m2_v1.mp4")
OUT = Path(r"C:\Users\yunlei\.zcode\workspace\default\h3_map_baseline\yym3_diag\m1_full_strip.png")
STEP, TW, TH, COLS = 24, 336, 192, 6

tiles = []
with av.open(str(VID)) as c:
    idx = -1
    for f in c.decode(c.streams.video[0]):
        idx += 1
        if idx % STEP:
            continue
        im = Image.fromarray(f.to_ndarray(format="rgb24")).resize((TW, TH), Image.LANCZOS)
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, 74, 14], fill=(0, 0, 0))
        d.text((3, 1), f"f{idx} {idx/24:.1f}s", fill=(255, 230, 80))
        tiles.append(im)
rows = (len(tiles) + COLS - 1) // COLS
sheet = Image.new("RGB", (TW * COLS, TH * rows), (20, 20, 20))
for i, t in enumerate(tiles):
    sheet.paste(t, ((i % COLS) * TW, (i // COLS) * TH))
sheet.save(OUT)
print("saved", OUT, len(tiles), "tiles", rows, "rows")
