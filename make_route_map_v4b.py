"""De-objected conditioning map: NO filled discs/markers on the route (they materialize as
3D objects). Thin dark route line + tiny hollow waypoint rings + small labels."""
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import shutil

SRC = Path(r"C:\Users\yunlei\AppData\Local\Comfy-Desktop\ComfyUI-Shared\input\yym_orig_nl.png")
OUT = Path(r"C:\Users\yunlei\.zcode\workspace\default\h3_map_baseline\yym_route_v4b.png")
IN = Path(r"C:\Users\yunlei\AppData\Local\Comfy-Desktop\ComfyUI-Shared\input\yym_route_v4b.png")

F = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 15)
im = Image.open(SRC).convert("RGB")
d = ImageDraw.Draw(im)

# repaint the route area: erase the original thick red line & colored dots by sampling path grey
# simpler: draw route thin dark-red over whatever exists (discs get covered by tiny rings below)
route = [(112, 414), (676, 414), (676, 900)]
# cover original orange/yellow discs with path-grey fill
GREY = (74, 70, 66)  # sampled-ish path grey
for (x, y), r in [((112, 414), 16), ((676, 900), 14)]:
    d.ellipse([x - r, y - r, x + r, y + r], fill=GREY)
# thin route line (dark red, 4px)
d.line(route, fill=(150, 30, 30), width=4, joint="curve")
# tiny hollow rings at waypoints
for (x, y) in route:
    d.ellipse([x - 9, y - 9, x + 9, y + 9], outline=(150, 30, 30), width=3)
# small labels
d.text((128, 396), "start", font=F, fill=(120, 30, 30), stroke_width=2, stroke_fill=(255, 255, 255))
d.text((688, 396), "turn", font=F, fill=(120, 30, 30), stroke_width=2, stroke_fill=(255, 255, 255))
d.text((688, 884), "end", font=F, fill=(120, 30, 30), stroke_width=2, stroke_fill=(255, 255, 255))
d.line([40, 36, 40, 20], fill=(60, 60, 60), width=2)
d.text((34, 40), "N", font=F, fill=(60, 60, 60))

im.save(OUT)
shutil.copy(OUT, IN)
print("saved", OUT)
