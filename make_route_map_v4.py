"""v4 route map: annotate the yym base map with leg labels, waypoints, direction arrows.
Route (already drawn in red on base): W0 orange (112,414) -> east along causeway -> W1 junction (676,414) -> south -> W2 yellow end (676,900)."""
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import shutil

SRC = Path(r"C:\Users\yunlei\AppData\Local\Comfy-Desktop\ComfyUI-Shared\input\yym_orig_nl.png")
OUT_WS = Path(r"C:\Users\yunlei\.zcode\workspace\default\h3_map_baseline\yym_route_v4.png")
OUT_IN = Path(r"C:\Users\yunlei\AppData\Local\Comfy-Desktop\ComfyUI-Shared\input\yym_route_v4.png")

FB = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 34)
FS = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 24)

im = Image.open(SRC).convert("RGB")
print("size", im.size)
d = ImageDraw.Draw(im)


def label(x, y, txt, font=FB):
    bb = d.textbbox((0, 0), txt, font=font)
    w, h = bb[2] - bb[0], bb[3] - bb[1]
    d.rectangle([x - 6, y - 4, x + w + 6, y + h + 6], fill=(255, 255, 255), outline=(0, 0, 0), width=2)
    d.text((x, y), txt, font=font, fill=(10, 10, 10))


def arrow(x1, y1, x2, y2):
    d.line([x1, y1, x2, y2], fill=(255, 255, 255), width=12)
    d.line([x1, y1, x2, y2], fill=(0, 0, 0), width=4)


# leg labels along the two red legs
label(330, 352, "L1 EAST on causeway (water both sides)")
label(700, 640, "L2 SOUTH lakeside path (lake on RIGHT)")
# waypoint markers
label(150, 352, "START W0")
label(600, 300, "W1 junction: TURN RIGHT (South)")
label(560, 940, "W2 END terrace")
# direction arrows on route
arrow(300, 414, 420, 414)   # east
arrow(676, 560, 676, 700)   # south
# compass
label(40, 40, "N", FS)
arrow(70, 110, 70, 70)

im.save(OUT_WS)
shutil.copy(OUT_WS, OUT_IN)
print("saved", OUT_WS, "and copied into ComfyUI input")
