# M3 conditioning map v2: route on the MAIN tree-lined avenue (x~495-510),
# crosshair inpainted at its true spot, bottom info text covered.
from PIL import Image, ImageDraw
import math, shutil

BASE = r"C:\Users\yunlei\.zcode\workspace\default\h3_map_baseline"
INPUT = r"C:\Users\yunlei\AppData\Local\Comfy-Desktop\ComfyUI-Shared\input"

im = Image.open(BASE + r"\maps\M3_clean.png").convert("RGB")   # 855x1875
im = im.crop((0, 45, 855, 1825))                                # 855x1780 final

def inpaint(img, box, dy):
    x0, y0, x1, y1 = box
    img.paste(img.crop((x0, y0 + dy, x1, y1 + dy)), (x0, y0))

# crosshair "+" on the main avenue centreline
inpaint(im, (486, 822, 548, 908), -135)
# bottom-left "谷歌影像图(无偏移) 17级" info text
inpaint(im, (20, 1700, 348, 1780), -120)
im.save(BASE + r"\maps\M3_clean2.png")

POLY = [
    (495, 1700), (495, 1550), (500, 1400), (500, 1250), (500, 1050),
    (500, 830), (502, 620), (503, 460), (505, 330), (510, 140),
]
d = ImageDraw.Draw(im, "RGBA")
RED = (226, 30, 30, 235)
d.line(POLY, fill=RED, width=12, joint="curve")

def arrow(p0, p1, L=44, spread=26):
    ax, ay = p1
    th = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    for s in (1, -1):
        t2 = th + math.pi - s * math.radians(spread)
        d.line([(ax, ay), (ax + L * math.cos(t2), ay + L * math.sin(t2))], fill=RED, width=10)

arrow(POLY[2], POLY[3])
arrow(POLY[6], POLY[7])

sx, sy = POLY[0]
d.ellipse([sx-16, sy-16, sx+16, sy+16], fill=(255, 210, 40, 255), outline=(0,0,0,255), width=3)
ex, ey = POLY[-1]
d.ellipse([ex-16, ey-16, ex+16, ey+16], fill=(255, 130, 30, 255), outline=(0,0,0,255), width=3)

im.save(BASE + r"\maps\M3_route.png")
shutil.copy(BASE + r"\maps\M3_route.png", INPUT + r"\M3_route.png")
shutil.copy(BASE + r"\maps\M3_clean2.png", INPUT + r"\M3_clean2.png")
print("M3_route v2", im.size)
