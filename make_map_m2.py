# M2 conditioning map from E:/D/map_build/map/9.jpg:
# 1. crop chrome (further top/right trim on top of M2_clean)
# 2. inpaint app crosshair + Custom/scale watermarks by cloning nearby texture
# 3. draw route polyline + start/end dots + arrowheads -> maps/M2_route.png
from PIL import Image, ImageDraw, ImageFont
import math, shutil

BASE = r"C:\Users\yunlei\.zcode\workspace\default\h3_map_baseline"
INPUT = r"C:\Users\yunlei\AppData\Local\Comfy-Desktop\ComfyUI-Shared\input"

im = Image.open(BASE + r"\maps\M2_clean.png").convert("RGB")   # 840x1850
im = im.crop((0, 70, 820, 1850))                               # 820x1780 final
W, H = im.size
OY = 70  # y-offset vs M2_clean coords used during planning

def inpaint(img, box, dy):
    """clone a strip of same size from dy pixels up over box"""
    x0, y0, x1, y1 = box
    src = (x0, y0 + dy, x1, y1 + dy)
    img.paste(img.crop(src), (x0, y0))

# boxes below are in FINAL cropped coords (y already local)
# 1) crosshair "+" right of the route, near fence/cone
inpaint(im, (400, 885, 486, 956), -100)
# 2) "Custom (317)" watermark bottom-left (starts at image edge)
inpaint(im, (0, 1552, 268, 1642), -100)
# 3) "20米" scale bar bottom-right edge
inpaint(im, (630, 1552, 820, 1645), -110)
# 4) leftover watermark glyph fragments next to the route on the water
inpaint(im, (253, 1542, 337, 1618), -95)
im.save(BASE + r"\maps\M2_clean2.png")

# route in FINAL coords (y = clean_y - 70)
POLY = [
    (340, 1845 - OY), (330, 1700 - OY), (320, 1560 - OY), (315, 1400 - OY),
    (310, 1250 - OY), (305, 1120 - OY), (300, 990 - OY), (285, 800 - OY),
    (270, 650 - OY), (250, 500 - OY), (200, 380 - OY), (150, 250 - OY), (120, 150 - OY),
]

d = ImageDraw.Draw(im, "RGBA")
RED = (226, 30, 30, 235)
d.line(POLY, fill=RED, width=12, joint="curve")

def arrowhead(p_prev, p_end, L=42, ang=28):
    dx, dy = p_end[0] - p_prev[0], p_end[1] - p_prev[1]
    a = math.atan2(dy, dx)
    for s in (1, -1):
        a2 = a + s * math.radians(ang)
        d.line([p_end, (p_end[0] - L * math.cos(a2 + math.pi) * -1, p_end[1] - L * math.sin(a2 + math.pi) * -1)],
               fill=RED, width=10)

def arrow(p0, p1, L=44, spread=26):
    ax, ay = p1
    th = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    for s in (1, -1):
        t2 = th + math.pi - s * math.radians(spread)
        d.line([(ax, ay), (ax + L * math.cos(t2), ay + L * math.sin(t2))], fill=RED, width=10)

# arrowheads roughly mid-bridge and mid-junction
arrow(POLY[3], POLY[4])
arrow(POLY[9], POLY[10])

# start / end dots
sx, sy = POLY[0]
d.ellipse([sx-16, sy-16, sx+16, sy+16], fill=(255, 210, 40, 255), outline=(0,0,0,255), width=3)
ex, ey = POLY[-1]
d.ellipse([ex-16, ey-16, ex+16, ey+16], fill=(255, 130, 30, 255), outline=(0,0,0,255), width=3)

im.save(BASE + r"\maps\M2_route.png")
shutil.copy(BASE + r"\maps\M2_route.png", INPUT + r"\M2_route.png")
shutil.copy(BASE + r"\maps\M2_clean2.png", INPUT + r"\M2_clean2.png")
print("M2_route.png", im.size, "POLY[0..2]=", POLY[:3], "POLY[-1]=", POLY[-1])
