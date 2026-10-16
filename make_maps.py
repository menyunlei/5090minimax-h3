# Generate synthetic test maps for MiniMax H3 map->video baseline experiments.
# Spec: white=traversable ground, black/gray=building, dark gray=road,
# blue=water, green=vegetation, red=route, yellow dot=start, orange dot=end,
# small legend bottom-right.
import os, shutil
from PIL import Image, ImageDraw, ImageFont, ImageOps

OUT = r"C:\Users\yunlei\.zcode\workspace\default\h3_map_baseline\maps"
INPUT_DIR = r"C:\Users\yunlei\AppData\Local\Comfy-Desktop\ComfyUI-Shared\input"
os.makedirs(OUT, exist_ok=True)

W = H = 1024
WHITE = (255, 255, 255)
BUILDING = (25, 25, 25)
ROAD = (85, 85, 85)
ROUTE = (230, 30, 30)
START = (255, 210, 0)
END = (255, 140, 0)
WATER = (65, 125, 190)
VEG = (95, 165, 95)
ROAD_W = 44
ROUTE_W = 9

try:
    FONT = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 24)
except Exception:
    FONT = ImageFont.load_default(size=24)


def canvas():
    return Image.new("RGB", (W, H), WHITE)


def road_h(d, y, x0=90, x1=934):
    d.rectangle([x0, y - ROAD_W // 2, x1, y + ROAD_W // 2], fill=ROAD)


def road_v(d, x, y0, y1):
    d.rectangle([x - ROAD_W // 2, y0, x + ROAD_W // 2, y1], fill=ROAD)


def dot(d, xy, color, r=14):
    x, y = xy
    d.ellipse([x - r, y - r, x + r, y + r], fill=color, outline=WHITE, width=3)


def arrow(d, xy, direction):
    x, y = xy
    s = 22
    if direction == "right":
        pts = [(x + s, y), (x - s // 2, y - s // 2), (x - s // 2, y + s // 2)]
    elif direction == "up":
        pts = [(x, y - s), (x - s // 2, y + s // 2), (x + s // 2, y + s // 2)]
    elif direction == "left":
        pts = [(x - s, y), (x + s // 2, y - s // 2), (x + s // 2, y + s // 2)]
    d.polygon(pts, fill=ROUTE)


DRAW_LEGEND = True  # set False to skip legend (no-legend ablation)


def legend(d):
    if not DRAW_LEGEND:
        return
    entries = [
        (BUILDING, "building"), (ROAD, "road"), (VEG, "vegetation"),
        (WATER, "water"), (ROUTE, "route (camera path)"),
        (START, "start"), (END, "end"),
    ]
    lw, lh, row = 250, 34, 26
    x0, y0 = W - lw - 20, H - lh * len(entries) - 30
    d.rectangle([x0, y0, x0 + lw, y0 + lh * len(entries) + 8], fill=WHITE, outline=(120, 120, 120), width=2)
    for i, (c, label) in enumerate(entries):
        y = y0 + 10 + i * lh
        d.rectangle([x0 + 12, y, x0 + 44, y + 20], fill=c, outline=(80, 80, 80))
        d.text((x0 + 56, y - 3), label, fill=(20, 20, 20), font=FONT)


def save(img, name):
    p = os.path.join(OUT, name + ".png")
    img.save(p)
    shutil.copy(p, os.path.join(INPUT_DIR, name + ".png"))
    print("saved", name)


# ---------- C1: straight road, buildings one side / open ground other side ----------
def c1(buildings_north=True):
    img = canvas(); d = ImageDraw.Draw(img)
    road_h(d, 470)
    ys = (300, 430) if buildings_north else (560, 690)
    xs = 140
    while xs < 700:
        d.rectangle([xs, ys[0], xs + 110, ys[1]], fill=BUILDING)
        xs += 140
    if not buildings_north:  # keep clear of legend on the south edge
        pass
    d.line([120, 470, 904, 470], fill=ROUTE, width=ROUTE_W)
    dot(d, (120, 470), START); dot(d, (904, 470), END)
    arrow(d, (170, 470), "right")
    legend(d)
    return img

save(c1(True), "C1_orig")   # buildings north = LEFT of travel, open south = RIGHT
save(c1(False), "C1_edit")  # mirrored sides

# ---------- C2: 90-degree turn; original turns LEFT, edit turns RIGHT ----------
def c2():
    img = canvas(); d = ImageDraw.Draw(img)
    road_v(d, 512, 320, 934)
    road_h(d, 320, 140, 512)
    # buildings flanking the vertical segment (east side) and horizontal segment (north side)
    for y0 in (430, 610, 790):
        d.rectangle([600, y0, 760, y0 + 130], fill=BUILDING)
    for x0 in (190, 340):
        d.rectangle([x0, 150, x0 + 120, 250], fill=BUILDING)
    d.line([512, 904, 512, 320, 140, 320], fill=ROUTE, width=ROUTE_W, joint="curve")
    dot(d, (512, 904), START); dot(d, (140, 320), END)
    arrow(d, (512, 850), "up")
    legend(d)
    return img

save(c2(), "C2_orig")                       # LEFT turn
save(ImageOps.mirror(c2()), "C2_edit")      # RIGHT turn (mirrored map)

# ---------- C3: courtyard hole inside a block vs solid block ----------
def c3(courtyard=True):
    img = canvas(); d = ImageDraw.Draw(img)
    road_h(d, 470)
    d.rectangle([300, 260, 760, 430], fill=BUILDING)
    if courtyard:
        d.rectangle([420, 285, 640, 405], fill=WHITE)
    d.rectangle([140, 620, 260, 720], fill=BUILDING)
    d.rectangle([820, 620, 900, 700], fill=BUILDING)
    d.line([120, 470, 904, 470], fill=ROUTE, width=ROUTE_W)
    dot(d, (120, 470), START); dot(d, (904, 470), END)
    arrow(d, (170, 470), "right")
    legend(d)
    return img

save(c3(True), "C3_orig")    # courtyard (open gap) beside the route
save(c3(False), "C3_edit")   # same block filled solid

# ---------- C4: water on one side vs vegetation ----------
def c4(water=True):
    img = canvas(); d = ImageDraw.Draw(img)
    road_h(d, 470)
    fill = WATER if water else VEG
    d.rectangle([90, 540, 934, 930], fill=fill)
    d.rectangle([150, 290, 280, 400], fill=BUILDING)
    d.rectangle([420, 290, 560, 400], fill=BUILDING)
    d.rectangle([700, 290, 840, 400], fill=VEG)
    d.line([120, 470, 904, 470], fill=ROUTE, width=ROUTE_W)
    dot(d, (120, 470), START); dot(d, (904, 470), END)
    arrow(d, (170, 470), "right")
    legend(d)
    return img

save(c4(True), "C4_orig")    # water south = RIGHT of travel
save(c4(False), "C4_edit")   # same area vegetated, no water

# ---------- no-legend ablation versions ----------
DRAW_LEGEND = False
save(c1(True), "C1_orig_nl")
save(c1(False), "C1_edit_nl")
save(c2(), "C2_orig_nl")
save(ImageOps.mirror(c2()), "C2_edit_nl")
save(c3(True), "C3_orig_nl")
save(c3(False), "C3_edit_nl")
save(c4(True), "C4_orig_nl")
save(c4(False), "C4_edit_nl")

print("all maps done (with + no-legend)")
