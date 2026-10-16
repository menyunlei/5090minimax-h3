# Yuanmingyuan (Old Summer Palace) park map for the chained-turn test.
# Layout: main tree-lined path runs north; big lake on the west; at the junction
# plaza, a stone causeway branches LEFT (west) across the lake.
# Colors (same spec, no legend): white=open lawn, green=trees, blue=water,
# dark-gray=path/causeway, black=ruins blocks, red=route, yellow=start, orange=end.
from PIL import Image, ImageDraw
import numpy as np
from pathlib import Path

W = H = 1024
OUT_MAP = Path(r"C:\Users\yunlei\.zcode\workspace\default\h3_map_baseline\maps\yym_orig_nl.png")
IN_DIR = Path(r"C:\Users\yunlei\AppData\Local\Comfy-Desktop\ComfyUI-Shared\input")

img = Image.new("RGB", (W, H), (255, 255, 255))
d = ImageDraw.Draw(img)

# lake (west half), gentle organic boundary
lake_pol = [(0, 140), (430, 130), (470, 260), (450, 420), (500, 520), (470, 700),
            (430, 880), (0, 940)]
d.polygon(lake_pol, fill=(60, 110, 210))

# main path (east side), north-south
d.rectangle([680, 60, 740, 960], fill=(70, 70, 70))
# junction plaza
d.ellipse([620, 340, 800, 460], fill=(70, 70, 70))
# causeway west across the lake
d.rectangle([60, 378, 700, 428], fill=(70, 70, 70))

# ruins blocks (black) north and south of plaza on the east side
d.rectangle([600, 200, 660, 280], fill=(0, 0, 0))
d.rectangle([700, 160, 770, 250], fill=(0, 0, 0))
d.rectangle([610, 500, 680, 580], fill=(0, 0, 0))

# tree clusters (green), east side + around plaza
for cx, cy, r in [(560, 120, 46), (820, 140, 52), (840, 320, 44), (560, 660, 52),
                  (820, 620, 46), (760, 780, 52), (880, 860, 44), (600, 860, 44),
                  (850, 470, 40), (570, 380, 34), (810, 540, 44)]:
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(40, 140, 60))

# lotus patch in lake (lighter blue-green) SE corner of lake, away from causeway
d.ellipse([140, 560, 320, 720], fill=(90, 180, 150))

# route: north along path, LEFT onto the causeway
d.line([(710, 930), (710, 410), (130, 403)], fill=(255, 0, 0), width=14, joint="curve")
d.ellipse([710 - 16, 930 - 16, 710 + 16, 930 + 16], fill=(255, 220, 0))      # yellow start
d.ellipse([130 - 16, 403 - 13, 130 + 16, 403 + 19], fill=(255, 140, 0))      # orange end

img.save(OUT_MAP)
img.save(IN_DIR / "yym_orig_nl.png")
print("saved", OUT_MAP, "and input dir copy")
