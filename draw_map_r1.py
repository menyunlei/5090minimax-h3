"""Draw map R1: southern-Chinese canal old-town route.
Start (south, bottom) -> NORTH through old-town street -> stone bridge over E-W canal
-> main street -> RIGHT 90 turn EAST into tree lane -> plaza by north pond (END).
Two variants: R1_orig.png (with legend) and R1_orig_nl.png (no legend)."""
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

OUT = Path(r"C:\Users\yunlei\.zcode\workspace\default\h3_map_baseline\maps")
S = 1024
C_BUILD = (20, 20, 20)
C_ROAD = (74, 74, 74)
C_VEG = (106, 175, 114)
C_WATER = (72, 111, 165)
C_ROUTE = (226, 30, 30)
C_START = (245, 205, 0)
C_END = (245, 150, 30)


def route(dd, with_arrows=True):
    # polyline: start -> bridge -> junction -> plaza end
    pts = [(145, 940), (145, 340), (880, 340)]
    dd.line(pts, fill=C_ROUTE, width=12, joint="curve")
    if with_arrows:
        # arrowhead on north leg (moving up)
        dd.polygon([(135, 660), (155, 660), (145, 640)], fill=C_ROUTE)
        # arrowhead on east leg (moving right)
        dd.polygon([(500, 330), (500, 350), (522, 340)], fill=C_ROUTE)
    # start / end dots
    dd.ellipse([129, 924, 161, 956], fill=C_START, outline=(255, 255, 255), width=4)
    dd.ellipse([864, 324, 896, 356], fill=C_END, outline=(255, 255, 255), width=4)


def draw(with_legend, path):
    im = Image.new("RGB", (S, S), (255, 255, 255))
    d = ImageDraw.Draw(im)
    # water: E-W canal + northern pond
    d.rectangle([60, 560, 960, 700], fill=C_WATER)
    d.rectangle([620, 120, 960, 260], fill=C_WATER)
    # roads: N street (incl. bridge over canal), E tree lane, plaza
    d.rectangle([120, 360, 170, 940], fill=C_ROAD)
    d.rectangle([170, 320, 830, 370], fill=C_ROAD)
    d.rectangle([820, 290, 950, 430], fill=C_ROAD)
    # buildings: flank southern approach + main street
    for box in [(60, 760, 105, 920), (195, 760, 260, 920),
                (60, 420, 105, 540), (195, 420, 255, 540),
                (60, 380, 105, 405), (195, 380, 255, 405)]:
        d.rectangle(box, fill=C_BUILD)
    # vegetation: trees along lane (north & south), park by pond
    d.rectangle([210, 240, 560, 300], fill=C_VEG)
    d.rectangle([210, 390, 560, 470], fill=C_VEG)
    d.rectangle([830, 120, 950, 260], fill=C_VEG)  # NE park by pond edge overlap? draw over water corner
    d.rectangle([820, 450, 940, 520], fill=C_VEG)  # green south of plaza
    # route on top
    route(d)
    if with_legend:
        font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 26)
        items = [("building", C_BUILD), ("road", C_ROAD), ("vegetation", C_VEG),
                 ("water", C_WATER), ("route (camera path)", C_ROUTE),
                 ("start", C_START), ("end", C_END)]
        lx, ly, lw, lh = 700, 780, 310, 236
        d.rectangle([lx, ly, lx + lw, ly + lh], fill=(255, 255, 255), outline=(120, 120, 120), width=2)
        for i, (name, col) in enumerate(items):
            y = ly + 10 + i * 32
            d.rectangle([lx + 14, y + 4, lx + 44, y + 28], fill=col)
            d.text((lx + 54, y), name, font=font, fill=(10, 10, 10))
    im.save(OUT / path)
    print("saved", path)


draw(True, "R1_orig.png")
draw(False, "R1_orig_nl.png")
