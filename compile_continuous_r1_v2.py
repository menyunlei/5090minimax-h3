"""R1 one-shot v2: same 6-clip compile, but the minimap now has a MOVING position dot
gliding along the route polyline (plus traveled-trail highlight). Caption unchanged.
Timeline<->route waypoints were chosen per verified segment content (S6 dot crawls to END)."""
import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

OUT_DIR = Path(r"C:\Users\yunlei\AppData\Local\Comfy-Desktop\ComfyUI-Shared\output")
BASE = Path(r"C:\Users\yunlei\.zcode\workspace\default\h3_map_baseline")
OUT_VID = BASE / "continuous_r1_v2.mp4"
W, H = 1344, 768
FPS = 24

SEGMENTS = [f"h3rt1_s{i}_00001_.mp4" for i in range(1, 7)]
CAPTION = "One continuous ride through a Jiangnan canal old town   MiniMax H3   map-anchored chain of 6"
MAP_IMG = BASE / "maps" / "R1_orig.png"
FONT_CAP = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 20)

THUMB = 240          # minimap width in px
MSCALE = THUMB / 1024.0

# route waypoints in 1024-map space: (frame_index_in_final_video, x, y)
# segment content per verified strips: S1 ends mid-bridge, S2 mid-main-street,
# S3 just past the right turn, S4 mid-lane, S5 near balustrade, S6 crawls to END.
WAYPOINTS = [
    (0,    145, 940),
    (242,  145, 628),   # S1 end: bridge crest
    (484,  145, 470),   # S2 end: main street mid
    (726,  270, 340),   # S3 end: just after junction, in lane
    (968,  530, 340),   # S4 end: mid tree lane
    (1210, 815, 340),   # S5 end: near balustrade
    (1380, 874, 340),   # S6 ride-in: gliding to END
    (1452, 880, 340),   # final
]


def pos_at(frame):
    for i in range(len(WAYPOINTS) - 1):
        f0, x0, y0 = WAYPOINTS[i]
        f1, x1, y1 = WAYPOINTS[i + 1]
        if f0 <= frame <= f1:
            t = 0.0 if f1 == f0 else (frame - f0) / (f1 - f0)
            return (x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)
    return (WAYPOINTS[-1][1], WAYPOINTS[-1][2])


def make_map_base():
    """Pre-render the static minimap tile WITHOUT the route-progress highlight."""
    m = Image.open(MAP_IMG).convert("RGBA").resize((THUMB, THUMB), Image.LANCZOS)
    tile = Image.new("RGBA", (THUMB + 16, THUMB + 16), (0, 0, 0, 150))
    tile.alpha_composite(m, (8, 8))
    return tile


# list of FULL polyline pts for the trail (1024 space)
POLY = [(145, 940), (145, 340), (880, 340)]


def trail_partial(upto):
    """Return the list of polyline points up to interpolated position `upto`."""
    pts = [POLY[0]]
    acc_limit = None
    # walk polyline segments
    total = [0.0]
    for a, b in zip(POLY[:-1], POLY[1:]):
        total.append(total[-1] + abs(b[0] - a[0]) + abs(b[1] - a[1]))
    # distance of upto along polyline
    du = 0.0
    for a, b in zip(POLY[:-1], POLY[1:]):
        seg = abs(b[0] - a[0]) + abs(b[1] - a[1])
        # is upto on this segment? (they are axis aligned)
        if (min(a[0], b[0]) - 0.5 <= upto[0] <= max(a[0], b[0]) + 0.5 and
                min(a[1], b[1]) - 0.5 <= upto[1] <= max(a[1], b[1]) + 0.5):
            du += abs(upto[0] - a[0]) + abs(upto[1] - a[1])
            break
        du += seg
    out = [POLY[0]]
    run = 0.0
    for a, b in zip(POLY[:-1], POLY[1:]):
        seg = abs(b[0] - a[0]) + abs(b[1] - a[1])
        if run + seg <= du:
            out.append(b)
            run += seg
        else:
            r = 0 if seg == 0 else (du - run) / seg
            out.append((a[0] + (b[0] - a[0]) * r, a[1] + (b[1] - a[1]) * r))
            break
    return out


def overlay_factory():
    base = make_map_base()
    def apply(rgb, fidx):
        img = Image.fromarray(rgb).convert("RGBA")
        ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)
        # minimap with trail + dot
        tile = base.copy()
        td = ImageDraw.Draw(tile)
        p = pos_at(fidx)
        trail = trail_partial(p)
        if len(trail) >= 2:
            td.line([(8 + x * MSCALE, 8 + y * MSCALE) for x, y in trail],
                    fill=(255, 214, 0, 255), width=3)
        cx, cy = 8 + p[0] * MSCALE, 8 + p[1] * MSCALE
        pulse = 1 if (fidx // 12) % 2 == 0 else 0
        td.ellipse([cx - 8 - pulse, cy - 8 - pulse, cx + 8 + pulse, cy + 8 + pulse],
                   outline=(255, 255, 255, 200), width=2)
        td.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=(255, 214, 0, 255),
                   outline=(120, 60, 0, 255), width=1)
        ov.alpha_composite(tile, (14, 14))
        # fixed caption
        bb = d.textbbox((0, 0), CAPTION, font=FONT_CAP)
        tw = bb[2] - bb[0]
        d.rectangle([W - tw - 34, 14, W - 12, 46], fill=(0, 0, 0, 150))
        d.text((W - tw - 23, 21), CAPTION, font=FONT_CAP, fill=(255, 230, 80, 255))
        return np.array(Image.alpha_composite(img, ov).convert("RGB"))
    return apply


def main():
    apply = overlay_factory()
    missing = [n for n in SEGMENTS if not (OUT_DIR / n).exists()]
    if missing:
        print("MISSING:", missing)
        return
    out = av.open(str(OUT_VID), mode="w")
    os_ = out.add_stream("h264", rate=FPS)
    os_.width, os_.height, os_.pix_fmt = W, H, "yuv420p"
    os_.options = {"crf": "19", "preset": "veryfast"}
    total = 0
    for idx, name in enumerate(SEGMENTS):
        with av.open(str(OUT_DIR / name)) as inp:
            n = 0
            for frame in inp.decode(inp.streams.video[0]):
                n += 1
                if idx > 0 and n == 1:
                    continue
                arr = apply(frame.to_ndarray(format="rgb24"), total)
                for pkt in os_.encode(av.VideoFrame.from_ndarray(arr, format="rgb24")):
                    out.mux(pkt)
                total += 1
        print(f"seg {idx+1}/6 {name} done (cum {total})")
    for pkt in os_.encode():
        out.mux(pkt)
    out.close()
    print(f"DONE {OUT_VID}  {total} frames ~{total/FPS:.1f}s {OUT_VID.stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
