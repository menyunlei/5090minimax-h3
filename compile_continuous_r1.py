"""R1 continuous 60s one-shot: 6 x 243-frame clips, anchor-dup frame dropped per join,
STRICTLY FIXED overlays (R1 route map inset + one fixed caption). No per-segment text."""
import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

OUT_DIR = Path(r"C:\Users\yunlei\AppData\Local\Comfy-Desktop\ComfyUI-Shared\output")
BASE = Path(r"C:\Users\yunlei\.zcode\workspace\default\h3_map_baseline")
OUT_VID = BASE / "continuous_r1_v1.mp4"
W, H = 1344, 768

SEGMENTS = [f"h3rt1_s{i}_00001_.mp4" for i in range(1, 7)]

CAPTION = "One continuous ride through a Jiangnan canal old town   MiniMax H3   map-anchored chain of 6"
MAP_IMG = BASE / "maps" / "R1_orig.png"   # legend version: still legible as a map block when small
FONT_CAP = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 20)


def fixed_overlay_factory():
    m = None
    if MAP_IMG.exists():
        m = Image.open(MAP_IMG).convert("RGBA")
        m.thumbnail((240, 240), Image.LANCZOS)
    tile = None
    if m is not None:
        tile = Image.new("RGBA", (m.width + 16, m.height + 16), (0, 0, 0, 150))
        tile.alpha_composite(m, (8, 8))
    def apply(rgb):
        img = Image.fromarray(rgb).convert("RGBA")
        ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)
        if tile is not None:
            ov.alpha_composite(tile, (14, 14))
        bb = d.textbbox((0, 0), CAPTION, font=FONT_CAP)
        tw = bb[2] - bb[0]
        d.rectangle([W - tw - 34, 14, W - 12, 46], fill=(0, 0, 0, 150))
        d.text((W - tw - 23, 21), CAPTION, font=FONT_CAP, fill=(255, 230, 80, 255))
        return np.array(Image.alpha_composite(img, ov).convert("RGB"))
    return apply


def main():
    apply = fixed_overlay_factory()
    missing = [n for n in SEGMENTS if not (OUT_DIR / n).exists()]
    if missing:
        print("MISSING:", missing)
        return
    out = av.open(str(OUT_VID), mode="w")
    os_ = out.add_stream("h264", rate=24)
    os_.width, os_.height, os_.pix_fmt = W, H, "yuv420p"
    os_.options = {"crf": "19", "preset": "veryfast"}
    total = 0
    join_at = []
    for idx, name in enumerate(SEGMENTS):
        with av.open(str(OUT_DIR / name)) as inp:
            n = 0
            for frame in inp.decode(inp.streams.video[0]):
                n += 1
                if idx > 0 and n == 1:
                    if idx == 0:
                        pass
                    join_at.append(total)
                    continue   # drop duplicated anchor frame at each join
                arr = apply(frame.to_ndarray(format="rgb24"))
                for pkt in os_.encode(av.VideoFrame.from_ndarray(arr, format="rgb24")):
                    out.mux(pkt)
                total += 1
        print(f"seg {idx+1}/6 {name} done (cum {total})")
    for pkt in os_.encode():
        out.mux(pkt)
    out.close()
    print("JOINS_AT", join_at)
    print(f"DONE {OUT_VID}  {total} frames ~{total/24:.1f}s {OUT_VID.stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
