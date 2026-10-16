"""v3 continuous 60s one-shot: 6 x 241-frame H3 clips, anchor-dup frame dropped per join,
STRICTLY FIXED overlays only (static map inset + one fixed caption). No per-segment text."""
import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

OUT_DIR = Path(r"C:\Users\yunlei\AppData\Local\Comfy-Desktop\ComfyUI-Shared\output")
BASE = Path(r"C:\Users\yunlei\.zcode\workspace\default\h3_map_baseline")
OUT_VID = BASE / "continuous_yym_v3.mp4"
W, H = 1344, 768

SEGMENTS = [
    "h3yym3_s1_00001_.mp4",
    "h3yym3_s2_00001_.mp4",
    "h3yym3_s3_00001_.mp4",
    "h3yym3_s4_00001_.mp4",
    "h3yym3_s5_00001_.mp4",
    "h3yym3_s6_00001_.mp4",
]

CAPTION = "One continuous ride through Yuanmingyuan-style imperial garden   MiniMax H3   map-anchored chain of 6"
IN_DIR = Path(r"C:\Users\yunlei\AppData\Local\Comfy-Desktop\ComfyUI-Shared\input")
MAP_IMG = IN_DIR / "yym_orig_nl.png"   # the map that actually conditioned S1
FONT_CAP = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 20)


def load_map_inset():
    for cand in [MAP_IMG, BASE / "yym_orig_nl.png", BASE / "maps" / "yym_orig_nl.png"]:
        if cand.exists():
            im = Image.open(cand).convert("RGBA")
            im.thumbnail((300, 224), Image.LANCZOS)
            return im
    return None


def fixed_overlay_factory():
    m = load_map_inset()
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
    for idx, name in enumerate(SEGMENTS):
        with av.open(str(OUT_DIR / name)) as inp:
            n = 0
            for frame in inp.decode(inp.streams.video[0]):
                n += 1
                if idx > 0 and n == 1:
                    continue   # drop the duplicated anchor frame at each join
                arr = apply(frame.to_ndarray(format="rgb24"))
                for pkt in os_.encode(av.VideoFrame.from_ndarray(arr, format="rgb24")):
                    out.mux(pkt)
                total += 1
        print(f"seg {idx+1}/6 {name} done (cum {total})")
    for pkt in os_.encode():
        out.mux(pkt)
    out.close()
    print(f"DONE {OUT_VID}  {total} frames ~{total/24:.1f}s {OUT_VID.stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
