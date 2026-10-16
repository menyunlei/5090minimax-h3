"""v4 compile: 5 map-conditioned hops (Ref2V route-map + prev-frame refs), anchor-dup drop,
FIXED overlays only: annotated route map inset top-left + one fixed caption top-right."""
import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

OUT_DIR = Path(r"C:\Users\yunlei\AppData\Local\Comfy-Desktop\ComfyUI-Shared\output")
BASE = Path(r"C:\Users\yunlei\.zcode\workspace\default\h3_map_baseline")
OUT_VID = BASE / "continuous_yym_v4_mapeveryhop.mp4"
W, H = 1344, 768

SEGMENTS = [
    "h3v4_h1_00001_.mp4",
    "h2_00001_.mp4",
    "h3v4_h3_00001_.mp4",
    "h3v4_h4_00001_.mp4",
    "h3v4_h5_00001_.mp4",
]

CAPTION = "Map conditions EVERY segment: route-map + prev-frame refs per hop   MiniMax H3"
FONT_CAP = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 20)


def load_map_inset():
    m = Image.open(BASE / "yym_route_v4.png").convert("RGBA")  # user-visible labelled map
    m.thumbnail((300, 300), Image.LANCZOS)
    tile = Image.new("RGBA", (m.width + 16, m.height + 16), (0, 0, 0, 150))
    tile.alpha_composite(m, (8, 8))
    return tile


def main():
    tile = load_map_inset()
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
                    continue
                img = Image.fromarray(frame.to_ndarray(format="rgb24")).convert("RGBA")
                ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
                d = ImageDraw.Draw(ov)
                ov.alpha_composite(tile, (14, 14))
                bb = d.textbbox((0, 0), CAPTION, font=FONT_CAP)
                tw = bb[2] - bb[0]
                d.rectangle([W - tw - 34, 14, W - 12, 46], fill=(0, 0, 0, 150))
                d.text((W - tw - 23, 21), CAPTION, font=FONT_CAP, fill=(255, 230, 80, 255))
                arr = np.array(Image.alpha_composite(img, ov).convert("RGB"))
                for pkt in os_.encode(av.VideoFrame.from_ndarray(arr, format="rgb24")):
                    out.mux(pkt)
                total += 1
        print(f"seg {idx+1}/5 {name} (cum {total})")
    for pkt in os_.encode():
        out.mux(pkt)
    out.close()
    print(f"DONE {OUT_VID}  {total} frames ~{total/24:.1f}s {OUT_VID.stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
