"""Turn-experiment showreel: map-only (fails) vs map+text timeline (works) vs text-only (works).
10 clips x 5.17s = ~52s, labelled, muted, h264."""
import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

OUT_DIR = Path(r"C:\Users\yunlei\AppData\Local\Comfy-Desktop\ComfyUI-Shared\output")
MAP_DIR = Path(r"C:\Users\yunlei\.zcode\workspace\default\h3_map_baseline\maps")
OUT_VID = Path(r"C:\Users\yunlei\.zcode\workspace\default\h3_map_baseline\showreel_turn.mp4")
W, H = 1344, 768

# (video file, map thumb or None, top caption, bottom caption)
CLIPS = [
    ("h3mapNL_C2_orig_s1_00001_.mp4", "C2_orig_nl.png",
     "A  map-only (v2 baseline)   left-turn route   seed 42",
     "camera goes STRAIGHT — turn never executed"),
    ("h3mapNL_C2_edit_s1_00001_.mp4", "C2_edit_nl.png",
     "A  map-only (v2 baseline)   right-turn route   seed 42",
     "camera goes STRAIGHT — turn never executed"),
    ("h3mapNL_C2_orig_s2_00001_.mp4", "C2_orig_nl.png",
     "A  map-only (v2 baseline)   left-turn route   seed 1337",
     "camera goes STRAIGHT — turn never executed"),
    ("h3mapNL_C2_orig_s3_00001_.mp4", "C2_orig_nl.png",
     "A  map-only (v2 baseline)   left-turn route   seed 20261006",
     "camera goes STRAIGHT — turn never executed"),
    ("h3turn_B_maptext_s1_00001_.mp4", "C2_orig_nl.png",
     "B  map + TEXT timeline (turn left 90deg at 2-4s)   seed 42",
     "LEFT TURN EXECUTED  \u2713"),
    ("h3turn_B_maptext_s2_00001_.mp4", "C2_orig_nl.png",
     "B  map + TEXT timeline (turn left 90deg at 2-4s)   seed 1337",
     "LEFT TURN EXECUTED  \u2713"),
    ("h3turn_B_maptext_s3_00001_.mp4", "C2_orig_nl.png",
     "B  map + TEXT timeline (turn left 90deg at 2-4s)   seed 20261006",
     "LEFT TURN EXECUTED  \u2713"),
    ("h3turn_C_textonly_s1_00001_.mp4", None,
     "C  TEXT timeline only, NO map   seed 42",
     "LEFT TURN EXECUTED  \u2713 (gentle)"),
    ("h3turn_C_textonly_s2_00001_.mp4", None,
     "C  TEXT timeline only, NO map   seed 1337",
     "LEFT TURN EXECUTED  \u2713"),
    ("h3turn_C_textonly_s3_00001_.mp4", None,
     "C  TEXT timeline only, NO map   seed 20261006",
     "LEFT TURN EXECUTED  \u2713"),
]

FONT_BOLD = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 22)
FONT = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 20)
FONT_SM = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 16)


def load_map_thumb(path: Path, size: int = 224) -> Image.Image:
    im = Image.open(path).convert("RGB")
    im.thumbnail((size, size))
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    canvas.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
    return canvas


def draw_overlay(frame_rgb, thumb, top, bottom):
    img = Image.fromarray(frame_rgb)
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    # top-right caption
    bb = d.textbbox((0, 0), top, font=FONT_BOLD)
    tw = bb[2] - bb[0]
    rx = W - tw - 24
    d.rectangle([rx - 12, 14, W - 12, 48], fill=(0, 0, 0, 160))
    d.text((rx, 20), top, font=FONT_BOLD, fill=(255, 230, 80, 255))
    # bottom result bar
    bb = d.textbbox((0, 0), bottom, font=FONT)
    bw = bb[2] - bb[0]
    d.rectangle([0, H - 44, W, H], fill=(0, 0, 0, 170))
    d.text(((W - bw) // 2, H - 36), bottom, font=FONT,
           fill=(120, 255, 140, 255) if "EXECUTED" in bottom else (255, 255, 255, 255))
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    if thumb is not None:
        img.paste(thumb, (14, 14))
        d2 = ImageDraw.Draw(img)
        d2.rectangle([12, 12, 240, 240], outline=(255, 255, 255), width=2)
        d2.text((16, 242), "input map", font=FONT_SM, fill=(255, 255, 255))
    else:
        d2 = ImageDraw.Draw(img)
        d2.rectangle([14, 14, 190, 44], fill=(0, 0, 0, 170))
        d2.text((24, 20), "no map input", font=FONT_SM, fill=(200, 200, 200))
    return np.array(img)


def main():
    total = 0
    out = av.open(str(OUT_VID), mode="w")
    os_ = out.add_stream("h264", rate=24)
    os_.width, os_.height, os_.pix_fmt = W, H, "yuv420p"
    os_.options = {"crf": "20", "preset": "veryfast"}
    for i, (name, map_name, top, bottom) in enumerate(CLIPS, 1):
        p = OUT_DIR / name
        thumb = load_map_thumb(MAP_DIR / map_name) if map_name else None
        n = 0
        with av.open(str(p)) as inp:
            for frame in inp.decode(inp.streams.video[0]):
                arr = draw_overlay(frame.to_ndarray(format="rgb24"), thumb, top, bottom)
                for pkt in os_.encode(av.VideoFrame.from_ndarray(arr, format="rgb24")):
                    out.mux(pkt)
                n += 1
        total += n
        print(f"[{i}/{len(CLIPS)}] {name} ({n} frames)")
    for pkt in os_.encode():
        out.mux(pkt)
    out.close()
    print(f"DONE {OUT_VID}  {total} frames ~{total/24:.1f}s  {OUT_VID.stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
