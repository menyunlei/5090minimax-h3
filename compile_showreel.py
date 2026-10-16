"""Compile a labelled 50s+ showreel from the multiseed no-legend H3 outputs.

Picks 10 clips (4 cases x orig+edit at seed 1, plus C2 seed 2 to show seed stability),
overlays each frame with a map thumbnail + case/seed/result caption, and concatenates
into a single h264 .mp4 (muted — visuals are the point of the experiment).
"""
from __future__ import annotations
import av
import numpy as np
from fractions import Fraction
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

OUT_DIR = Path(r"C:\Users\yunlei\AppData\Local\Comfy-Desktop\ComfyUI-Shared\output")
MAP_DIR = Path(r"C:\Users\yunlei\.zcode\workspace\default\h3_map_baseline\maps")
OUT_VID = Path(r"C:\Users\yunlei\.zcode\workspace\default\h3_map_baseline\showreel_multiseed.mp4")

W, H = 1344, 768

# (file_stem, map_file, caption_top, caption_bottom)
# seed 1 = 42, seed 2 = 1337, seed 3 = 20261006
CLIPS = [
    ("h3mapNL_C1_orig_s1_00001_", "C1_orig_nl.png",
     "C1  sides swap   |   variant: ORIG   |   seed 42",
     "result: buildings on BOTH sides — side spec IGNORED  (0/3 seeds pass)"),
    ("h3mapNL_C1_edit_s1_00001_", "C1_edit_nl.png",
     "C1  sides swap   |   variant: EDIT (swap L/R building sides)   |   seed 42",
     "result: sides still wrong — edit causally felt but not translated"),
    ("h3mapNL_C2_orig_s1_00001_", "C2_orig_nl.png",
     "C2  left/right turn   |   variant: ORIG (left turn)   |   seed 42",
     "result: camera goes STRAIGHT — red route curve never executed"),
    ("h3mapNL_C2_edit_s1_00001_", "C2_edit_nl.png",
     "C2  left/right turn   |   variant: EDIT (right turn)   |   seed 42",
     "result: straight again — turn 6/6 fail across all seeds"),
    ("h3mapNL_C3_orig_s1_00001_", "C3_orig_nl.png",
     "C3  courtyard -> solid block   |   variant: ORIG (courtyard)   |   seed 42",
     "result: weak directional signal only"),
    ("h3mapNL_C3_edit_s1_00001_", "C3_edit_nl.png",
     "C3  courtyard -> solid block   |   variant: EDIT (filled block)   |   seed 42",
     "result: partial local change — not a strict pass"),
    ("h3mapNL_C4_orig_s1_00001_", "C4_orig_nl.png",
     "C4  water -> vegetation   |   variant: ORIG (water region)   |   seed 42",
     "result: WATER completely ABSENT from the world  (3/3 seeds)"),
    ("h3mapNL_C4_edit_s1_00001_", "C4_edit_nl.png",
     "C4  water -> vegetation   |   variant: EDIT (veg region)   |   seed 42",
     "result: veg edit felt causally, but orig water never appeared"),
    ("h3mapNL_C2_orig_s2_00001_", "C2_orig_nl.png",
     "C2  left turn   |   ORIG   |   seed 1337   (seed-stability check)",
     "result: still straight — turn failure replicates across seeds"),
    ("h3mapNL_C2_edit_s2_00001_", "C2_edit_nl.png",
     "C2  right turn   |   EDIT   |   seed 1337   (seed-stability check)",
     "result: still straight — turn failure replicates across seeds"),
]

# fonts
FONT_BOLD = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 22)
FONT      = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 20)
FONT_SM   = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 16)


def load_map_thumb(path: Path, size: int = 224) -> Image.Image:
    im = Image.open(path).convert("RGB")
    im.thumbnail((size, size))
    # pad to square on white, with 2px white border
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    off = ((size - im.width) // 2, (size - im.height) // 2)
    canvas.paste(im, off)
    return canvas


def draw_overlay(frame_rgb: np.ndarray, thumb: Image.Image, top: str, bottom: str) -> np.ndarray:
    img = Image.fromarray(frame_rgb)
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)

    # --- top-left: map thumbnail with border + label ---
    tx, ty = 14, 14
    tw, th = thumb.size
    # semi-transparent backing
    d.rectangle([tx - 4, ty - 4, tx + tw + 4, ty + th + 28], fill=(0, 0, 0, 150))
    # border
    d.rectangle([tx - 2, ty - 2, tx + tw + 2, ty + th + 2], outline=(255, 255, 255, 230), width=2)
    d.text((tx + 2, ty + th + 4), "input map", font=FONT_SM, fill=(255, 255, 255, 230))

    # --- top-right: case caption ---
    tw_bbox = d.textbbox((0, 0), top, font=FONT_BOLD)
    tw_w = tw_bbox[2] - tw_bbox[0]
    rx = W - tw_w - 24
    d.rectangle([rx - 12, 14, W - 12, 14 + 34], fill=(0, 0, 0, 160))
    d.text((rx, 20), top, font=FONT_BOLD, fill=(255, 230, 80, 255))

    # --- bottom: result bar ---
    bb = d.textbbox((0, 0), bottom, font=FONT)
    bw = bb[2] - bb[0]
    bx = (W - bw) // 2
    d.rectangle([0, H - 44, W, H], fill=(0, 0, 0, 170))
    d.text((bx, H - 36), bottom, font=FONT, fill=(255, 255, 255, 255))

    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    # paste thumb (it's RGB, onto the now-RGB img)
    img.paste(thumb, (tx, ty))
    return np.array(img)


def main() -> None:
    total_frames = 0
    out = av.open(str(OUT_VID), mode="w")
    ostream = out.add_stream("h264", rate=Fraction(24, 1))
    ostream.width = W
    ostream.height = H
    ostream.pix_fmt = "yuv420p"
    ostream.options = {"crf": "20", "preset": "veryfast"}

    for i, (stem, map_name, top, bottom) in enumerate(CLIPS, 1):
        clip_path = OUT_DIR / (stem + ".mp4")
        thumb = load_map_thumb(MAP_DIR / map_name)
        print(f"[{i}/{len(CLIPS)}] {stem}  ({clip_path.stat().st_size/1e6:.1f} MB)")
        n = 0
        with av.open(str(clip_path)) as inp:
            istream = inp.streams.video[0]
            for frame in inp.decode(istream):
                arr = frame.to_ndarray(format="rgb24")
                arr = draw_overlay(arr, thumb, top, bottom)
                out_frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
                for pkt in ostream.encode(out_frame):
                    out.mux(pkt)
                n += 1
        total_frames += n
        print(f"    decoded+re-encoded {n} frames")

    # flush
    for pkt in ostream.encode():
        out.mux(pkt)
    out.close()
    dur = total_frames / 24.0
    print(f"\nDONE  {OUT_VID}")
    print(f"      {total_frames} frames, ~{dur:.1f} s, {OUT_VID.stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
