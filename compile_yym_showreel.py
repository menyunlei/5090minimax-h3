"""Yuanmingyuan showreel: single-clip contrast + 3 seeds x chained segments. ~51.7s."""
import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

OUT_DIR = Path(r"C:\Users\yunlei\AppData\Local\Comfy-Desktop\ComfyUI-Shared\output")
OUT_VID = Path(r"C:\Users\yunlei\.zcode\workspace\default\h3_map_baseline\showreel_yym.mp4")
W, H = 1344, 768

CLIPS = [
    ("h3yym_contrast_s1_00001_.mp4",
     "CONTRAST: one single clip, map + text timeline, Yuanmingyuan park   seed 42",
     "it turned... by cheating: drone-height aerial view, 1.6m bicycle POV spec violated"),
    ("h3yym_seg1_s1_00001_.mp4",
     "S1  SEG 1/3 approach   map yym + text   seed 42",
     "ride the tree-lined path (lake left), STOP at the junction plaza"),
    ("h3yym_rot_s1_00001_.mp4",
     "S1  SEG 2/3 TURN in place   anchored at SEG1 end   seed 42",
     "rotate LEFT until facing the causeway, water on both sides"),
    ("h3yym_seg3_s1_00001_.mp4",
     "S1  SEG 3/3 depart   anchored at rot end   seed 42",
     "ride STRAIGHT along the causeway, water on both sides"),
    ("h3yym_seg1_s2_00001_.mp4",
     "S2  SEG 1/3 approach   map yym + text   seed 1337",
     "ride the tree-lined path (lake left), STOP at the junction plaza"),
    ("h3yym_rot_s2_00001_.mp4",
     "S2  SEG 2/3 TURN in place   anchored at SEG1 end   seed 1337",
     "rotate LEFT until facing the causeway, water on both sides"),
    ("h3yym_seg3_s2_00001_.mp4",
     "S2  SEG 3/3 depart   anchored at rot end   seed 1337",
     "ride STRAIGHT along the causeway, water on both sides"),
    ("h3yym_seg1_s3_00001_.mp4",
     "S3  SEG 1/3 approach   map yym + text   seed 20261006",
     "ride the tree-lined path (lake left), STOP at the junction plaza"),
    ("h3yym_rot_s3_00001_.mp4",
     "S3  SEG 2/3 TURN in place   anchored at SEG1 end   seed 20261006",
     "rotate LEFT until facing the causeway, water on both sides"),
    ("h3yym_seg3_s3_00001_.mp4",
     "S3  SEG 3/3 depart   anchored at rot end   seed 20261006",
     "ride STRAIGHT along the causeway, water on both sides"),
]

FONT_BOLD = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 22)
FONT = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 20)


def draw_overlay(rgb, top, bottom, white):
    img = Image.fromarray(rgb)
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    bb = d.textbbox((0, 0), top, font=FONT_BOLD)
    rx = W - (bb[2] - bb[0]) - 24
    d.rectangle([rx - 12, 14, W - 12, 48], fill=(0, 0, 0, 160))
    d.text((rx, 20), top, font=FONT_BOLD, fill=(255, 230, 80, 255))
    bb = d.textbbox((0, 0), bottom, font=FONT)
    d.rectangle([0, H - 44, W, H], fill=(0, 0, 0, 170))
    d.text(((W - (bb[2] - bb[0])) // 2, H - 36), bottom, font=FONT,
           fill=(255, 255, 255, 255) if white else (120, 255, 140, 255))
    return np.array(Image.alpha_composite(img.convert("RGBA"), ov).convert("RGB"))


def main():
    total = 0
    out = av.open(str(OUT_VID), mode="w")
    os_ = out.add_stream("h264", rate=24)
    os_.width, os_.height, os_.pix_fmt = W, H, "yuv420p"
    os_.options = {"crf": "20", "preset": "veryfast"}
    for i, (name, top, bottom) in enumerate(CLIPS, 1):
        p = OUT_DIR / name
        if not p.exists():
            print("MISSING", name); continue
        white = "CONTRAST" in top
        n = 0
        with av.open(str(p)) as inp:
            for frame in inp.decode(inp.streams.video[0]):
                arr = draw_overlay(frame.to_ndarray(format="rgb24"), top, bottom, white)
                for pkt in os_.encode(av.VideoFrame.from_ndarray(arr, format="rgb24")):
                    out.mux(pkt)
                n += 1
        total += n
        print(f"[{i}/{len(CLIPS)}] {name} ({n})")
    for pkt in os_.encode():
        out.mux(pkt)
    out.close()
    print(f"DONE {OUT_VID}  {total} frames ~{total/24:.1f}s {OUT_VID.stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
