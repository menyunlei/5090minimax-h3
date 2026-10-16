"""Final chain showreel: 1 single-clip undershoot contrast + 3 seeds x 3 chained segments.
10 clips x 5.17s = ~51.7s. Shows: single-clip text turn undershoots -> chained
approach/in-place-rotate/depart works."""
import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

OUT_DIR = Path(r"C:\Users\yunlei\AppData\Local\Comfy-Desktop\ComfyUI-Shared\output")
OUT_VID = Path(r"C:\Users\yunlei\.zcode\workspace\default\h3_map_baseline\showreel_chain.mp4")
W, H = 1344, 768

CLIPS = [
    ("h3turn_B_maptext_s1_00001_.mp4",
     "A1  single clip, text timeline (seed 42)",
     "turn UNDERSHOOTS: rotates only ~60deg, lands mid-intersection"),
    ("h3chain_seg1_s1_00001_.mp4",
     "S1  SEG 1/3 approach   map C2 + text   seed 42",
     "ride to the corner, STOP at the crosswalk facing the intersection"),
    ("h3chain_rot_s1_00001__00001_.mp4", None, None),
    ("h3seg3_s1_00001_.mp4",
     "S1  SEG 3/3 depart   I2V anchored at rot end   seed 42",
     "ride STRAIGHT down the new street (the cross street from SEG1)"),
    ("h3chain_seg1_s2_00001_.mp4",
     "S2  SEG 1/3 approach   map C2 + text   seed 1337",
     "ride to the corner, STOP at the crosswalk facing the intersection"),
    ("h3rot_s2_00001_.mp4",
     "S2  SEG 2/3 TURN in place   I2V anchored at SEG1 end   seed 1337",
     "rotate LEFT at the corner until facing the cross street"),
    ("h3seg3_s2_00001_.mp4",
     "S2  SEG 3/3 depart   I2V anchored at rot end   seed 1337",
     "ride STRAIGHT down the new street (the cross street from SEG1)"),
    ("h3chain_seg1_s3_00001_.mp4",
     "S3  SEG 1/3 approach   map C2 + text   seed 20261006",
     "ride to the corner, STOP at the crosswalk facing the intersection"),
    ("h3rot_s3_00001_.mp4",
     "S3  SEG 2/3 TURN in place   I2V anchored at SEG1 end   seed 20261006",
     "rotate LEFT at the corner until facing the cross street"),
    ("h3seg3_s3_00001_.mp4",
     "S3  SEG 3/3 depart   I2V anchored at rot end   seed 20261006",
     "ride STRAIGHT down the new street (the cross street from SEG1)"),
]
# fix rot clip names (files are h3rot_s{i}_00001_.mp4)
CLIPS[2] = ("h3rot_s1_00001_.mp4",
            "S1  SEG 2/3 TURN in place   I2V anchored at SEG1 end   seed 42",
            "rotate LEFT at the corner until facing the cross street")

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
        white = "UNDERSHOOTS" in bottom
        n = 0
        with av.open(str(p)) as inp:
            for frame in inp.decode(inp.streams.video[0]):
                arr = draw_overlay(frame.to_ndarray(format="rgb24"), top, bottom, white)
                for pkt in os_.encode(av.VideoFrame.from_ndarray(arr, format="rgb24")):
                    out.mux(pkt)
                n += 1
        total += n
        print(f"[{i}/{len(CLIPS)}] {name} ({n} frames)")
    for pkt in os_.encode():
        out.mux(pkt)
    out.close()
    print(f"DONE {OUT_VID}  {total} frames ~{total/24:.1f}s {OUT_VID.stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
