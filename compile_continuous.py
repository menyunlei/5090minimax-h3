"""Continuous 60s ride: 12 chained segments for seed 42, anchor-dup frames dropped at joins,
one FIXED overlay for the whole video (map inset + caption + tiny segment counter)."""
import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

OUT_DIR = Path(r"C:\Users\yunlei\AppData\Local\Comfy-Desktop\ComfyUI-Shared\output")
OUT_VID = Path(r"C:\Users\yunlei\.zcode\workspace\default\h3_map_baseline\continuous_yym_60s.mp4")
MAP_IMG = Path(r"C:\Users\yunlei\.zcode\workspace\default\h3_map_baseline\maps\yym_orig_nl.png")
W, H = 1344, 768

SEGMENTS = [
    ("h3yym2_seg1_s1_00001_.mp4", "approach (map route)"),
    ("h3yym2_rot_s1_00001_.mp4", "turn LEFT in place"),
    ("h3yym2_seg3_s1_00001_.mp4", "causeway crossing"),
    ("h3yym2_cnt4_00001_.mp4", "to the island gate"),
    ("h3yym2_cnt5_00001_.mp4", "through the pavilion"),
    ("h3yym2_cnt6_00001_.mp4", "island shore"),
    ("h3yym2_cnt7_00001_.mp4", "arrive at small bridge"),
    ("h3yym2_cnt8_00001_.mp4", "turn RIGHT in place"),
    ("h3yym2_cnt9_00001_.mp4", "small bridge crossing"),
    ("h3yym2_cnt10_00001_.mp4", "willow shore"),
    ("h3yym2_cnt11_00001_.mp4", "lake lookout stop"),
    ("h3yym2_cnt12_00001_.mp4", "ending panorama pan"),
]

FONT_BOLD = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 22)
FONT = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 18)

MAP_THUMB = Image.open(MAP_IMG).convert("RGB").resize((224, 224), Image.LANCZOS)


def draw_overlay(rgb, seg_idx, seg_label):
    img = Image.fromarray(rgb)
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    d.rectangle([12, 12, 240, 240], fill=(0, 0, 0, 160))
    top = "Yuanmingyuan map -> ONE continuous chained ride   12 segments, handlebar FPV   seed 42"
    bb = d.textbbox((0, 0), top, font=FONT_BOLD)
    rx = W - (bb[2] - bb[0]) - 24
    d.rectangle([rx - 12, 14, W - 12, 48], fill=(0, 0, 0, 160))
    d.text((rx, 20), top, font=FONT_BOLD, fill=(255, 230, 80, 255))
    d.rectangle([0, H - 40, W, H], fill=(0, 0, 0, 170))
    note = "no cuts -- segments joined on anchor frames"
    bb2 = d.textbbox((0, 0), note, font=FONT)
    d.text(((W - (bb2[2] - bb2[0])) // 2, H - 33), note, font=FONT, fill=(200, 200, 205, 255))
    d.text((16, H - 33), f"SEG {seg_idx}/12  {seg_label}", font=FONT, fill=(120, 255, 140, 255))
    out = Image.alpha_composite(img.convert("RGBA"), ov).convert("RGB")
    out.paste(MAP_THUMB, (16, 16))
    return np.array(out)


def main():
    total = 0
    out = av.open(str(OUT_VID), mode="w")
    os_ = out.add_stream("h264", rate=24)
    os_.width, os_.height, os_.pix_fmt = W, H, "yuv420p"
    os_.options = {"crf": "20", "preset": "veryfast"}
    for si, (name, label) in enumerate(SEGMENTS, 1):
        p = OUT_DIR / name
        if not p.exists():
            print("MISSING", name, "-- chain incomplete"); return
        n = 0
        with av.open(str(p)) as inp:
            for fi, frame in enumerate(inp.decode(inp.streams.video[0])):
                if si > 1 and fi == 0:
                    continue  # anchor duplicate of previous segment's last frame
                arr = draw_overlay(frame.to_ndarray(format="rgb24"), si, label)
                for pkt in os_.encode(av.VideoFrame.from_ndarray(arr, format="rgb24")):
                    out.mux(pkt)
                n += 1
        total += n
        print(f"seg {si:2d}: {name} (+{n} frames)")
    for pkt in os_.encode():
        out.mux(pkt)
    out.close()
    print(f"DONE {OUT_VID}  {total} frames ~{total/24:.1f}s {OUT_VID.stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
