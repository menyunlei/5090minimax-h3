# Dense filmstrip diagnosis for the Yuanmingyuan chain clips.
# 12 frames per clip at ~0.43s (every 10 frames), timestamp burned, 4 rows x 3 cols.
import av, os
import numpy as np
from PIL import Image, ImageDraw

OUT = r"C:\Users\yunlei\AppData\Local\Comfy-Desktop\ComfyUI-Shared\output"
DST = r"C:\Users\yunlei\.zcode\workspace\default\h3_map_baseline\yym_diag"
os.makedirs(DST, exist_ok=True)

CLIPS = [
    "h3yym_contrast_s1_00001_.mp4",
    "h3yym_seg1_s1_00001_.mp4", "h3yym_rot_s1_00001_.mp4", "h3yym_seg3_s1_00001_.mp4",
    "h3yym_seg1_s2_00001_.mp4", "h3yym_rot_s2_00001_.mp4", "h3yym_seg3_s2_00001_.mp4",
    "h3yym_seg1_s3_00001_.mp4", "h3yym_rot_s3_00001_.mp4", "h3yym_seg3_s3_00001_.mp4",
]

TW, TH = 448, 256  # tile size

def extract(path, picks):
    c = av.open(path)
    frames = {}
    want = set(picks)
    for i, f in enumerate(c.decode(video=0)):
        if i in want:
            a = f.to_ndarray(format="rgb24")
            frames[i] = Image.fromarray(a).resize((TW, TH), Image.LANCZOS)
        if i > max(picks):
            break
    c.close()
    return frames

for clip in CLIPS:
    picks = list(range(0, 120, 10))  # 0..110 -> 12 frames, ~0.43s steps
    fr = extract(os.path.join(OUT, clip), picks)
    W, H = TW * 3, TH * 4
    sheet = Image.new("RGB", (W, H), (10, 10, 10))
    d = ImageDraw.Draw(sheet)
    for k, idx in enumerate(picks):
        if idx not in fr:
            continue
        r, ccol = divmod(k, 3)
        x, y = ccol * TW, r * TH
        sheet.paste(fr[idx], (x, y))
        t = idx / 24.0
        d.rectangle([x + 4, y + 4, x + 96, y + 26], fill=(0, 0, 0))
        d.text((x + 8, y + 8), f"f{idx} {t:.2f}s", fill=(255, 255, 0))
    name = clip.replace("_00001_.mp4", "") + "_strip.png"
    sheet.save(os.path.join(DST, name))
    print("saved", name, "frames:", len(fr))
print("DONE")
