# Dense filmstrips through the turn window + robust 2-D phase correlation.
# 1) Filmstrip: every 0.25s from 1.5s to 4.75s (14 frames) for chosen videos.
# 2) 2-D phase correlation on downsampled grayscale between frames 0.5s apart:
#    estimates (dx, dy) shift with a confidence = peak height / corr mean.
import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

OUT_DIR = Path(r"C:\Users\yunlei\AppData\Local\Comfy-Desktop\ComfyUI-Shared\output")
AN_DIR = Path(r"C:\Users\yunlei\.zcode\workspace\default\h3_map_baseline\turn_analysis")
FPS = 24.0
FONT = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 20)
FONT_SM = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 15)

VIDS = ["h3turn_B_maptext_s2_00001_.mp4", "h3turn_C_textonly_s2_00001_.mp4",
        "h3turn_B_maptext_s1_00001_.mp4", "h3turn_C_textonly_s3_00001_.mp4"]


def decode(path):
    with av.open(str(path)) as c:
        return [f.to_ndarray(format="rgb24") for f in c.decode(c.streams.video[0])]


def filmstrip(name, frames):
    times = [1.5 + 0.25 * k for k in range(14)]
    idx = [min(int(round(t * FPS)), len(frames) - 1) for t in times]
    thumbs = [Image.fromarray(frames[i]).resize((384, 219)) for i in idx]
    cols, rows = 7, 2
    W, H = cols * 384, 40 + rows * (219 + 22)
    sheet = Image.new("RGB", (W, H), (18, 18, 18))
    d = ImageDraw.Draw(sheet)
    d.text((8, 6), name + "   turn window 1.5s-4.75s, step 0.25s", font=FONT, fill=(255, 230, 80))
    for k, (im, t) in enumerate(zip(thumbs, times)):
        r, c = divmod(k, cols)
        x, y = c * 384, 40 + r * (219 + 22)
        sheet.paste(im, (x, y))
        d.text((x + 4, y + 220), f"t={t:.2f}s", font=FONT_SM, fill=(240, 240, 240))
    out = AN_DIR / (Path(name).stem + "_dense.png")
    sheet.save(out)
    print("saved", out.name)


def shift2d(a, b, w=336):
    """Peak (dx,dy) shift a->b on downsampled grayscale; positive dx = content moved right = camera yaw LEFT."""
    ia = np.asarray(Image.fromarray(a).convert("L").resize((w, 192)), float)
    ib = np.asarray(Image.fromarray(b).convert("L").resize((w, 192)), float)
    ia -= ia.mean(); ib -= ib.mean()
    fa = np.fft.rfft2(ia); fb = np.fft.rfft2(ib)
    cross = fa * np.conj(fb)
    cross /= np.maximum(np.abs(cross), 1e-9)  # phase correlation
    p = np.abs(np.fft.irfft2(cross, s=ia.shape))
    dy, dx = np.unravel_index(np.argmax(p), p.shape)
    if dx > w // 2: dx -= w
    if dy > 192 // 2: dy -= 192
    conf = p[dy % 192, dx % w] / (p.mean() + 1e-12)
    return dx, dy, float(conf)


def motion_report(name, frames):
    # pairs 0.5s apart across the whole video
    step = int(0.5 * FPS)
    rows = []
    scale = 1344 / 336
    for i in range(0, len(frames) - step, step):
        t = i / FPS
        dx, dy, conf = shift2d(frames[i], frames[i + step])
        dx *= scale
        yaw = dx / (1344 / 90)  # px -> deg, assuming 90deg hFOV
        rows.append((t, dx, dy, conf, yaw))
    print(f"\n{name}   (pairs 0.5s apart; dx>0 = content right = yaw LEFT)")
    for t, dx, dy, conf, yaw in rows:
        tag = "" if conf > 50 else "  <- weak corr!"
        print(f"  {t:4.1f}s -> {t+0.5:4.1f}s  dx={dx:+6.0f}px dy={dy:+5.0f}px conf={conf:5.1f} yaw={yaw:+6.1f}deg{tag}")


def main():
    for name in VIDS:
        frames = decode(OUT_DIR / name)
        filmstrip(name, frames)
        motion_report(name, frames)


if __name__ == "__main__":
    main()
