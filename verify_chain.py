# Verify the chained turn:
#  - seam check: pixel diff between seg1 last frame and seg2 frame 0
#  - yaw accumulation inside seg2 via 2-D phase corr, 0.25s steps, 0-3.5s
#  - filmstrip per seed (0s..5.1s)
import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

OUT_DIR = Path(r"C:\Users\yunlei\AppData\Local\Comfy-Desktop\ComfyUI-Shared\output")
AN_DIR = Path(r"C:\Users\yunlei\.zcode\workspace\default\h3_map_baseline\turn_analysis")
FPS = 24.0
FONT = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 20)
FONT_SM = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 15)


def decode(path):
    with av.open(str(path)) as c:
        return [f.to_ndarray(format="rgb24") for f in c.decode(c.streams.video[0])]


def shift2d(a, b, w=336):
    ia = np.asarray(Image.fromarray(a).convert("L").resize((w, 192)), float)
    ib = np.asarray(Image.fromarray(b).convert("L").resize((w, 192)), float)
    ia -= ia.mean(); ib -= ib.mean()
    fa = np.fft.rfft2(ia); fb = np.fft.rfft2(ib)
    cross = fa * np.conj(fb)
    cross /= np.maximum(np.abs(cross), 1e-9)
    p = np.abs(np.fft.irfft2(cross, s=ia.shape))
    dy, dx = np.unravel_index(np.argmax(p), p.shape)
    if dx > w // 2: dx -= w
    if dy > 192 // 2: dy -= 192
    conf = p[dy % 192, dx % w] / (p.mean() + 1e-12)
    # negative dx (measured) = content moved right = camera yawed LEFT
    return dx, dy, float(conf)


def main():
    for si in (1, 2, 3):
        seg1 = decode(OUT_DIR / f"h3chain_seg1_s{si}_00001_.mp4")
        seg2 = decode(OUT_DIR / f"h3chain_seg2_s{si}_00001_.mp4")
        seam = np.abs(seg1[-1].astype(float) - seg2[0].astype(float)).mean()
        # yaw accumulation over seg2 first 3.5s, 0.25s steps
        step = 6
        cum = 0.0; rows = []
        scale = 1344 / 336
        for i in range(0, int(3.5 * FPS), step):
            dx, dy, conf = shift2d(seg2[i], seg2[i + step])
            yaw = (-dx) * scale / (1344 / 90.0)  # LEFT positive
            cum += yaw
            rows.append((i / FPS, yaw, conf))
        # filmstrip
        times = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.25, 5.1]
        idx = [min(int(round(t * FPS)), len(seg2) - 1) for t in times]
        thumbs = [Image.fromarray(seg2[i]).resize((430, 246)) for i in idx]
        cols, rows_n = 5, 2
        W, H = cols * 430, 34 + rows_n * (246 + 20)
        sheet = Image.new("RGB", (W, H), (18, 18, 18))
        d = ImageDraw.Draw(sheet)
        d.text((8, 4), f"h3chain_seg2_s{si}  (anchored 90-deg LEFT turn 0-2.5s)",
               font=FONT, fill=(255, 230, 80))
        for k, (im, t) in enumerate(zip(thumbs, times)):
            r, c = divmod(k, cols)
            x, y = c * 430, 34 + r * (246 + 20)
            sheet.paste(im, (x, y))
            d.text((x + 4, y + 247), f"t={t:.1f}s", font=FONT_SM, fill=(240, 240, 240))
        out = AN_DIR / f"seg2_s{si}_strip.png"
        sheet.save(out)
        print(f"seed s{si}: seam_diff={seam:.1f}/255   cum_yaw(seg2 0-3.5s)={cum:+.1f} deg")
        print("   " + " ".join(f"{t:.2f}s:{y:+.1f}" for t, y, cf in rows))


if __name__ == "__main__":
    main()
