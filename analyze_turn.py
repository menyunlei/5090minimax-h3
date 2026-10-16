# Turn analysis: for each turn-experiment video, extract 6 frames on a grid and
# estimate horizontal pan between consecutive frames via phase correlation.
# A 90-degree turn should show a large, consistent horizontal shift band during 2-4s;
# straight riding shows only small expansion-driven shifts.
import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

OUT_DIR = Path(r"C:\Users\yunlei\AppData\Local\Comfy-Desktop\ComfyUI-Shared\output")
AN_DIR = Path(r"C:\Users\yunlei\.zcode\workspace\default\h3_map_baseline\turn_analysis")
AN_DIR.mkdir(exist_ok=True)

VIDS = [f"h3turn_{c}_s{i}_00001_.mp4" for c in ("B_maptext", "C_textonly") for i in (1, 2, 3)]

# sample times (s) covering straight phase, turn window, post-turn phase
T_TIMES = [0.0, 1.0, 2.0, 2.7, 3.4, 4.1, 4.8, 5.1]
FPS = 24.0


def grab_frames(path: Path):
    with av.open(str(path)) as c:
        stream = c.streams.video[0]
        frames = [f.to_ndarray(format="rgb24") for f in c.decode(stream)]
    idx = [min(int(round(t * FPS)), len(frames) - 1) for t in T_TIMES]
    return frames, idx


def h_shift(a: np.ndarray, b: np.ndarray, max_shift: int = 300) -> float:
    """Dominant horizontal shift (px) from a to b via 1-D phase correlation of
    column-mean luminance profiles. Positive = scene moved right in b (camera panned left)."""
    ga = a[..., :3].mean(axis=(0, 2))  # per-column mean -> 1-D profile
    gb = b[..., :3].mean(axis=(0, 2))
    ga = ga - ga.mean(); gb = gb - gb.mean()
    n = len(ga)
    fa = np.fft.rfft(ga); fb = np.fft.rfft(gb)
    corr = np.fft.irfft(fa * np.conj(fb), n=n)
    # keep shifts in [-max_shift, max_shift]
    half = n // 2
    lag = np.concatenate([np.arange(0, max_shift + 1), np.arange(n - max_shift, n)])
    best = lag[np.argmax(corr[lag])]
    return best if best <= half else best - n


def pan_profile(frames, idx):
    """Per-interval horizontal shift between consecutive sampled frames (px)."""
    return [h_shift(frames[idx[i]], frames[idx[i + 1]]) for i in range(len(idx) - 1)]


FONT = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 20)
FONT_SM = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 16)


def contact_sheet(path: Path, frames, idx, pans) -> Path:
    thumbs = [Image.fromarray(frames[i]).resize((448, 256)) for i in idx]
    cols, rows = 4, 2
    W, H = cols * 448, rows * (256 + 54) + 60
    sheet = Image.new("RGB", (W, H), (20, 20, 20))
    d = ImageDraw.Draw(sheet)
    d.text((10, 8), path.name + "   (straight 0-2s | LEFT turn 2-4s | straight 4-5.2s)",
           font=FONT, fill=(255, 230, 80))
    for k, (im, t, p) in enumerate(zip(thumbs, T_TIMES, pans + [0.0])):
        r, c = divmod(k, cols)
        x, y = c * 448, 60 + r * (256 + 54)
        sheet.paste(im, (x, y))
        d.rectangle([x, y, x + 447, y + 255], outline=(90, 90, 90))
        pan = "" if k == 0 else f"  pan {p:+.0f}px"
        d.text((x + 6, y + 258), f"t={t:.1f}s{pan}", font=FONT_SM, fill=(255, 255, 255))
    out = AN_DIR / (path.stem + "_sheet.png")
    sheet.save(out)
    return out


def main():
    results = {}
    for name in VIDS:
        p = OUT_DIR / name
        if not p.exists():
            print("MISSING", name); continue
        frames, idx = grab_frames(p)
        pans = pan_profile(frames, idx)
        sheet = contact_sheet(p, frames, idx, pans)
        results[name] = pans
        print(f"{name}: pans(px) {pans}  -> {sheet.name}")
    np.save(AN_DIR / "pans.npy", results, allow_pickle=True)


if __name__ == "__main__":
    main()
