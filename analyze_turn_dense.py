# Dense per-frame yaw accumulation for the turn-experiment videos.
# For each video: shift between EVERY consecutive frame via 1-D phase correlation
# on the column-mean luminance profile, clamped to a plausible per-frame range,
# then accumulated. Also reports correlation peak strength per frame to detect
# scene morphs (weak peaks = the two frames are unrelated, shift is noise).
# Calibration: 1344 px wide; if hFOV~90deg then ~14.9 px per degree of yaw.
import av
import numpy as np
from pathlib import Path

OUT_DIR = Path(r"C:\Users\yunlei\AppData\Local\Comfy-Desktop\ComfyUI-Shared\output")
VIDS = [f"h3turn_{c}_s{i}_00001_.mp4" for c in ("B_maptext", "C_textonly") for i in (1, 2, 3)]
FPS = 24.0
PX_PER_DEG = 1344 / 90.0  # assumes ~90 deg horizontal FOV


def col_profile(a):
    g = a[..., :3].mean(axis=(0, 2))
    return g - g.mean()


def frame_shift(ga, gb, clamp=150):
    n = len(ga)
    fa = np.fft.rfft(ga); fb = np.fft.rfft(gb)
    corr = np.fft.irfft(fa * np.conj(fb), n=n)
    half = n // 2
    pos = corr[: clamp + 1]
    neg = corr[n - clamp:]
    if pos.max() >= neg.max():
        shift = int(np.argmax(pos)); strength = pos.max()
    else:
        shift = int(np.argmax(neg)) - n; strength = neg.max()
    return shift, float(strength)


def analyze(path):
    with av.open(str(path)) as c:
        frames = [f.to_ndarray(format="rgb24") for f in c.decode(c.streams.video[0])]
    profs = [col_profile(f) for f in frames]
    shifts, strengths = [], []
    for i in range(len(profs) - 1):
        s, st = frame_shift(profs[i], profs[i + 1])
        shifts.append(s); strengths.append(st)
    shifts = np.array(shifts, float)
    strengths = np.array(strengths, float)
    ref = np.median(strengths)
    good = strengths > 0.3 * ref  # strong-correlation frames only
    cum = np.cumsum(np.where(good, shifts, 0.0))
    return cum, good, strengths


def main():
    for name in VIDS:
        p = OUT_DIR / name
        cum, good, strengths = analyze(p)
        total = cum[-1]
        # window sums (index i = shift from frame i to i+1)
        w_pre = cum[int(2.0 * FPS)] if len(cum) > int(2.0 * FPS) else 0
        w_turn = cum[min(int(4.0 * FPS), len(cum) - 1)] - w_pre
        n_weak = int((~good).sum())
        print(f"{name}")
        print(f"  total yaw over 5.2s : {total:7.0f}px  = {total / PX_PER_DEG:5.1f} deg"
              f"  (weak-corr frames masked: {n_weak}/{len(good)})")
        print(f"  yaw during 2.0-4.0s  : {w_turn:7.0f}px  = {w_turn / PX_PER_DEG:5.1f} deg")
        # timeline of cumulative yaw every 0.5s
        marks = [c / PX_PER_DEG for c in cum[::12][:11]]
        print(f"  cum yaw @0,0.5,...5s  : " + " ".join(f"{m:+5.1f}" for m in marks))


if __name__ == "__main__":
    main()
