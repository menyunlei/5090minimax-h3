# Verify v2 chain seams: |seg1 last - rot first| and |rot last - seg3 first| per seed.
import av
import numpy as np
from pathlib import Path

OUT_DIR = Path(r"C:\Users\yunlei\AppData\Local\Comfy-Desktop\ComfyUI-Shared\output")

def frame_at(p, idx):
    with av.open(str(p)) as c:
        fr = None
        for i, f in enumerate(c.decode(c.streams.video[0])):
            if i == idx:
                return f.to_ndarray(format="rgb24").astype(np.float32)
            fr = f.to_ndarray(format="rgb24")
        return fr.astype(np.float32)  # last

for si in (1, 2, 3):
    seg1 = OUT_DIR / f"h3yym2_seg1_s{si}_00001_.mp4"
    rot = OUT_DIR / f"h3yym2_rot_s{si}_00001_.mp4"
    seg3 = OUT_DIR / f"h3yym2_seg3_s{si}_00001_.mp4"
    d1 = np.abs(frame_at(seg1, -1) - frame_at(rot, 0)).mean()
    d2 = np.abs(frame_at(rot, -1) - frame_at(seg3, 0)).mean()
    print(f"seed{si}: seg1->rot seam {d1:.2f}/255   rot->seg3 seam {d2:.2f}/255")
