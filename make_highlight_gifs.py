# Generate README highlight GIFs (PyAV decode -> Pillow quantize -> animated GIF).
import av, os
from PIL import Image

BASE = r"C:\Users\yunlei\.zcode\workspace\default\h3_map_baseline"
OUTD = os.path.join(BASE, "assets")
os.makedirs(OUTD, exist_ok=True)

CLIPS = [
    # (src, start_sec, dur_sec, out_name)
    ("continuous_m1_v1.mp4",      11.0, 5.0, "m1_turn.gif"),    # rolling 90-deg right at the real T-junction
    ("continuous_m1_v1.mp4",       0.0, 5.0, "m1_start.gif"),   # quiet-street opening, dot at route start
    ("continuous_r1_v2.mp4",       5.5, 5.0, "r1_bridge.gif"),  # cresting the stone arch bridge
    ("continuous_yym_v3.mp4",     10.2, 5.0, "yym_causeway.gif"),  # causeway, lotus lake both sides
]
FPS_OUT, WIDTH = 10, 480

for src, t0, dur, name in CLIPS:
    path = os.path.join(BASE, src)
    c = av.open(path)
    stream = c.streams.video[0]
    step = 24 // FPS_OUT  # source is 24fps
    tgt = int(dur * FPS_OUT)
    frames = []
    idx = -1
    start_f = int(t0 * 24)
    for f in c.decode(stream):
        idx += 1
        if idx < start_f:
            continue
        if (idx - start_f) % step == 0:
            im = f.to_image().convert("RGB")
            h = int(im.height * WIDTH / im.width)
            im = im.resize((WIDTH, h), Image.LANCZOS)
            frames.append(im)
            if len(frames) >= tgt:
                break
    c.close()
    out = os.path.join(OUTD, name)
    gif = frames[0].convert("P", palette=Image.ADAPTIVE, colors=256)
    rest = [x.convert("P", palette=Image.ADAPTIVE, colors=256) for x in frames[1:]]
    gif.save(out, save_all=True, append_images=rest, duration=int(1000 / FPS_OUT),
             loop=0, optimize=False, dither=Image.FLOYDSTEINBERG)
    mb = os.path.getsize(out) / 1048576
    print(f"{name}: {len(frames)}f {mb:.1f}MB")
