"""Extract last frame of a video to PNG, into ComfyUI Shared input dir.
Usage: python last_frame.py <video_path> <out_png_name>"""
import av, sys
from pathlib import Path
from PIL import Image

SHARED_IN = Path(r"C:\Users\yunlei\AppData\Local\Comfy-Desktop\ComfyUI-Shared\input")
vid, name = sys.argv[1], sys.argv[2]
last = None
n = 0
with av.open(vid) as c:
    for f in c.decode(c.streams.video[0]):
        last = f.to_ndarray(format="rgb24")
        n += 1
Image.fromarray(last).save(SHARED_IN / name)
print(f"frames={n} saved {SHARED_IN / name}")
