"""Clean conditioning map: thin red route + tiny markers, minimal graphic surface to prevent
appearance leak into generated video. The big-labelled yym_route_v4.png stays the USER/figure asset."""
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import shutil

SRC = Path(r"C:\Users\yunlei\AppData\Local\Comfy-Desktop\ComfyUI-Shared\input\yym_orig_nl.png")
OUT = Path(r"C:\Users\yunlei\.zcode\workspace\default\h3_map_baseline\yym_route_v4_clean.png")
IN = Path(r"C:\Users\yunlei\AppData\Local\Comfy-Desktop\ComfyUI-Shared\input\yym_route_v4_clean.png")

F = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 16)
im = Image.open(SRC).convert("RGB")
d = ImageDraw.Draw(im)
# base already has the red route; add tiny numbered waypoint tags only
d.text((126, 396), "W0 start", font=F, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
d.text((690, 396), "W1 turn", font=F, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
d.text((690, 884), "W2 end", font=F, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
# small north tick
d.line([40, 36, 40, 20], fill=(0, 0, 0), width=2)
d.text((34, 40), "N", font=F, fill=(0, 0, 0))
im.save(OUT)
shutil.copy(OUT, IN)
print("saved", OUT)
