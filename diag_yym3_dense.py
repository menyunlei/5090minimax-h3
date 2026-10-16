# Filmstrip for the 241-frame v2 segments: 16 frames every 15 (~0.63s).
import av, os, sys
from PIL import Image, ImageDraw

OUT = r"C:\Users\yunlei\AppData\Local\Comfy-Desktop\ComfyUI-Shared\output"
DST = r"C:\Users\yunlei\.zcode\workspace\default\h3_map_baseline\yym3_diag"
os.makedirs(DST, exist_ok=True)

TW, TH = 448, 256
for clip in sys.argv[1:]:
    picks = list(range(0, 241, 15))
    c = av.open(os.path.join(OUT, clip))
    fr = {}
    for i, f in enumerate(c.decode(video=0)):
        if i in set(picks):
            fr[i] = Image.fromarray(f.to_ndarray(format="rgb24")).resize((TW, TH), Image.LANCZOS)
        if i > max(picks):
            break
    c.close()
    sheet = Image.new("RGB", (TW * 4, TH * 4), (10, 10, 10))
    d = ImageDraw.Draw(sheet)
    for k, idx in enumerate(picks):
        if idx not in fr:
            continue
        r, cc = divmod(k, 4)
        x, y = cc * TW, r * TH
        sheet.paste(fr[idx], (x, y))
        d.rectangle([x + 4, y + 4, x + 96, y + 26], fill=(0, 0, 0))
        d.text((x + 8, y + 8), f"f{idx} {idx/24:.2f}s", fill=(255, 255, 0))
    name = clip.replace("_00001_.mp4", "") + "_strip.png"
    sheet.save(os.path.join(DST, name))
    print("saved", name, len(fr))
print("DONE")
