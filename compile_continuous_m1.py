# M1 real-satellite-map continuous ride: concat h3m1_s1..s6 (drop join dupes),
# fixed caption bar + tall minimap (the REAL satellite route overlay) with a
# MOVING position dot + traveled trail. Output continuous_m1_v1.mp4.
import av, os
from PIL import Image, ImageDraw, ImageFont

BASE = r"C:\Users\yunlei\.zcode\workspace\default\h3_map_baseline"
OUT  = r"C:\Users\yunlei\AppData\Local\Comfy-Desktop\ComfyUI-Shared\output"
CLIP_NAMES = [f"h3m1_s{i}_00001_.mp4" for i in range(1, 7)]
RESULT = os.path.join(BASE, "continuous_m1_v1.mp4")

MAP_IMG = os.path.join(BASE, "maps", "M1_route.png")   # 872x1630 with red polyline
MAP_H = 500                                             # thumb height on frame
MAP_W, MAP_H_FULL = 872, 1630

CAPTION = "One continuous ride through a real residential neighbourhood - satellite-map anchored - MiniMax H3 chain of 6"

# route polyline in original 872x1630 map coords
POLY = [(300,1620),(335,1330),(360,1180),(395,970),(430,810),(500,758),(700,700),(830,652)]

# (frame, x, y) keyframes along the polyline; linear interp in between.
# S boundaries: S1 0-242 *new-frames*; each following clip adds 242 frames.
WAY = [
    (0,    300,1620),  # S1 start
    (242,  360,1180),  # S1 end: rolling toward the junction, just before the S-curve
    (484,  560, 730),  # S2 end: after the RIGHT 90 arc, mid-cruise on the main road
    (726,  700, 700),  # S3 end: main road straight
    (968,  790, 665),  # S4 end: canopy->hedge transition, faint rise bend
    (1210, 820, 658),  # S5 end: approaching the kerbside pull-out
    (1360, 830, 652),  # S6 mid: glide done, parked at pull-out, slow pan left
    (1452, 830, 652),  # end
]

def pos_at(f):
    if f <= WAY[0][0]: return WAY[0][1], WAY[0][2]
    if f >= WAY[-1][0]: return WAY[-1][1], WAY[-1][2]
    for (f0,x0,y0),(f1,x1,y1) in zip(WAY, WAY[1:]):
        if f0 <= f <= f1:
            t = 0 if f1==f0 else (f-f0)/(f1-f0)
            return x0+(x1-x0)*t, y0+(y1-y0)*t
    return WAY[-1][1], WAY[-1][2]

def trail_upto(pt):
    """walk POLY, return partial polyline ending at interpolated pt"""
    pts = [POLY[0]]
    for a,b in zip(POLY, POLY[1:]):
        ax,ay,bx,by = *a,*b
        # param of pt projected onto segment ab
        dx,dy = bx-ax, by-ay
        L2 = dx*dx+dy*dy
        if L2==0: continue
        t = ((pt[0]-ax)*dx + (pt[1]-ay)*dy)/L2
        if t >= 1: pts.append(b); continue
        if t <= 0: break
        pts.append((ax+dx*t, ay+dy*t)); break
    return pts

# ---- load minimap ----
thumb = Image.open(MAP_IMG).convert("RGB")
s = MAP_H / thumb.height
TW = int(thumb.width*s)
thumb = thumb.resize((TW, MAP_H), Image.LANCZOS)
# add thin white border
framed = Image.new("RGB", (TW+4, MAP_H+4), (255,255,255))
framed.paste(thumb, (2,2))

# fonts
def font(sz):
    for name in ("arialbd.ttf","arial.ttf","segoeui.ttf"):
        try: return ImageFont.truetype(name, sz)
        except Exception: pass
    return ImageFont.load_default()
F_CAP = font(30)

# ---- video pass ----
ins = []
for n in CLIP_NAMES:
    c = av.open(os.path.join(OUT, n))
    ins.append(c)
    print("open", n, "streams:", len(c.streams.video))

oc = av.open(RESULT, mode="w")
vs = oc.add_stream("libx264", rate=24)
vs.width, vs.height, vs.pix_fmt = 1344, 768, "yuv420p"
vs.options = {"crf":"19","preset":"medium"}

# audio: take from clip 1, copy/pad to total length after video pass would be
# complex; simpler: re-encode audio afterwards. For now include audio pass-through
# from concatenated clips is non-trivial with PyAV when sources have different
# timebases -> skip audio in this compile (same as R1 v1/v2 practice).
# (Chains are FPV b-roll; the deliverable has been video-only in prior cuts.)

g_idx = 0
first = True
for ci, c in enumerate(ins):
    for i, frame in enumerate(c.decode(video=0)):
        if ci>0 and i==0:      # drop duplicated anchor frame at joins
            continue
        img = frame.to_image().convert("RGB")
        dr = ImageDraw.Draw(img, "RGBA")
        # caption bar
        dr.rectangle([0,768-46,1344,768], fill=(0,0,0,170))
        bbox = dr.textbbox((0,0), CAPTION, font=F_CAP)
        dr.text(((1344-(bbox[2]-bbox[0]))//2, 768-46+6), CAPTION, font=F_CAP, fill=(255,255,255,255))
        # minimap panel (right side)
        px = 1344 - framed.width - 14
        py = 14
        dr.rectangle([px-6,py-6,px+framed.width+6,py+framed.height+6], fill=(0,0,0,110))
        img.paste(framed, (px,py))
        # trail + moving dot (in framed-thumb coords)
        pt = pos_at(g_idx)
        tx = px+2 + pt[0]*s; ty = py+2 + pt[1]*s
        tr = trail_upto(pt)
        if len(tr)>=2:
            dr.line([(px+2+x*s, py+2+y*s) for x,y in tr], fill=(255,210,40,255), width=3, joint="curve")
        # pulse ring
        pr = 6 + 3*((g_idx % 24)/24.0)
        dr.ellipse([tx-pr-3,ty-pr-3,tx+pr+3,ty+pr+3], outline=(255,255,255,220), width=2)
        dr.ellipse([tx-5,ty-5,tx+5,ty+5], fill=(255,210,40,255), outline=(0,0,0,255), width=1)
        f2 = av.VideoFrame.from_image(img)
        f2 = f2.reformat(format="yuv420p")
        for p in vs.encode(f2): oc.mux(p)
        g_idx += 1

for p in vs.encode(): oc.mux(p)
oc.close()
for c in ins: c.close()
print("wrote", RESULT, g_idx, "frames")
