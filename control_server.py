"""Interactive map-steering control server v3 (GPU1 lane, ComfyUI :8189).
Adds self-play + self-evaluation on top of v2's WASD/queue design:
  /drive    manual WASD real-time riding (same as v2).
  /mission  self-play cockpit: load a mission (waypoints) -> the page's autopilot rides the
            route live using the SAME kinematics/commit path a human uses; each committed
            ~90px leg queues a Ref2V hop on GPU1; every landed clip is auto-scored
            (join continuity, map-graphic red-leak ratio, motion energy) and shown in the
            mission scoreboard. GET /mission.json = machine-readable mission status.
  /         combined page (drive UI).
Truth = sim_pos (rider); video head = gen_pos lags minutes behind on a single 5090 lane."""
import av
import json
import math
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from PIL import Image

BASE = Path(r"C:\Users\yunlei\.zcode\workspace\default\h3_map_baseline")
STATE_DIR = BASE / "interactive"
STATE_DIR.mkdir(exist_ok=True)
STATE_FILE = STATE_DIR / "state.json"
OUT_DIR = Path(r"C:\Users\yunlei\AppData\Local\Comfy-Desktop\ComfyUI-Shared\output")
IN_DIR = Path(r"C:\Users\yunlei\AppData\Local\Comfy-Desktop\ComfyUI-Shared\input")
COMFY = "http://127.0.0.1:8189"
ROUTE_MAP_REF = "yym_route_v4b.png"      # conditioning map (clean)
USER_MAP = BASE / "yym_route_v4.png"     # labelled map shown in the page
PATH_SRC = BASE / "yym_orig_nl_px.png"   # raw pixel map for path snapping (un-annotated)
MAX_QUEUE = 8

POV = ("FIRST-PERSON BICYCLE CAMERA, non-negotiable: handlebar-mounted camera at exactly 1.6 "
       "meters, looking straight ahead horizontally; handlebar grips and both rider's hands "
       "ALWAYS VISIBLE at the bottom edge of every frame; ground fills the lower half; "
       "horizon near the middle; camera NEVER rises; NO aerial/drone/bird's-eye view.")
MAP = ("<Picture 1> is the AUTHORITATIVE NAVIGATION MAP, north up: top-down map of a "
       "Yuanmingyuan-style imperial garden park; blue = lotus lake, grey = stone paths, "
       "green dots = trees, black blocks = red-walled imperial halls. The map is a flat "
       "planning legend only; it must NEVER appear inside the video world in any form - "
       "no red lines, dots, markers, text or rings painted anywhere; real paths are plain "
       "grey stone.")
CONT = ("<Picture 2> is your EXACT current first-person view: the clip MUST open on this "
        "identical view and move forward from it without hesitation or repositioning.")

def init_state():
    if STATE_FILE.exists():
        st = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        p, h = st.get("pos", [676, 894]), st.get("heading", 180.0)
        st.setdefault("sim_pos", list(p)); st.setdefault("sim_heading", h)
        st.setdefault("gen_pos", list(p)); st.setdefault("gen_heading", h)
        st.setdefault("queue", [])
        st.setdefault("busy", False)
        st.setdefault("evals", [])
        st.setdefault("mission", None)
        return st
    st = {"pos": [676, 894], "heading": 180.0,
          "sim_pos": [676, 894], "sim_heading": 180.0,
          "gen_pos": [676, 894], "gen_heading": 180.0,
          "clips": ["h3v4_h5_00001_.mp4"], "gen_no": 0, "busy": False,
          "queue": [], "evals": [], "mission": None,
          "last_msg": "ready: at W2 terrace, facing the lake (west)"}
    STATE_FILE.write_text(json.dumps(st, indent=1))
    return st

ST = init_state()
LOCK = threading.Lock()
KICK = threading.Event()

def save():
    STATE_FILE.write_text(json.dumps(ST, indent=1))

def path_mask():
    if not hasattr(path_mask, "m"):
        p = PATH_SRC
        if not p.exists():
            p = Path(IN_DIR / "yym_route_v4b.png")
        a = np.asarray(Image.open(p).convert("RGB")).astype(np.int16)
        grey = (np.abs(a[..., 0] - a[..., 1]) < 28) & (np.abs(a[..., 1] - a[..., 2]) < 28) \
               & (a[..., 0] > 40) & (a[..., 0] < 130)
        ys, xs = np.nonzero(grey)
        path_mask.m = np.stack([xs, ys], axis=1).astype(np.float32)
    return path_mask.m

def snap(x, y):
    pts = path_mask()
    d = np.hypot(pts[:, 0] - x, pts[:, 1] - y)
    i = int(np.argmin(d))
    return (int(pts[i, 0]), int(pts[i, 1])), float(d[i])

def snap_traj(samples):
    out, worst = [], 0.0
    for x, y in samples:
        (sx, sy), d = snap(float(x), float(y))
        out.append([sx, sy]); worst = max(worst, d)
    ded = [out[0]] if out else []
    for p in out[1:]:
        if p != ded[-1]:
            ded.append(p)
    return ded, worst

def zone_text(x, y):
    if 380 <= y <= 460 and x < 700:
        return ("the narrow stone causeway crossing the lotus lake: white marble balustrades "
                "glide past on BOTH sides at hand height, lotus-covered lake water shines "
                "on both sides of the path")
    if 640 <= x <= 730 and y > 460:
        return ("the lakeside gallery path: the open lotus lake is on the WEST side at eye "
                "level, willow trees shade the east side, carved balustrades appear wherever "
                "the path hugs the water")
    if 640 <= x <= 720 and y < 380:
        return ("the tree-lined avenue in the north part of the park: old trees close in on "
                "both sides of the gravel path, red-walled halls stand nearby")
    return ("a stone park path in the imperial garden: old trees, stone paving, red-walled "
            "halls and lake views appearing as the path winds")

def ang_diff(a, b):
    return ((a - b + 180) % 360) - 180

def traj_phases(samples, start_heading):
    """Classify a ridden polyline into maneuvers. Design: a leg usually carries ONE
    maneuver; whole-leg net heading decides the arc class; a clear direction REVERSAL
    (S-curve) recurses at the extremum. Position smoothing kills snap zigzag."""
    if len(samples) < 2:
        return None
    pts = [tuple(p) for p in samples]
    sm = []
    for i in range(len(pts)):
        lo, hi = max(0, i - 2), min(len(pts), i + 3)
        sm.append((sum(p[0] for p in pts[lo:hi]) / (hi - lo),
                   sum(p[1] for p in pts[lo:hi]) / (hi - lo)))
    dists, heads = [], []
    for i in range(1, len(sm)):
        dx, dy = sm[i][0] - sm[i-1][0], sm[i][1] - sm[i-1][1]
        dists.append(math.hypot(dx, dy))
        heads.append(math.degrees(math.atan2(dy, dx)) % 360)
    total = sum(dists)
    if total < 12:
        return None
    SH = []
    for i in range(len(heads)):
        lo, hi = max(0, i - 2), min(len(heads), i + 3)
        vs = [math.radians(h) for h in heads[lo:hi]]
        SH.append(math.degrees(math.atan2(sum(math.sin(v) for v in vs),
                                          sum(math.cos(v) for v in vs))) % 360)

    def mean_head(hs):
        vs = [math.radians(h) for h in hs]
        return math.degrees(math.atan2(sum(math.sin(v) for v in vs),
                                       sum(math.cos(v) for v in vs))) % 360

    def classify(i0, i1, h0):
        seg = heads[i0:i1]
        dl = dists[i0:i1]
        if not seg:
            return []
        k = min(3, len(seg))
        h_start = mean_head(seg[:k]); h_end = mean_head(seg[-k:])
        net = ang_diff(h_end, h_start)
        devs = [ang_diff(h, h_start) for h in seg]
        lo_d, hi_d = min(devs), max(devs)
        # clear direction reversal relative to the leg's own start -> split at extremum
        if hi_d - lo_d > 40 and hi_d > 20 and lo_d < -20:
            kk = devs.index(hi_d if hi_d > -lo_d else lo_d)
            if 0 < kk < len(seg) - 1:
                L = sum(dl[:kk]); R = sum(dl[kk:])
                if L >= 15 and R >= 15:
                    left = classify(i0, i0 + kk, h0)
                    right = classify(i0 + kk, i1, (h0 + devs[kk]) % 360)
                    if left and right:
                        return left + right
        return [{"i0": i0, "i1": i1, "len": max(sum(dl), 1.0), "turn": net}]

    phases = classify(0, len(heads), start_heading)
    if not phases:
        return None
    return phases, total

def phase_text(ph, heading_before):
    side = None
    if ph["turn"] > 6: side = "LEFT"
    elif ph["turn"] < -6: side = "RIGHT"
    h_end = (heading_before + ph["turn"]) % 360
    if side is None:
        return "ride STRAIGHT ahead along the path, steady cruise speed", h_end
    m = abs(ph["turn"])
    if m < 60:
        fg = "a smooth gentle arc"
    elif m < 135:
        fg = "a smooth continuous arc WHILE MOVING, leaning into the bend like a real cyclist"
    else:
        fg = "ONE smooth continuous U-turn WHILE MOVING, the view sweeping around"
    return (f"WITHOUT slowing down, bend the path into {fg} of about {int(m)} degrees to "
            f"the {side}"),h_end

def spec_from_traj(samples, start_heading):
    got = traj_phases(samples, start_heading)
    if got is None:
        return None
    phases, total = got
    mx, my = samples[len(samples) // 2]
    ex, ey = samples[-1]
    lines = ["ONE CONTINUOUS RIDE, no stopping anywhere (total 10.0 seconds, 24 fps)."]
    t, h = 0.0, start_heading
    for ph in phases:
        ph["t"] = 10.0 * ph["len"] / total
        txt, h = phase_text(ph, h)
        lines.append(f"- {t - ph['t']:.1f}s-{t:.1f}s: {txt}.")
        t += ph["t"]
    lines.append(f"Stay on the path you are already riding; ahead it passes "
                 f"{zone_text(mx, my)}, then continues to "
                 f"{zone_text(ex, ey)}.")
    lines.append("Photorealistic, summer daylight, steady cycling speed, natural head "
                 "stabilization. The clip ends CUT while still moving at cruise speed - "
                 "do NOT slow down, do NOT stop, no cuts inside the clip.")
    return ("\n".join(lines), (ex, ey),
            (start_heading + sum(p["turn"] for p in phases)) % 360,
            [{"len_px": round(p["len"], 1), "turn_deg": round(p["turn"], 1)} for p in phases])

def first_frame_of(mp4_path):
    fr = None
    with av.open(str(mp4_path)) as c:
        for f in c.decode(c.streams.video[0]):
            fr = f.to_ndarray(format="rgb24"); break
    return fr

def last_frame_of(mp4_path):
    fr = None
    with av.open(str(mp4_path)) as c:
        for f in c.decode(c.streams.video[0]):
            fr = f.to_ndarray(format="rgb24")
    return fr

def evaluate_clip(prev_path, new_path):
    """Automatic QC scores for a landed hop."""
    prev_last = last_frame_of(prev_path)
    new_first = first_frame_of(new_path)
    join = float(np.abs(prev_last.astype(np.int16) - new_first.astype(np.int16)).mean())
    reds, diffs, prev_f = [], [], None
    n = 0
    with av.open(str(new_path)) as c:
        for f in c.decode(c.streams.video[0]):
            n += 1
            if n % 24 != 1:
                continue
            a = f.to_ndarray(format="rgb24")
            r = a[..., 0].astype(np.int16); g = a[..., 1].astype(np.int16); b = a[..., 2].astype(np.int16)
            reds.append(float((((r > 200) & (g < 70) & (b < 70)).mean())))
            if prev_f is not None:
                diffs.append(float(np.abs(a.astype(np.int16) - prev_f.astype(np.int16)).mean()))
            prev_f = a
    return {"join_diff": round(join, 2),
            "red_leak_pct": round(100 * (sum(reds) / max(len(reds), 1)), 4),
            "motion_mean": round(sum(diffs) / max(len(diffs), 1), 2)}

def build_job(prefix, anchor_png, prompt, length=241):
    return {"prompt": {
        "1": {"class_type": "UNETLoader", "inputs": {
            "unet_name": "minimax_h3_fl2va_pruned_int8_convrot.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": ["1", 0],
            "lora_name": "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
            "strength_model": 1.0}},
        "3": {"class_type": "CLIPLoader", "inputs": {
            "clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "type": "minimax"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_video_vae_int8_convrot.safetensors"}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"}},
        "16": {"class_type": "LoadImage", "inputs": {"image": ROUTE_MAP_REF}},
        "18": {"class_type": "LoadImage", "inputs": {"image": anchor_png}},
        "17": {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {
            "clip": ["3", 0], "vae": ["4", 0], "prompt": prompt,
            "width": 1344, "height": 768, "length": length, "ref_image_size": "match",
            "ref_images.ref_image_1": ["16", 0], "ref_images.ref_image_2": ["18", 0]}},
        "7": {"class_type": "RandomNoise", "inputs": {"noise_seed": 42}},
        "8": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "9": {"class_type": "BasicScheduler", "inputs": {
            "model": ["2", 0], "scheduler": "simple", "steps": 8, "denoise": 1.0}},
        "10": {"class_type": "BasicGuider", "inputs": {"model": ["2", 0], "conditioning": ["17", 0]}},
        "11": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["7", 0], "guider": ["10", 0], "sampler": ["8", 0],
            "sigmas": ["9", 0], "latent_image": ["17", 1]}},
        "12": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["4", 0]}},
        "13": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["11", 0], "vae": ["5", 0]}},
        "14": {"class_type": "CreateVideo", "inputs": {
            "images": ["12", 0], "fps": 24, "audio": ["13", 0], "codec": "auto"}},
        "15": {"class_type": "SaveVideo", "inputs": {
            "video": ["14", 0], "filename_prefix": prefix, "format": "auto"}},
    }, "client_id": "zcode-h3-steer"}

def post_prompt(wf):
    req = urllib.request.Request(COMFY + "/prompt", data=json.dumps(wf).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())

def generate_one(samples):
    """Render one queued trajectory leg; returns True on success. Advances gen head + evals."""
    with LOCK:
        gen_no = ST["gen_no"] + 1
        start_heading = ST["gen_heading"]
        prev_clip = OUT_DIR / ST["clips"][-1]
        ST["busy"] = True
        ST["gen_no"] = gen_no
    prefix = f"h3ctrl_{gen_no:03d}"
    out_name = f"{prefix}_00001_.mp4"
    try:
        built = spec_from_traj(samples, start_heading)
        if built is None or len(samples) < 2 or \
                math.hypot(samples[-1][0] - samples[0][0], samples[-1][1] - samples[0][1]) < 10:
            raise RuntimeError("trajectory too short")
        spec, (ex, ey), end_heading, phases_meta = built
        anchor = f"steer_anchor_{gen_no:03d}.png"
        Image.fromarray(last_frame_of(prev_clip)).save(IN_DIR / anchor)
        with LOCK:
            ST["last_msg"] = f"generating {prefix}: {phases_meta}"
            save()
        prompt = MAP + "\n\n" + CONT + "\n\n" + POV + "\n\n" + spec
        post_prompt(build_job(prefix, anchor, prompt))
        deadline = time.time() + 2400
        p = OUT_DIR / out_name
        while time.time() < deadline:
            if p.exists() and p.stat().st_size > 100000:
                time.sleep(3)
                break
            time.sleep(8)
        if not p.exists():
            raise RuntimeError("timeout waiting for output")
        ev = evaluate_clip(prev_clip, p)
        with LOCK:
            ST["clips"].append(out_name)
            ST["gen_pos"] = [ex, ey]
            ST["gen_heading"] = end_heading
            ST["pos"] = [ex, ey]
            ST["heading"] = end_heading
            ST["busy"] = False
            ST["last_msg"] = f"ready: {prefix} done, video head at ({ex},{ey})"
            ST["evals"].append({"hop": prefix, "to": [ex, ey], "phases": phases_meta,
                                "ts": time.strftime("%H:%M:%S"), **ev})
            save()
        return True
    except Exception as e:
        with LOCK:
            ST["busy"] = False
            ST["last_msg"] = f"error: {e}"
            save()
        return False

def worker_loop():
    while True:
        KICK.wait()
        KICK.clear()
        while True:
            with LOCK:
                item = ST["queue"].pop(0) if ST["queue"] else None
                if item is not None:
                    save()
            if item is None:
                break
            ok = generate_one(item["samples"])
            if not ok and not item.get("retried"):
                item["retried"] = True
                with LOCK:
                    ST["queue"].insert(0, item)
                    save()
                time.sleep(20)
                KICK.set()
                break

CSS = """body{background:#121418;color:#eee;font-family:Segoe UI,Arial;margin:16px}
h2{font-size:17px;margin:6px 0}
#wrap{display:flex;gap:18px}
#mapc{position:relative}
canvas.cv{border:2px solid #444;width:560px;display:block}
video{width:640px;background:#000}
#st{margin:6px 0;font-size:14px;color:#9fe08a}
#stx{font-size:13px;color:#bbb;margin-top:6px}
.bad{color:#f08080}
kbd{background:#2a2e36;border:1px solid #555;border-radius:4px;padding:1px 6px;font-size:12px}
small{color:#999}
.marker{position:absolute;pointer-events:none;transform:translate(-50%,-50%)}
table{border-collapse:collapse;font-size:13px;margin-top:8px}
td,th{border:1px solid #3a3f49;padding:3px 8px;text-align:left}
th{color:#9fc3ff}
td.ok{color:#9fe08a}td.warn{color:#f0c060}td.poor{color:#f08080}
"""

JS_CORE = """
const cv=document.getElementById('cv'),ctx=cv.getContext('2d');
const img=new Image();img.src='/map.png';
img.onload=()=>draw();
const v=document.getElementById('v'),st=document.getElementById('st'),stx=document.getElementById('stx');
const me=document.getElementById('me'),gh=document.getElementById('gh');
let px=680,py=700,hd=270,spd=0,traj=[],lastX=null,lastY=null,lastBase=null;
let clips=[];
const keys={autopilot:false};let committedDist=0,autoMission=null,wpIdx=0,done=false,lastFail=0;
addEventListener('keydown',e=>{keys[e.key.toLowerCase()]=true;if(e.key===' '){commit();e.preventDefault();}});
addEventListener('keyup',e=>keys[e.key.toLowerCase()]=false);
let lastT=performance.now();
function tick(){
 const now=performance.now();let dt=(now-lastT)/1000;lastT=now;if(dt>0.6)dt=0.6;const F=dt*60;
 if(keys.autopilot&&autoMission&&!done){
   const wp=autoMission.waypoints[wpIdx];
   if(wp){const want=(Math.atan2(wp[1]-py,wp[0]-px)*180/Math.PI+360)%360;
    let rel=((want-hd+540)%360)-180;
    hd=(hd+Math.max(-2.2*F,Math.min(2.2*F,rel))+360)%360;
    const d=Math.hypot(wp[0]-px,wp[1]-py);
    spd=Math.min(spd+0.06*F,1.1);
    if(d<10){wpIdx++;if(wpIdx>=autoMission.waypoints.length){done=true;spd=0;
      stx.textContent+='  【任务路径已骑完】';}}
   }
 }else{
  if(keys['w'])spd=Math.min(spd+0.11*F,3.0);
  else if(keys['s'])spd=Math.max(spd-0.25*F,0);
  else spd=Math.max(spd-0.035*F,0);
  const stt=(keys['a']?-1:0)+(keys['d']?1:0);
  if(stt)hd=(hd+stt*1.9*Math.min(spd,2.2)*F+360)%360;}
 if(spd>0.02){const r=hd*Math.PI/180;px+=Math.cos(r)*spd*F;py+=Math.sin(r)*spd*F;
  px=Math.max(4,Math.min(1020,px));py=Math.max(4,Math.min(1020,py));
  if(lastX===null||Math.hypot(px-lastX,py-lastY)>6){traj.push([Math.round(px),Math.round(py)]);
   committedDist+=Math.hypot(px-(lastX??px),py-(lastY??py));lastX=px;lastY=py;
   if(traj.length>300)traj.shift();}
  if(committedDist>200&&(Date.now()-lastFail>3000))commit();}
 draw();}
function draw(){ctx.drawImage(img,0,0,1024,1024);
 if(autoMission){ctx.strokeStyle='#4da3ff';ctx.lineWidth=2;ctx.setLineDash([8,6]);
  ctx.beginPath();let sx=px,sy=py;ctx.moveTo(sx,sy);
  for(let i=wpIdx;i<autoMission.waypoints.length;i++){ctx.lineTo(autoMission.waypoints[i][0],autoMission.waypoints[i][1]);}
  ctx.stroke();ctx.setLineDash([]);
  autoMission.waypoints.forEach((w,i)=>{ctx.fillStyle=i<wpIdx?'#35c46f':(i===wpIdx?'#ffd34d':'#7d8cff');
   ctx.beginPath();ctx.arc(w[0],w[1],7,0,7);ctx.fill();
   ctx.fillStyle='#000';ctx.font='11px sans';ctx.fillText(String(i+1),w[0]-3,w[1]+4);});}
 if(traj.length>1){ctx.strokeStyle='#ffd34d';ctx.lineWidth=3;ctx.beginPath();
  ctx.moveTo(traj[0][0],traj[0][1]);for(const p of traj)ctx.lineTo(p[0],p[1]);ctx.stroke();}
 const k=560/1024;
 me.style.left=(px*k)+'px';me.style.top=(py*k)+'px';
 me.style.transform=`translate(-50%,-50%) rotate(${hd+90}deg)`;}
async function poll(){try{const r=await fetch('/state');const s=await r.json();
 st.textContent=s.last_msg;st.className=/error/.test(s.last_msg)?'bad':'';
 let extra=`队列 ${s.queue_len} 段 · 影像头 (${s.gen_pos[0]},${s.gen_pos[1]}) ${Math.round(s.gen_heading)}°`;
 if(keys.autopilot&&!done)stx.textContent=extra+(autoMission?` · 自动巡航 目标点 ${wpIdx+1}/${autoMission.waypoints.length}`:' · 等待任务');
 else stx.textContent=extra;
 const k=560/1024;gh.style.left=(s.gen_pos[0]*k)+'px';gh.style.top=(s.gen_pos[1]*k)+'px';
 gh.style.transform=`translate(-50%,-50%) rotate(${s.gen_heading+90}deg)`;
 if(s.clips.join('|')!==clips.join('|')){clips=s.clips;
  if(!v.src||v.ended){v.src='/clip/'+clips[clips.length-1];v.play().catch(()=>{});}}
 if(window.onState)onState(s);
 }catch(e){}setTimeout(poll,3000);}
v.addEventListener('ended',()=>{const i=clips.indexOf(v.src.split('/').pop());
 if(i>=0&&i<clips.length-1){v.src='/clip/'+clips[i+1];v.play().catch(()=>{});}});
async function commit(){
 if(traj.length<2)return;
 const leg=[...(lastBase?[lastBase]:[]),...traj];
 traj=[];committedDist=0;
 const resp=await fetch('/steer_traj',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({traj:leg.slice(-320)})});
 const t=await resp.text();
 if(!resp.ok){traj=leg;committedDist=120;lastFail=Date.now();
   if(resp.status===429&&keys.autopilot)spd=0;
 }else{lastBase=leg[leg.length-1];}
}
cv.addEventListener('click',async e=>{
 const r=cv.getBoundingClientRect();
 const x=Math.round((e.clientX-r.left)*1024/r.width),y=Math.round((e.clientY-r.top)*1024/r.height);
 const resp=await fetch('/steer',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({x,y})});
 const t=await resp.text();st.textContent=resp.ok?('已入队: '+t):('未接受: '+t);
 st.className=resp.ok?'':'bad';});
"""

PAGE_DRIVE = """<!DOCTYPE html><html><head><meta charset="utf-8"><title>H3 WASD ride</title>
<style>""" + CSS + """</style></head><body>
<h2>H3 手动实时骑行 · WASD 开地图，影像随后追上来（GPU1 渲染车道）</h2>
<div id="wrap">
<div id="mapc"><canvas class="cv" id="cv" width="1024" height="1024"></canvas>
<div class="marker" id="me"><svg width="26" height="26"><polygon points="13,0 24,24 13,18 2,24" fill="#f6c90e" stroke="#000"/></svg></div>
<div class="marker" id="gh"><svg width="18" height="18"><polygon points="9,0 17,17 9,13 1,17" fill="#7d8cff" stroke="#000"/></svg></div>
</div>
<div><div id="st">loading...</div><video id="v" controls autoplay muted></video>
<div id="stx"></div>
<div><small><kbd>W</kbd>加速 <kbd>S</kbd>刹车 <kbd>A</kbd>/<kbd>D</kbd>转向 <kbd>空格</kbd>手动提交轨迹段。
每90像素自动提交一段→排队渲染；黄色箭头=你（实时），蓝色=影像头；队列≤8段；也可以直接点地图提交目标点。
自我驾驶模式在 <a style="color:#7d8cff" href="/mission">/mission</a>。</small></div>
</div></div>
<script>""" + JS_CORE + """
async function boot(){try{const r=await fetch('/state');const s=await r.json();
 px=s.sim_pos[0];py=s.sim_pos[1];hd=s.sim_heading;clips=s.clips;}catch(e){}
 setInterval(tick,16);poll();}
boot();
</script></body></html>"""

PAGE_MISSION = """<!DOCTYPE html><html><head><meta charset="utf-8"><title>H3 mission autopilot</title>
<style>""" + CSS + """
#score{max-height:300px;overflow:auto}
.mis{font-size:13px;color:#ccd;background:#1a1e25;border:1px solid #333;border-radius:6px;padding:8px;margin:8px 0;white-space:pre-wrap}
</style></head><body>
<h2>H3 自我驾驶任务舱 · 我（AI）按任务自动骑这条路，影像追上来后自动评分</h2>
<div id="wrap">
<div id="mapc"><canvas class="cv" id="cv" width="1024" height="1024"></canvas>
<div class="marker" id="me"><svg width="26" height="26"><polygon points="13,0 24,24 13,18 2,24" fill="#f6c90e" stroke="#000"/></svg></div>
<div class="marker" id="gh"><svg width="18" height="18"><polygon points="9,0 17,17 9,13 1,17" fill="#7d8cff" stroke="#000"/></svg></div>
</div>
<div style="flex:1;min-width:520px">
<div class="mis" id="misbox">等待任务下发…</div>
<div id="st">loading...</div><video id="v" controls autoplay muted></video>
<div id="stx"></div>
<div id="score"></div>
<div><small>蓝虚线=任务路线（数字=途经点），黄线=我已骑的轨迹（每90px自动提交一段进渲染队列）。
黄箭头=我（实时），蓝箭头=影像头。评分标准：接景差&lt;运动基线12.78 为 PASS；红漏=地图图形漏进画面的像素比；
motion=段内平均帧差（防卡死）。&lt;: 转人工复核条帧图。</small></div>
</div></div>
<script>""" + JS_CORE + """
function fmtEval(e){const jd=e.join_diff<=12.78?'ok':(e.join_diff<=25?'warn':'poor');
 const rl=e.red_leak_pct<0.05?'ok':(e.red_leak_pct<0.3?'warn':'poor');
 return `<tr><td>${e.hop}</td><td>(${e.to})</td><td>${JSON.stringify(e.phases)}</td>`+
 `<td class="${jd}">join ${e.join_diff}</td><td class="${rl}">red ${e.red_leak_pct}%</td>`+
 `<td>motion ${e.motion_mean}</td><td>${e.ts}</td></tr>`;}
window.onState=function(s){
 fetch('/mission.json').then(r=>r.json()).then(m=>{
  if(m.mission){misbox.textContent='任务「'+m.mission.name+'」\\n'+m.mission.desc+'\\n途经点：'+JSON.stringify(m.mission.waypoints);
   if(!autoMission&&!m.done){autoMission=m.mission;}}
  document.getElementById('score').innerHTML='<table><tr><th>段</th><th>终点</th><th>相位</th><th>接景</th><th>红漏</th><th>运动</th><th>时间</th></tr>'+(m.evals||[]).map(fmtEval).join('')+'</table>';
 }).catch(()=>{});};
async function boot(){try{const r=await fetch('/state');const s=await r.json();
 px=s.sim_pos[0];py=s.sim_pos[1];hd=s.sim_heading;clips=s.clips;
 const m=await (await fetch('/mission.json')).json();
 if(m.mission&&m.mission.autopilot){keys.autopilot=true;autoMission=m.mission;}
 }catch(e){}
 setInterval(tick,16);poll();}
boot();
</script></body></html>"""

class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass
    def _send(self, code, body, ctype="text/plain; charset=utf-8"):
        b = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)
    def do_GET(self):
        u = urlparse(self.path).path
        if u in ("/", "/drive"):
            self._send(200, PAGE_DRIVE, "text/html; charset=utf-8")
        elif u == "/mission":
            self._send(200, PAGE_MISSION, "text/html; charset=utf-8")
        elif u == "/mission.json":
            with LOCK:
                self._send(200, json.dumps({"mission": ST["mission"], "evals": ST["evals"],
                                            "queue_len": len(ST["queue"]),
                                            "gen_pos": ST["gen_pos"], "busy": ST["busy"]}),
                           "application/json")
        elif u == "/map.png":
            self._send(200, Path(USER_MAP).read_bytes(), "image/png")
        elif u == "/state":
            with LOCK:
                d = {k: ST[k] for k in ("pos", "heading", "clips", "busy", "last_msg",
                                        "sim_pos", "sim_heading", "gen_pos", "gen_heading",
                                        "gen_no")}
                d["queue_len"] = len(ST["queue"])
            self._send(200, json.dumps(d), "application/json")
        elif u.startswith("/clip/"):
            name = u.split("/", 2)[2]
            p = OUT_DIR / name
            if p.exists() and ".." not in name:
                self._send(200, p.read_bytes(), "video/mp4")
            else:
                self._send(404, "no such clip")
        else:
            self._send(404, "nope")
    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            d = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            self._send(400, "bad json"); return
        if self.path == "/mission/load":
            wps = d.get("waypoints")
            if not isinstance(wps, list) or not wps:
                self._send(400, "need waypoints [[x,y],...]"); return
            with LOCK:
                ST["mission"] = {"name": d.get("name", "mission"),
                                 "desc": d.get("desc", ""),
                                 "waypoints": [[int(x), int(y)] for x, y in wps],
                                 "autopilot": bool(d.get("autopilot", True)),
                                 "loaded": time.strftime("%H:%M:%S")}
                save()
            self._send(200, "mission loaded")
        elif self.path == "/steer_traj":
            samples = d.get("traj")
            if not isinstance(samples, list) or len(samples) < 2:
                self._send(400, "need >=2 samples"); return
            snapped, worst = snap_traj(samples[-320:])
            if len(snapped) < 2:
                self._send(400, "trajectory collapsed after snapping"); return
            if worst > 90:
                self._send(400, f"off-path (sample {int(worst)}px from nearest path)"); return
            with LOCK:
                if len(ST["queue"]) >= MAX_QUEUE:
                    self._send(429, f"render queue full ({MAX_QUEUE} legs)"); return
                ST["queue"].append({"samples": snapped})
                ST["sim_pos"] = snapped[-1]
                ST["sim_heading"] = math.degrees(math.atan2(
                    snapped[-1][1] - snapped[-2][1], snapped[-1][0] - snapped[-2][0])) % 360
                qlen = len(ST["queue"])
                save()
            KICK.set()
            self._send(202, f"queued leg ({qlen} in queue), {len(snapped)} samples")
        elif self.path == "/steer":
            try:
                x, y = int(d["x"],), int(d["y"])
            except Exception:
                self._send(400, "bad coords"); return
            (sx, sy), dist = snap(x, y)
            if dist > 200:
                self._send(400, f"off-path (nearest path {int(dist)}px away)"); return
            with LOCK:
                bx, by = (ST["queue"][-1]["samples"][-1] if ST["queue"] else ST["gen_pos"])
            steps = max(2, int(math.hypot(sx - bx, sy - by) // 8))
            traj = [[int(bx + (sx - bx) * i / steps), int(by + (sy - by) * i / steps)]
                    for i in range(steps + 1)]
            snapped, _ = snap_traj(traj)
            with LOCK:
                if len(ST["queue"]) >= MAX_QUEUE:
                    self._send(429, f"render queue full ({MAX_QUEUE} legs)"); return
                ST["queue"].append({"samples": snapped})
                ST["sim_pos"] = [sx, sy]
                qlen = len(ST["queue"])
                save()
            KICK.set()
            self._send(202, f"queued straight leg to ({sx},{sy}), {qlen} in queue")
        else:
            self._send(404, "nope")

if __name__ == "__main__":
    print("state:", {k: ST[k] for k in ("sim_pos", "gen_pos", "gen_no")})
    # boot recovery: a restart kills the old worker but ComfyUI may still be
    # rendering its job; entries pop-and-generate sequentially from here.
    ST["busy"] = False
    if ST["queue"]:
        ST["last_msg"] = f"boot: draining render queue ({len(ST['queue'])} legs pending)"
    save()
    threading.Thread(target=worker_loop, daemon=True).start()
    KICK.set()
    print("serving on http://127.0.0.1:8787  ( /drive manual, /mission self-play, GPU1 -> :8189 )")
    ThreadingHTTPServer(("127.0.0.1", 8787), H).serve_forever()
