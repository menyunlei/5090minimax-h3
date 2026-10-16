"""v4b/v5 chain extender: I2V hop whose anchor frame carries a route-map HUD inset top-left.
Frame-exact continuation (I2V first_frame) + map visible/conditioning inside every hop.
Usage: python chain_extend_hud.py <prev_video.mp4> <anchor.png> <out_prefix> <spec.txt> [length]"""
import av
import json
import sys
from PIL import Image
from pathlib import Path

OUT_DIR = Path(r"C:\Users\yunlei\AppData\Local\Comfy-Desktop\ComfyUI-Shared\output")
IN_DIR = Path(r"C:\Users\yunlei\AppData\Local\Comfy-Desktop\ComfyUI-Shared\input")
BASE = Path(r"C:\Users\yunlei\.zcode\workspace\default\h3_map_baseline")
JOBS = BASE / "jobs_v4"
JOBS.mkdir(exist_ok=True)
ROUTE_MAP = BASE / "yym_route_v4b.png"
HUD_W = 300

POV = ("FIRST-PERSON BICYCLE CAMERA, non-negotiable: handlebar-mounted camera at exactly 1.6 "
       "meters, looking straight ahead horizontally; the handlebar grips and both rider's "
       "hands are ALWAYS VISIBLE at the bottom edge of every frame; ground fills the lower "
       "half; horizon near the middle; camera NEVER rises; NO aerial/drone/bird's-eye/top-"
       "down view.")

HUD = ("A small navigation map HUD is pinned at the TOP-LEFT corner of the frame, part of "
       "the bike computer display: keep it visible, sharp and UNCHANGED for the whole clip - "
       "it is a screen element, NOT a painted object in the world. The HUD shows this ride's "
       "route (thin dark line, north up): 'start' at the west end of the lake causeway, east "
       "along the causeway to the junction plaza ('turn'), then a RIGHT turn south along the "
       "lakeside path to the lakeside terrace ('end'). Blue = lotus lake, grey = stone "
       "paths, green = trees, black = red-walled imperial halls. Ride the route it shows.")

def composite_hud(frame_arr, out_png):
    img = Image.fromarray(frame_arr).convert("RGBA")
    m = Image.open(ROUTE_MAP).convert("RGBA")
    m.thumbnail((HUD_W, HUD_W), Image.LANCZOS)
    tile = Image.new("RGBA", (m.width + 14, m.height + 14), (15, 15, 18, 220))
    tile.alpha_composite(m, (7, 7))
    img.alpha_composite(tile, (24, 24))
    img.convert("RGB").save(IN_DIR / out_png)

def last_frame(p):
    with av.open(str(p)) as c:
        fr = None
        for f in c.decode(c.streams.video[0]):
            fr = f.to_ndarray(format="rgb24")
        return fr

def main():
    prev, anchor, prefix, spec_file = sys.argv[1:5]
    length = int(sys.argv[5]) if len(sys.argv) > 5 else 241
    spec = Path(spec_file).read_text(encoding="utf-8").strip()
    composite_hud(last_frame(OUT_DIR / prev), anchor)
    print("hud anchor saved:", anchor)
    prompt = ("The ride continues from the FIRST FRAME exactly as shown - same place, same "
              "heading, same handlebar, same map HUD - keep moving forward from it without "
              "any hesitation.\n\n" + HUD + "\n\n" + POV + "\n\n" + spec)
    g = {"prompt": {
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
        "16": {"class_type": "LoadImage", "inputs": {"image": anchor}},
        "17": {"class_type": "MiniMaxH3ImageToVideo", "inputs": {
            "clip": ["3", 0], "vae": ["4", 0], "prompt": prompt,
            "width": 1344, "height": 768, "length": length, "first_frame": ["16", 0]}},
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
    }, "client_id": "zcode-h3-v4b"}
    out = JOBS / f"{prefix}.json"
    with open(out, "w") as f:
        json.dump(g, f, indent=1)
    print("built", out)

if __name__ == "__main__":
    main()
