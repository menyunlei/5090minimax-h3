"""v4 chain extender: Ref2V hop with TWO references = annotated route map + previous last frame.
Usage: python chain_extend_map.py <prev_video.mp4> <anchor.png> <out_prefix> <spec.txt> [length]
Writes jobs_v4/<out_prefix>.json with refs ref_image_1=route map, ref_image_2=anchor."""
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
ROUTE_MAP = "yym_route_v4b.png"

POV = ("FIRST-PERSON BICYCLE CAMERA, non-negotiable: handlebar-mounted camera at exactly 1.6 "
       "meters above the ground, looking straight ahead horizontally; the handlebar grips and "
       "both of the rider's hands holding them are ALWAYS VISIBLE at the bottom edge of the "
       "frame, in every frame; the ground fills the lower half; the horizon stays near the "
       "middle; the camera NEVER rises; absolutely NO aerial view, NO drone shot, NO "
       "bird's-eye, NO top-down view.")

MAP = ("<Picture 1> is the AUTHORITATIVE NAVIGATION MAP of this ride, north up: a top-down "
       "map of a Yuanmingyuan-style imperial garden park with the ride route drawn as a thin "
       "dark LINE from the west end of the lake causeway ('start') east along the causeway "
       "to the junction plaza ('turn'), then RIGHT turn SOUTH along the lakeside path to a "
       "lakeside terrace ('end'). Treat it as a GEOMETRIC SPECIFICATION ONLY: blue = lotus "
       "lake, grey = stone paths, green dots = trees, black blocks = red-walled imperial "
       "halls. The map is a flat planning legend only; it must NEVER appear inside the video "
       "world in any form - no red lines or stripes anywhere, no colored dots, no floating "
       "or painted markers, no text, no rings; real paths are plain grey stone.")

CONT = ("<Picture 2> is your EXACT current first-person view right now, the precise frame the "
        "ride continues from. The clip MUST open on exactly this view - same place, same "
        "heading, same handlebar and hands - and keep moving forward from it without any "
        "hesitation or re-orientation.")

def prev_last_frame(p):
    with av.open(str(p)) as c:
        fr = None
        for f in c.decode(c.streams.video[0]):
            fr = f.to_ndarray(format="rgb24")
        return fr

def main():
    prev, anchor, prefix, spec_file = sys.argv[1:5]
    length = int(sys.argv[5]) if len(sys.argv) > 5 else 241
    spec = Path(spec_file).read_text(encoding="utf-8").strip()
    Image.fromarray(prev_last_frame(OUT_DIR / prev)).save(IN_DIR / anchor)
    print("anchor saved:", anchor)
    prompt = MAP + "\n\n" + CONT + "\n\n" + POV + "\n\n" + spec
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
        "16": {"class_type": "LoadImage", "inputs": {"image": ROUTE_MAP}},
        "18": {"class_type": "LoadImage", "inputs": {"image": anchor}},
        "17": {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {
            "clip": ["3", 0], "vae": ["4", 0], "prompt": prompt,
            "width": 1344, "height": 768, "length": length,
            "ref_image_size": "match",
            "ref_images.ref_image_1": ["16", 0],
            "ref_images.ref_image_2": ["18", 0]}},
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
    }, "client_id": "zcode-h3-v4"}
    out = JOBS / f"{prefix}.json"
    with open(out, "w") as f:
        json.dump(g, f, indent=1)
    print("built", out)

if __name__ == "__main__":
    main()
