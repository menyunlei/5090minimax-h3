"""v4.1 chain extender: Ref2V hop with refs = route map (image) + TAIL VIDEO of previous hop.
The video reference carries position/heading/motion right up to the cut -> tighter joins
with the map still conditioning every hop.
Usage: python chain_extend_mapvid.py <prev_video.mp4> <out_prefix> <spec.txt> [length] [tail_frames]"""
import av
import json
import sys
import numpy as np
from pathlib import Path

OUT_DIR = Path(r"C:\Users\yunlei\AppData\Local\Comfy-Desktop\ComfyUI-Shared\output")
IN_DIR = Path(r"C:\Users\yunlei\AppData\Local\Comfy-Desktop\ComfyUI-Shared\input")
BASE = Path(r"C:\Users\yunlei\.zcode\workspace\default\h3_map_baseline")
JOBS = BASE / "jobs_v4"
JOBS.mkdir(exist_ok=True)
ROUTE_MAP = "yym_route_v4b.png"

POV = ("FIRST-PERSON BICYCLE CAMERA, non-negotiable: handlebar-mounted camera at exactly 1.6 "
       "meters, looking straight ahead horizontally; the handlebar grips and both rider's "
       "hands are ALWAYS VISIBLE at the bottom edge of every frame; ground fills the lower "
       "half; horizon near the middle; camera NEVER rises; NO aerial/drone/bird's-eye/top-"
       "down view.")

MAP = ("<Picture 1> is the AUTHORITATIVE NAVIGATION MAP of this ride, north up: a top-down "
       "map of a Yuanmingyuan-style imperial garden park with the ride route drawn as a thin "
       "dark LINE from the west end of the lake causeway ('start') east along the causeway "
       "to the junction plaza ('turn'), then a RIGHT turn SOUTH along the lakeside path to "
       "a lakeside terrace ('end'). Blue = lotus lake, grey = stone paths, green dots = "
       "trees, black blocks = red-walled imperial halls. The map is a flat planning legend "
       "only; it must NEVER appear inside the video world - no red lines, dots, markers, "
       "text or rings painted anywhere; real paths are plain grey stone.")

CONT = ("The REFERENCE VIDEO is the last few seconds of the ride so far: your exact current "
        "position, heading, speed and view. This clip MUST continue seamlessly from the END "
        "of that reference video - the identical place, heading and handlebar view, moving "
        "forward at the same speed without any hesitation or repositioning.")

def write_tail(prev_path, tail_path, n_tail):
    frames = []
    with av.open(str(prev_path)) as c:
        for f in c.decode(c.streams.video[0]):
            frames.append(f.to_ndarray(format="rgb24"))
    tail = frames[-n_tail:]
    out = av.open(str(tail_path), mode="w")
    st = out.add_stream("h264", rate=24)
    st.width, st.height, st.pix_fmt = 1344, 768, "yuv420p"
    for a in tail:
        for pkt in st.encode(av.VideoFrame.from_ndarray(a, format="rgb24")):
            out.mux(pkt)
    for pkt in st.encode():
        out.mux(pkt)
    out.close()
    return len(tail)

def main():
    prev, prefix, spec_file = sys.argv[1:4]
    length = int(sys.argv[4]) if len(sys.argv) > 4 else 241
    n_tail = int(sys.argv[5]) if len(sys.argv) > 5 else 72
    spec = Path(spec_file).read_text(encoding="utf-8").strip()
    tail_name = f"{prefix}_reftail.mp4"
    n = write_tail(OUT_DIR / prev, IN_DIR / tail_name, n_tail)
    print(f"tail saved: {tail_name} ({n} frames)")
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
        "20": {"class_type": "LoadVideo", "inputs": {"file": tail_name}},
        "21": {"class_type": "GetVideoComponents", "inputs": {"video": ["20", 0]}},
        "17": {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {
            "clip": ["3", 0], "vae": ["4", 0], "prompt": prompt,
            "width": 1344, "height": 768, "length": length,
            "ref_image_size": "match",
            "ref_images.ref_image_1": ["16", 0],
            "ref_videos.ref_video_1": ["21", 0]}},
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
    }, "client_id": "zcode-h3-v41"}
    out = JOBS / f"{prefix}.json"
    with open(out, "w") as f:
        json.dump(g, f, indent=1)
    print("built", out)

if __name__ == "__main__":
    main()
