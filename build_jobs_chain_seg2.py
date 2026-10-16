# Chain step 2: extract last frame of each seg1 video, copy to ComfyUI input dir,
# and build seg2 I2V jobs with first_frame = that last frame.
# seg2 prompt hard-anchors a 90-degree LEFT turn in the first ~2.5 s.
import json, os
import av
from PIL import Image
from pathlib import Path

OUT_DIR = Path(r"C:\Users\yunlei\AppData\Local\Comfy-Desktop\ComfyUI-Shared\output")
IN_DIR = Path(r"C:\Users\yunlei\AppData\Local\Comfy-Desktop\ComfyUI-Shared\input")
JOBS = Path(r"C:\Users\yunlei\.zcode\workspace\default\h3_map_baseline\jobs_chain")

SEEDS = [42, 1337, 20261006]

PROMPT_SEG2 = (
    "A first-person bicycle camera starts exactly at the corner of the intersection shown in "
    "the first frame, facing straight into the crossing toward the crosswalk.\n\n"
    "MANDATORY TIMELINE (total 5.2 seconds):\n"
    "- 0.0s-0.5s: begin moving and immediately start turning the handlebars LEFT.\n"
    "- 0.5s-2.5s: rotate the view smoothly and continuously LEFT by a full 90 degrees. "
    "The intersection ahead sweeps toward the right edge of the frame; the cross street on the "
    "left pivots into the center of the view.\n"
    "- 2.5s-5.2s: the turn is COMPLETE — the camera now faces straight down the cross street "
    "that was on the LEFT at t=0. Ride straight along it, buildings on both sides.\n\n"
    "The FULL 90-degree rotation is the single most important goal: the final heading must be "
    "perpendicular (90 degrees left) to the first frame's heading. Do not stop rotating early, "
    "do not keep crossing straight ahead.\n\n"
    "Camera height about 1.6 meters, smooth continuous motion, natural head stabilization, "
    "no cuts, no dissolves. Photorealistic urban environment, brick buildings, daylight."
)

def extract_last_frame(video: Path, out_png: Path) -> None:
    with av.open(str(video)) as c:
        frames = [f.to_ndarray(format="rgb24") for f in c.decode(c.streams.video[0])]
    Image.fromarray(frames[-1]).save(out_png)

def i2v(prompt, prefix, seed, first_png):
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
        "16": {"class_type": "LoadImage", "inputs": {"image": first_png}},
        "17": {"class_type": "MiniMaxH3ImageToVideo", "inputs": {
            "clip": ["3", 0], "vae": ["4", 0], "prompt": prompt,
            "width": 1344, "height": 768, "length": 124, "first_frame": ["16", 0]}},
        "7": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
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
    }, "client_id": "zcode-h3-chain"}

for si, seed in enumerate(SEEDS, 1):
    src = OUT_DIR / f"h3chain_seg1_s{si}_00001_.mp4"
    if not src.exists():
        print("MISSING seg1:", src.name); continue
    png_name = f"turn_seg1_last_s{si}.png"
    extract_last_frame(src, IN_DIR / png_name)
    wf = i2v(PROMPT_SEG2, f"h3chain_seg2_s{si}", seed, png_name)
    with open(JOBS / f"seg2_s{si}.json", "w") as f:
        json.dump(wf, f, indent=1)
    print("built seg2_s{} with first_frame={} (seed {})".format(si, png_name, seed))
print("done")
