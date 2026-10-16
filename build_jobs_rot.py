# seg2v2: PURE IN-PLACE ROTATION at the corner (no forward motion).
# Anchored to seg1's last frame; goal: full 90-deg LEFT yaw by ~4.5s. 3 seeds.
import json
from pathlib import Path

JOBS = Path(r"C:\Users\yunlei\.zcode\workspace\default\h3_map_baseline\jobs_chain")
SEEDS = [42, 1337, 20261006]

PROMPT_ROT = (
    "A first-person bicycle camera stands at the corner of an intersection, facing the "
    "crosswalk as in the first frame. The camera stays at the SAME SPOT at the corner for the "
    "entire clip — it only rotates in place, it does NOT ride forward, it does NOT cross the "
    "intersection.\n\n"
    "MANDATORY TIMELINE (total 5.2 seconds):\n"
    "- 0.0s-0.5s: hold still, then begin yawing LEFT at a constant rate.\n"
    "- 0.5s-4.0s: rotate the view smoothly and continuously LEFT, always in place.\n"
    "- 4.0s-5.2s: rotation complete — the camera has turned a FULL 90 degrees LEFT and now "
    "faces straight down the cross street that was on the LEFT at t=0. Hold this heading.\n\n"
    "The 90-degree in-place rotation is the single most important goal: the final heading must "
    "be exactly perpendicular to the first frame's heading. Do not stop early, do not drift "
    "back, do not move forward. Smooth continuous yaw, natural motion blur, no cuts, no "
    "dissolves. Photorealistic urban environment, brick buildings, daylight."
)

def i2v(prompt, prefix, seed, first_png, client="zcode-h3-chain"):
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
    }, "client_id": client}

for si, seed in enumerate(SEEDS, 1):
    wf = i2v(PROMPT_ROT, f"h3rot_s{si}", seed, f"turn_seg1_last_s{si}.png")
    with open(JOBS / f"rot_s{si}.json", "w") as f:
        json.dump(wf, f, indent=1)
    print(f"built rot_s{si} (in-place rotate, seed {seed})")
print("done: 3 jobs")
