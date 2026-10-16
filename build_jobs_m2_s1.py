# M2 S1: Ref2V with the real satellite map (river suspension bridge -> town junction).
import json
from pathlib import Path

BASE = Path(r"C:\Users\yunlei\.zcode\workspace\default\h3_map_baseline")
JOBS = BASE / "jobs_m2"
JOBS.mkdir(exist_ok=True)

SCENE = (
    "a small town river crossing in Britain: a road suspension bridge with white pylons and "
    "fanned white cables over a wide brown tidal river, moored leisure boats and a long dark "
    "barge below, and on the far bank a compact town junction with yellow box-junction hatch "
    "markings, a small tree-covered traffic island with a low white fence, and a high street "
    "of brick shops including a white-arched arcade front, summer daylight"
)

GEO = (
    "<Picture 1> is an aerial photograph used ONLY as a geometric route specification. The red "
    "line on it is the camera path through " + SCENE + ". Infer plausible 3D surroundings from "
    "the street layout. NEVER reproduce the aerial viewpoint of the picture itself, and under no "
    "circumstance let the picture's flat graphics appear in the world.\n\n"
)

POV = (
    "FIRST-PERSON BICYCLE CAMERA, non-negotiable: camera on the bicycle's handlebar at exactly "
    "1.6 meters above the ground, looking straight ahead horizontally; handlebar grips and both "
    "of the rider's hands are ALWAYS VISIBLE at the bottom edge of the frame; the ground fills "
    "the lower half; the horizon stays near the middle. The camera NEVER rises above 1.6 m. "
    "Absolutely NO aerial view, NO drone shot, NO bird's-eye view.\n\n"
)

S1 = GEO + POV + (
    "ONE CONTINUOUS RIDE, no stopping anywhere (total 10.0 seconds, 24 fps):\n"
    "- 0.0s-3.0s: riding north on the deck of the suspension bridge at steady cycling speed; the "
    "white railing is on the right at hand height; the tall white pylon tower with its fan of "
    "white cables stands on the RIGHT and grows closer; wide brown river water is visible on "
    "both sides below the parapets.\n"
    "- 3.0s-6.5s: pass the pylon base on the RIGHT; cables sweep past overhead-right; moored "
    "leisure boats and a long dark barge are visible far below on the LEFT over the railing.\n"
    "- 6.5s-10.0s: the bridge deck runs level toward the far bank; ahead, the town junction with "
    "its yellow box-junction hatch markings and the small tree-covered traffic island comes "
    "into view. The clip CUTS while still moving at full speed along the bridge — do NOT slow "
    "down, do NOT arrive at the far bank.\n"
    "The entire bridge ride must read as one unbroken push toward the far-bank junction."
)

def ref2v(prompt, prefix, seed, length, image="M2_route.png"):
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
        "16": {"class_type": "LoadImage", "inputs": {"image": image}},
        "17": {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {
            "clip": ["3", 0], "vae": ["4", 0], "prompt": prompt,
            "width": 1344, "height": 768, "length": length, "ref_image_size": "match",
            "ref_images.ref_image_1": ["16", 0]}},
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
    }, "client_id": "zcode-h3-m2"}

wf = ref2v(S1, "h3m2_s1", 42, 243)
with open(JOBS / "h3m2_s1.json", "w") as f:
    json.dump(wf, f, indent=1)
print("built jobs_m2/h3m2_s1.json")
