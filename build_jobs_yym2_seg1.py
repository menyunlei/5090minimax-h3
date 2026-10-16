# Yuanmingyuan v2: POV-locked prompts. Fix = explicit first-person bicycle embodiment
# (handlebars + rider's hands always in frame, camera locked at 1.6 m, anti-aerial negatives).
# Outputs use prefix h3yym2_. Same seeds as v1 for a controlled comparison.
import json, os
from pathlib import Path

JOBS = Path(r"C:\Users\yunlei\.zcode\workspace\default\h3_map_baseline\jobs_yym2")
os.makedirs(JOBS, exist_ok=True)
SEEDS = [42, 1337, 20261006]

SCENE = (
    "a Yuanmingyuan-style imperial garden park in Beijing: tree-lined gravel paths, a big "
    "ornamental lake with lotus leaves, white-marble ruins blocks of the Western Mansions, "
    "stone balustrades, a narrow stone causeway, Chinese pavilions in the distance, summer "
    "daylight"
)

POV = (
    "FIRST-PERSON BICYCLE CAMERA, non-negotiable: shot from a camera mounted on the bicycle's "
    "handlebar at exactly 1.6 meters above the ground, looking straight ahead horizontally. "
    "The handlebar grips and both of the rider's hands holding them are ALWAYS VISIBLE at the "
    "bottom edge of the frame, in every frame. The ground surface fills the lower half of the "
    "image; trees, walls and balustrades pass at eye level; the horizon stays near the middle "
    "of the frame. The camera NEVER rises above 1.6 m, NEVER looks down at the ground from "
    "above. Absolutely NO aerial view, NO drone shot, NO bird's-eye view, NO top-down or "
    "map-like overview.\n\n"
)

PROMPT_SEG1 = (
    "<Picture 1> is the authoritative spatial map of the world, not a visual-style reference. "
    "Treat it as a top-down geometric specification of " + SCENE + ". Do not imitate the "
    "appearance of the map; infer plausible 3D terrain and architecture.\n\n"
    + POV +
    "Generation task: a continuous photorealistic first-person bicycle ride north along the "
    "straight tree-lined gravel path, slow cycling speed, no cuts. Through the trees on your "
    "LEFT you see the lake surface with lotus leaves at eye level.\n\n"
    "CRITICAL ending constraint: the clip must END stopped right AT the stone plaza junction, "
    "facing straight north along the path, handlebar and hands still in frame. In the final "
    "frames the junction plaza lies just ahead, and the narrow stone causeway branches off to "
    "the LEFT toward the lake, seen from rider height. Do not enter the plaza; do not turn."
)

PROMPT_CONTRAST = (
    "<Picture 1> is the authoritative spatial map of the world, not a visual-style reference. "
    "Treat it as a top-down geometric specification of " + SCENE + ". Do not imitate the "
    "appearance of the map; infer plausible 3D terrain and architecture.\n\n"
    + POV +
    "Camera timeline (total 5.2 seconds, 24 fps), riding at cycling speed:\n"
    "- 0.0s-2.0s: ride straight north along the tree-lined gravel path; the lake is on your LEFT.\n"
    "- 2.0s-4.0s: at the stone plaza junction, turn LEFT by 90 degrees at constant smooth rate "
    "onto the stone causeway that crosses the lake.\n"
    "- 4.0s-5.2s: ride straight along the causeway with lake water on BOTH sides of the view.\n"
    "The left turn is MANDATORY: the heading at 5.2s must point 90 degrees left of the 0s "
    "heading, all at 1.6 m height with the handlebar in frame."
)

def ref2v(prompt, prefix, seed, image="yym_orig_nl.png"):
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
            "width": 1344, "height": 768, "length": 124, "ref_image_size": "match",
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
    }, "client_id": "zcode-h3-yym2"}

wf = ref2v(PROMPT_CONTRAST, "h3yym2_contrast_s1", 42)
with open(JOBS / "contrast_s1.json", "w") as f:
    json.dump(wf, f, indent=1)
print("built contrast_s1")
for si, seed in enumerate(SEEDS, 1):
    wf = ref2v(PROMPT_SEG1, f"h3yym2_seg1_s{si}", seed)
    with open(JOBS / f"seg1_s{si}.json", "w") as f:
        json.dump(wf, f, indent=1)
    print(f"built seg1_s{si} (seed {seed})")
print("done: 4 jobs")
