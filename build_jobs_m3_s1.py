# M3 S1: Ref2V with the real satellite map (Milan tree-lined avenue, south->north).
import json
from pathlib import Path

BASE = Path(r"C:\Users\yunlei\.zcode\workspace\default\h3_map_baseline")
JOBS = BASE / "jobs_m3"
JOBS.mkdir(exist_ok=True)

SCENE = (
    "a broad tree-lined residential avenue in Milan: red-tile-roof apartment blocks on the LEFT, "
    "a dense avenue tree line and parked cars on the RIGHT, side streets with zebra crossings "
    "joining at intervals, a circular piazza with a central fountain opening to the RIGHT midway, "
    "and further north wide park lawns on the RIGHT beyond the trees, bright summer daylight"
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
    "- 0.0s-3.0s: the ride starts exactly at the southern junction, rolling NORTH onto the avenue "
    "over its zebra crossing at steady cycling speed; the wide asphalt stretches ahead, red-roof "
    "apartment blocks on the LEFT, a dense tree line with parked cars on the RIGHT.\n"
    "- 3.0s-6.5s: settle into the middle of the avenue at cruise speed; balconies and shutters of "
    "the apartment blocks flick past on the LEFT; sunlight dapples through the tree canopy on "
    "the RIGHT; a side street mouth slides by.\n"
    "- 6.5s-10.0s: keep riding straight north along the same avenue; the circular piazza with its "
    "central fountain begins to open up ahead-RIGHT beyond the tree line. The clip CUTS while "
    "still moving at full cruise speed — do NOT slow down, do NOT arrive anywhere.\n"
    "The entire ride must read as one unbroken northbound push up the same avenue."
)

def ref2v(prompt, prefix, seed, length, image="M3_route.png"):
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
    }, "client_id": "zcode-h3-m3"}

wf = ref2v(S1, "h3m3_s1", 42, 243)
with open(JOBS / "h3m3_s1.json", "w") as f:
    json.dump(wf, f, indent=1)
print("built jobs_m3/h3m3_s1.json")
