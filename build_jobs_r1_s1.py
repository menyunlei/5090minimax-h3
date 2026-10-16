# R1 chain S1: 243-frame (10.1s) Ref2V with map R1_orig_nl.png. Canal old town:
# ride north up the side street, up and over the stone arch bridge WHILE MOVING,
# CUT mid-motion at the bridge crest with canal water on both sides. Prefix h3rt1_.
import json
from pathlib import Path

JOBS = Path(r"C:\Users\yunlei\.zcode\workspace\default\h3_map_baseline\jobs_r1")
JOBS.mkdir(exist_ok=True)

SCENE = (
    "a southern Chinese canal old town in Jiangnan style: white plastered walls, black "
    "clay-tile roofs, grey stone-slab streets, wooden two-storey shophouses with hanging "
    "red lanterns and fabric signboards, a gentle stone arch bridge over a green canal, "
    "moored wooden boats, bright summer daylight"
)

POV = (
    "FIRST-PERSON BICYCLE CAMERA, non-negotiable: shot from a camera mounted on the bicycle's "
    "handlebar at exactly 1.6 meters above the ground, looking straight ahead horizontally. "
    "The handlebar grips and both of the rider's hands holding them are ALWAYS VISIBLE at the "
    "bottom edge of the frame, in every frame. The ground surface fills the lower half of the "
    "frame; houses and shopfronts pass at eye level; the horizon stays near the middle. The "
    "camera NEVER rises above 1.6 m. Absolutely NO aerial view, NO drone shot, NO bird's-eye "
    "view, NO top-down view.\n\n"
)

S1 = (
    "<Picture 1> is the authoritative spatial map of the world, not a visual-style reference. "
    "Treat it as a top-down geometric specification of " + SCENE + ". Do not imitate the "
    "appearance of the map; infer plausible 3D terrain and architecture.\n\n"
    + POV +
    "ONE CONTINUOUS RIDE, no stopping anywhere (total 10.1 seconds, 24 fps):\n"
    "- 0.0s-4.0s: ride straight ahead along a grey stone-slab side street at steady cycling "
    "speed; white plaster walls and black-tile roofs of houses glide past on BOTH sides at "
    "eye level, fabric shop awnings swaying.\n"
    "- 4.0s-7.5s: ahead the street meets a canal and rises onto a gentle stone arch bridge; "
    "WITHOUT slowing down, ride up the smooth bridge ramp — the surface visibly curves "
    "upward, carved stone balustrades rise along both sides, green canal water with moored "
    "boats opens up BELOW on both sides.\n"
    "- 7.5s-10.1s: cresting the arch mid-bridge, still rolling at cruise speed; canal water "
    "visible on BOTH sides through the balustrades, the old-town main street with red "
    "lanterns visible beyond the far slope ahead. The clip CUTS at the crest while still "
    "moving at full speed — do NOT slow down, do NOT cut inside the clip.\n"
    "The rolling bridge ramp climbed WHILE MOVING is the single most important goal: at "
    "10.1s the camera is at the bridge crest mid-canal, still in motion."
)

def ref2v(prompt, prefix, seed, length, image="R1_orig_nl.png"):
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
    }, "client_id": "zcode-h3-r1"}

if __name__ == "__main__":
    wf = ref2v(S1, "h3rt1_s1", 42, 243)
    with open(JOBS / "h3rt1_s1.json", "w") as f:
        json.dump(wf, f, indent=1)
    print("built h3rt1_s1 (243 frames = 10.1s)")
