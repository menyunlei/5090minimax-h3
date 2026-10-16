# Generic chained-hop builder for the 60s continuous ride.
# Usage: python chain_extend.py <prev_video.mp4> <anchor.png> <out_prefix> <prompt.txt>
# Anchor png is saved into the ComfyUI input dir; job JSON into jobs_yym2\.
import av, json, sys
from PIL import Image
from pathlib import Path

OUT_DIR = Path(r"C:\Users\yunlei\AppData\Local\Comfy-Desktop\ComfyUI-Shared\output")
IN_DIR = Path(r"C:\Users\yunlei\AppData\Local\Comfy-Desktop\ComfyUI-Shared\input")
JOBS = Path(r"C:\Users\yunlei\.zcode\workspace\default\h3_map_baseline\jobs_yym2")

POV = (
    "A first-person bicycle camera — handlebar grips and the rider's hands ALWAYS VISIBLE at "
    "the bottom of the frame, camera locked at exactly 1.6 meters above the ground, NEVER "
    "rising, no aerial view, no drone shot — "
)

prev_name, anchor_png, out_prefix, prompt_file = sys.argv[1:5]
length = int(sys.argv[5]) if len(sys.argv) > 5 else 124
spec = Path(prompt_file).read_text().strip()
prompt = POV + "continues from the first frame exactly as shown. " + spec

src = OUT_DIR / prev_name
fr = None
with av.open(str(src)) as c:
    for f in c.decode(c.streams.video[0]):
        fr = f.to_ndarray(format="rgb24")
Image.fromarray(fr).save(IN_DIR / anchor_png)
print("anchor saved:", anchor_png)

wf = {"prompt": {
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
    "16": {"class_type": "LoadImage", "inputs": {"image": anchor_png}},
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
        "video": ["14", 0], "filename_prefix": out_prefix, "format": "auto"}},
}, "client_id": "zcode-h3-yym2-cont"}

out = JOBS / (out_prefix.replace("h3yym2_", "") + ".json")
with open(out, "w") as f:
    json.dump(wf, f, indent=1)
print("built", out.name)
