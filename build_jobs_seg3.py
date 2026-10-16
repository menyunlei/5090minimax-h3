# seg3: from each rot segment's last frame, ride straight along the NEW street.
import av, json
from PIL import Image
from pathlib import Path

OUT_DIR = Path(r"C:\Users\yunlei\AppData\Local\Comfy-Desktop\ComfyUI-Shared\output")
IN_DIR = Path(r"C:\Users\yunlei\AppData\Local\Comfy-Desktop\ComfyUI-Shared\input")
JOBS = Path(r"C:\Users\yunlei\.zcode\workspace\default\h3_map_baseline\jobs_chain")

SEEDS = [42, 1337, 20261006]

PROMPT_SEG3 = (
    "A first-person bicycle camera is stopped at the corner, already facing straight down "
    "the new street as in the first frame.\n\n"
    "Timeline (total 5.2 seconds): begin riding forward immediately and ride STRAIGHT along "
    "this street at cycling speed for the whole clip. Keep the camera centered on the road, "
    "buildings on both sides, camera height about 1.6 meters, smooth forward motion, natural "
    "head stabilization, realistic parallax. Do NOT turn, do NOT rotate the view, do NOT cut. "
    "Photorealistic urban street, brick buildings, daylight."
)

def last_frame(p):
    with av.open(str(p)) as c:
        fr = None
        for f in c.decode(c.streams.video[0]):
            fr = f.to_ndarray(format="rgb24")
        return fr

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
    src = OUT_DIR / f"h3rot_s{si}_00001_.mp4"
    png = f"turn_rot_last_s{si}.png"
    Image.fromarray(last_frame(src)).save(IN_DIR / png)
    wf = i2v(PROMPT_SEG3, f"h3seg3_s{si}", seed, png)
    with open(JOBS / f"seg3_s{si}.json", "w") as f:
        json.dump(wf, f, indent=1)
    print(f"built seg3_s{si} anchored on {png} (seed {seed})")
print("done: 3 seg3 jobs")
