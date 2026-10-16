# M1 chain S1: 243f Ref2V conditioned on REAL satellite map+M1_route.png.
# Quiet curving residential lane -> S-curve north toward the junction. Cut mid-motion.
# Prefix h3m1_. NOTE: extra anti-aerial wording because satellite ref correlates with top-down views.
import json
from pathlib import Path

JOBS = Path(r"C:\Users\yunlei\.zcode\workspace\default\h3_map_baseline\jobs_m1")
JOBS.mkdir(exist_ok=True)

SCENE = (
    "a quiet British residential neighbourhood: a narrow two-lane tarmac road curving gently "
    "through brick houses with front gardens, hedgerows and low brick walls, parked cars along "
    "the kerbs, a distinctive large house with pale-green mansard roofs set back behind trees, "
    "a big mature tree canopy spreading over the junction ahead, white lane dashes, bright "
    "summer daylight"
)

POV = (
    "FIRST-PERSON BICYCLE CAMERA, non-negotiable: shot from a camera mounted on the bicycle's "
    "handlebar at exactly 1.6 meters above the ground, looking straight ahead horizontally. "
    "The handlebar grips and both of the rider's hands holding them are ALWAYS VISIBLE at the "
    "bottom edge of the frame, in every frame. The road surface fills the lower half of the "
    "frame; hedges, fences and house fronts pass at eye level; the horizon stays near the "
    "middle. The camera NEVER rises above 1.6 m. Absolutely NO aerial view, NO drone shot, "
    "NO bird's-eye view, NO top-down view.\n\n"
)

S1 = (
    "<Picture 1> is an aerial photograph used ONLY as a geometric route specification. The "
    "red line on it is the camera path through the neighbourhood: it starts at the bottom of "
    "the curving residential road, follows it north, then turns right at the junction onto "
    "the wider main road. Take the 3D world from " + SCENE + "; NEVER reproduce the aerial "
    "viewpoint of the picture itself.\n\n"
    + POV +
    "ONE CONTINUOUS RIDE, no stopping anywhere (total 10.1 seconds, 24 fps):\n"
    "- 0.0s-4.0s: ride north along the quiet residential road at steady cycling speed; "
    "hedgerows, low brick walls and garden fronts glide past on the LEFT at eye level, trees "
    "and glimpses of a pale-green mansard roof on the RIGHT; a parked car or two sits by the "
    "kerb.\n"
    "- 4.0s-7.0s: the road bends gently RIGHT then LEFT in a smooth S-curve; lean slightly "
    "into each bend WHILE MOVING like a real cyclist, no braking; the wing of the big tree "
    "canopy ahead grows over the road's end.\n"
    "- 7.0s-10.1s: road straightens toward the junction; ahead the wider main road crosses "
    "left-to-right with white dashes, the big tree canopy spread over its far side. The clip "
    "CUTS while still moving at full cruise speed toward the junction — do NOT slow down, do "
    "NOT turn yet, no cuts inside the clip."
)

def ref2v(prompt, prefix, seed, length, image="M1_route.png"):
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
    }, "client_id": "zcode-h3-m1"}

if __name__ == "__main__":
    wf = ref2v(S1, "h3m1_s1", 42, 243)
    with open(JOBS / "h3m1_s1.json", "w") as f:
        json.dump(wf, f, indent=1)
    print("built h3m1_s1 (243 frames = 10.1s)")
