# AGENTS.md — operational playbook for AI agents

You are an autonomous agent who just received this repo. This file tells you how to operate the pipeline **without re-deriving anything**. Read `README.md` first for context.

## Environment facts (this project's home box)

- OS: Windows 11, Git Bash shell. Python used for all pipeline scripts = the ComfyUI venv:
  `C:\Users\yunlei\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI (1)\ComfyUI\.venv\Scripts\python.exe`
- ComfyUI shared dirs: `C:\Users\yunlei\AppData\Local\Comfy-Desktop\ComfyUI-Shared\{models,input,output,temp}`
  - Conditioning maps and anchor images must be **copied into `input/`** before submitting a job that references them.
  - Rendered clips appear in `output/` as `<prefix>_00001_.mp4`.
- Server endpoints: compute instance on `http://127.0.0.1:8189` (GPU1, `--cuda-device 1 --vram-headroom 3.0`). A second instance may exist on 8188.
- **GPU policy**: this box is used over RDP; GPU0 drives the display. NEVER run CUDA work on GPU0 (filling it black-screens the user's session). GPU0 has also physically dropped off the PCIe bus before — if `nvidia-smi` loses it, only the user can reboot; keep working via GPU1.
- Multiple agent sessions may share the 8189 FIFO queue. Use unique filename prefixes per session. After EVERY `POST /prompt`, verify the returned `prompt_id` appears in `/queue` or `/history` — submissions during a server restart are silently swallowed; resubmit if absent.

## The node graph (what builders emit)

`UNETLoader(minimax_h3_fl2va_pruned_int8_convrot)` → `LoraLoaderModelOnly(turbo 8-step, 1.0)` → `CLIPLoader(qwen3vl_32b_minimax_h3_nvfp4_awq, type=minimax)` → `VAELoader` ×2 (video int8, audio fp32) → conditioning node → `RandomNoise(seed)` → `KSamplerSelect(euler)` → `BasicScheduler(simple, 8 steps, denoise 1.0)` → `SamplerCustomAdvanced` → `VAEDecode(+Audio)` → `CreateVideo(24fps)` → `SaveVideo(prefix)`.

- Segment-1 conditioning node: `MiniMaxH3ReferenceToVideo` with `"ref_images.ref_image_1": ["16", 0]` — the **dotted autogrow key must be spelled exactly this way** or the reference image is silently dropped (no error, just no conditioning).
- Chain conditioning node: `MiniMaxH3ImageToVideo` with `first_frame` = previous clip's actual last frame.
- Settings that work: 1344×768, 243 frames (≈10.1 s), seed 42, 8 steps. One job ≈ 6–7 min on one RTX 5090.
- The turbo 8-step LoRA MUTES audio; deliverables are video-only.

## One-shot chaining discipline (do not violate)

1. Write segment N's spec from segment N−1's **actual last frame** (extract it, look at it), never from the plan alone — the anchor dominates any text/image mismatch.
2. Every clip ends **mid-motion at cruise speed**. The phrase that works: "The clip CUTS while still moving at full speed — do NOT slow down, do NOT arrive."
3. Turns are executed as moving arcs INSIDE a clip ("WITHOUT slowing down, bend the path into a smooth continuous LEFT arc of 90 degrees … like a real cyclist leaning into a bend").
4. POV preamble (verbatim, both Ref2V and I2V): first-person bicycle camera, handlebar grips + rider's hands always visible at the bottom edge, camera locked at exactly 1.6 m, horizon mid-frame, "no aerial view, no drone shot, no bird's-eye view".
5. Compile = drop frame 0 of segments 2..N (it duplicates the anchor), fixed caption, minimap inset. Overlays that change per segment are forbidden EXCEPT smoothly continuous motion (the moving position dot + traveled trail are fine — established by user feedback: the ban targets text swaps that draw eyes to joins).

## Conditioning-map rules

- De-object every conditioning map: thin route line, hollow rings, small text, NO filled discs — otherwise salient graphics materialize inside the 3D world (red route painted on road; dots become buoys). Also keep the prompt clause "map is a flat planning legend … NEVER appears in the video world".
- Real satellite images: crop UI chrome; inpaint the centre crosshair by CLONING a nearby clean road strip (a solid patch reads as a grey block); draw the route on the image; open the prompt with "aerial photograph used ONLY as a geometric route specification … NEVER reproduce the aerial viewpoint". This combination produced zero aerial collapse.

## Verification discipline (mandatory before delivery)

1. Decode every delivered segment as a dense filmstrip (`diag_yym3_dense.py`) and eyeball it — metrics alone are not QC.
2. Compiled video: `verify_*.py` (join diffs vs median interframe motion; joins OK when < ~0.8× baseline), straddle sheet around each join, full-video strip (`strip_*_full.py`), moving-dot QC (`qc_m1_minimap.py`).
3. **Check the QC sheet's content matches the video being verified** — filename argument slips have produced green QCs of the wrong video.
4. Deliverable = the video file itself, numbers second (user preference).

## Known failure modes & fixes

- Server dead / jobs vanished → relaunch with explicit `--models/input/output/temp-directory` flags (NOT `--base-directory`; it breaks on missing `custom_nodes`).
- GPU0 "GPU is lost" (PCIe drop) → job dies; move work to the 8189 GPU1 instance; recovery = user reboots.
- Ref leak (map graphics in-world) → de-object the map + flat-legend clause; re-roll seed if contamination persists.
- Mid-chain spec mismatch (spec says water LEFT, frame shows water RIGHT) → interrupt the running job (`POST /interrupt`), rewrite the spec from the actual frame, resubmit.
- HUD/JS overlays: if you script browser-side loops on this box's desktop, timer throttling in background panes makes dt explode — use dt-based kinematics, clamp dt ≤ 0.6 s.
