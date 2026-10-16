# H3 steer stack watchdog: relaunches ComfyUI(:8189, GPU1) and control_server(:8787) when dead.
$BASE = "C:\Users\yunlei\.zcode\workspace\default\h3_map_baseline"
$VENV = "C:\Users\yunlei\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI (1)\ComfyUI\.venv\Scripts\python.exe"
$CUI  = "C:\Users\yunlei\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI (1)\ComfyUI"
$SH   = "C:\Users\yunlei\AppData\Local\Comfy-Desktop\ComfyUI-Shared"
$log  = Join-Path $BASE "watchdog.log"

function Test-Port($url) {
  try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 4 -Uri $url | Out-Null; return $true }
  catch { return $false }
}

function Get-RecentProc($match, $secs) {
  Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -match $match } |
    Where-Object { ((New-TimeSpan -Start $_.CreationDate -End (Get-Date)).TotalSeconds) -lt $secs }
}

$stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

if (-not (Test-Port "http://127.0.0.1:8189/system_stats")) {
  if (Get-RecentProc "main\.py --port 8189" 200) {
    Add-Content $log "$stamp 8189 port down but a fresh launcher exists -> skip (avoid duplicate)"
  } else {
    Add-Content $log "$stamp ComfyUI:8189 dead -> relaunching (GPU0 masked via CUDA_VISIBLE_DEVICES=1)"
    Start-Process -FilePath "cmd.exe" -WindowStyle Hidden -WorkingDirectory $CUI `
      -ArgumentList "/c","set CUDA_VISIBLE_DEVICES=1&& `"$VENV`" main.py --port 8189 --vram-headroom 3.0 --models-directory `"$SH\models`" --input-directory `"$SH\input`" --output-directory `"$SH\output`" --temp-directory `"$SH\temp`" >> `"$BASE\comfy8189_wd.log`" 2>> `"$BASE\comfy8189_wd.err.log`""
  }
}
$gpu0ok = ((& "nvidia-smi" "-i" "0" "--query-gpu=name" "--format=csv,noheader" 2>$null) -match "NVIDIA")
$gpu0limRaw = (& "nvidia-smi" "-i" "0" "--query-gpu=power.limit" "--format=csv,noheader,nounits" 2>$null)
$gpu0lim = 0.0; [double]::TryParse(($gpu0limRaw | Out-String).Trim(), [ref]$gpu0lim) | Out-Null
if (-not $gpu0ok) {
  Add-Content $log "$stamp GPU0 not present (PCIe drop?) -> 8190 lane relaunch SKIPPED"
} elseif ($gpu0lim -ge 560) {
  Add-Content $log "$stamp GPU0 at FULL power ($gpu0lim W) -> 8190 SKIPPED (policy: capped-only; run gpu_stack_fix.bat elevated)"
} elseif (-not (Test-Port "http://127.0.0.1:8190/system_stats")) {
  Add-Content $log "$stamp ComfyUI:8190 (GPU0 prefetch lane) dead -> relaunching"
  Start-Process -FilePath $VENV -WindowStyle Hidden -WorkingDirectory $CUI `
    -ArgumentList "main.py","--port","8190","--cuda-device","0","--vram-headroom","3.0",`
      "--models-directory","$SH\models","--input-directory","$SH\input",`
      "--output-directory","$SH\output","--temp-directory","$SH\temp" `
    -RedirectStandardOutput (Join-Path $BASE "comfy8190_wd.log") `
    -RedirectStandardError  (Join-Path $BASE "comfy8190_wd.err.log")
}
if (-not (Test-Port "http://127.0.0.1:8787/state")) {
  if (Get-RecentProc "control_server" 150) {
    Add-Content $log "$stamp 8787 port down but a fresh launcher exists -> skip (avoid duplicate)"
  } else {
    Add-Content $log "$stamp control_server:8787 dead -> relaunching"
    Start-Process -FilePath $VENV -WindowStyle Hidden -WorkingDirectory $BASE `
      -ArgumentList "control_server.py" `
      -RedirectStandardOutput (Join-Path $BASE "steer_wd.log") `
      -RedirectStandardError  (Join-Path $BASE "steer_wd.err.log")
  }
}
