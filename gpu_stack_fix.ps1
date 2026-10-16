# gpu_stack_fix.ps1 — RUN ELEVATED (管理员). Revives GPU0 without reboot if possible,
# caps GPU0 power to 300W (double-card-but-not-full-power policy), restarts the H3 lanes.
$BASE = "C:\Users\yunlei\.zcode\workspace\default\h3_map_baseline"
$log  = Join-Path $BASE "gpu_fix.log"
function L($m) { $s = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m"; Add-Content $log $s; Write-Output $s }

$idA = "PCI\VEN_10DE&DEV_2B85&SUBSYS_205710DE&REV_A1\0DFA57CA422DB04800"
$idB = "PCI\VEN_10DE&DEV_2B85&SUBSYS_205710DE&REV_A1\4&26a0e71&0&0DFA57CA422DB04800"

L "=== gpu_stack_fix start ==="
$probe = (nvidia-smi --query-gpu=index,name --format=csv,noheader 2>&1 | Out-String).Trim()
$lost  = (nvidia-smi 2>&1 | Out-String) -match "GPU is lost"
L "before: $probe"
L "GPU0 lost flag: $lost"

if ($lost) {
  L "cycling both RTX 5090 adapters (disable -> enable)..."
  foreach ($id in @($idA, $idB)) {
    $r = pnputil /disable-device "$id" /force 2>&1 | Out-String
    L "disable $($id.Substring(0,60))... -> $($r.Trim())"
  }
  Start-Sleep 5
  foreach ($id in @($idB, $idA)) {
    $r = pnputil /enable-device "$id" 2>&1 | Out-String
    L "enable  $($id.Substring(0,60))... -> $($r.Trim())"
    Start-Sleep 3
  }
  Start-Sleep 10
}

$after = nvidia-smi --query-gpu=index,name --format=csv,noheader 2>&1 | Out-String
L "after: $($after.Trim())"
$stillLost = (nvidia-smi 2>&1 | Out-String) -match "GPU is lost"

if ($stillLost) {
  L "RESULT: GPU0 still lost after device cycle. DRIVER NEEDS A REAL REBOOT."
  L "ACTION FOR USER: reboot the PC, then run this script again (it will then just set the power cap + start lanes)."
  exit 3
}

# GPU0 present -> apply the power cap (full-power runs are what drops it)
$lim = nvidia-smi -i 0 -pl 300 2>&1 | Out-String
L "power cap -pl 300 -> $($lim.Trim())"
if ($lim -notmatch "300") {
  $lim2 = nvidia-smi -i 0 -pl 400 2>&1 | Out-String
  L "fallback -pl 400 -> $($lim2.Trim())"
}
$check = nvidia-smi -i 0 --query-gpu=power.limit --format=csv,noheader 2>&1 | Out-String
L "GPU0 power.limit now: $($check.Trim())"

# kick the lane watchdog immediately (it relaunches whichever lane is missing)
schtasks /Run /TN "H3SteerWatchdog" 2>&1 | Out-String | ForEach-Object { L "watchdog kick: $($_.Trim())" }
L "waiting for lanes to come up..."
$ok8189 = $false; $ok8190 = $false
foreach ($i in 1..40) {
  Start-Sleep 8
  try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 4 http://127.0.0.1:8189/system_stats | Out-Null; $ok8189 = $true } catch {}
  try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 4 http://127.0.0.1:8190/system_stats | Out-Null; $ok8190 = $true } catch {}
  if ($ok8189 -and $ok8190) { break }
}
L "RESULT: 8189(GPU1 full)=$ok8189  8190(GPU0@300W)=$ok8190"
if ($ok8189 -and $ok8190) { L "DUAL-LANE IS UP. queued legs will render on GPU1; prefetch on capped GPU0." }
elseif ($ok8189) { L "Single lane (GPU1) is up; GPU0 lane not started (watchdog may keep it gated until cap confirmed)." }
else { L "Neither lane came up yet - check comfy8189_wd.err.log / comfy8190_wd.err.log" }
L "=== gpu_stack_fix end ==="
