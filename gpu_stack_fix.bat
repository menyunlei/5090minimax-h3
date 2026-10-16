@echo off
REM One-click: elevate and run gpu_stack_fix.ps1 (GPU0 revive + 300W cap + lane restart)
powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Process powershell -Verb RunAs -ArgumentList '-NoProfile -ExecutionPolicy Bypass -File \"C:\Users\yunlei\.zcode\workspace\default\h3_map_baseline\gpu_stack_fix.ps1\"'"
