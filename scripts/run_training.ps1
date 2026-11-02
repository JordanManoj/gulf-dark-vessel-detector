# Train + evaluate both detectors on OSSDD and save metrics (DECISIONS.md D10, D21, D23).
# Run from a normal PowerShell window:  powershell -ExecutionPolicy Bypass -File scripts\run_training.ps1
Set-Location (Split-Path $PSScriptRoot -Parent)
$PY = ".venv\Scripts\python.exe"
foreach ($m in "yolov8s", "yolo26s") {
    if (Test-Path "runs\$m") { Remove-Item -Recurse -Force "runs\$m" }   # avoid Ultralytics renaming to yolov8s2
    Write-Host "=== Training $m ===" -ForegroundColor Cyan
    & $PY -W ignore -m src.train --data data/processed/ossdd --model "models/$m.pt" --name $m --batch 8 --workers 2
    if (-not (Test-Path "runs\$m\weights\best.pt")) { Write-Host "Training $m failed, stopping." -ForegroundColor Red; break }
    Write-Host "=== Evaluating $m ===" -ForegroundColor Cyan
    & $PY -W ignore -m src.eval_chips --data data/processed/ossdd --weights "runs/$m/weights/best.pt" --name $m
}
Write-Host "ALL DONE. Metrics in results\metrics_*.json" -ForegroundColor Green
