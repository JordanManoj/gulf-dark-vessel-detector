Set-Location "C:\Users\jorda\Downloads\Dark Vessel Detector"
if (Test-Path "runs\yolo26s") { Remove-Item -Recurse -Force "runs\yolo26s" }
Write-Host "=== Training yolo26s (batch 4) ===" -ForegroundColor Cyan
& .venv\Scripts\python.exe -W ignore -m src.train --data data/processed/ossdd --model models/yolo26s.pt --name yolo26s --batch 4 --workers 2 2>&1 | Tee-Object -FilePath results\train_yolo26s.log
if (Test-Path "runs\yolo26s\weights\best.pt") {
  Write-Host "=== Evaluating yolo26s ===" -ForegroundColor Cyan
  & .venv\Scripts\python.exe -W ignore -m src.eval_chips --data data/processed/ossdd --weights runs/yolo26s/weights/best.pt --name yolo26s 2>&1 | Tee-Object -FilePath results\eval_yolo26s.log
  Write-Host "ALL DONE" -ForegroundColor Green
} else { Write-Host "yolo26s training failed - see results\train_yolo26s.log" -ForegroundColor Red }
