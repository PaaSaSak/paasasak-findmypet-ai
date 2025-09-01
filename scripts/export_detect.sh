#!/usr/bin/env bash
# YOLOv8 학습 가중치(.pt)를 ONNX로 변환하는 스크립트
# - 입력: runs/detect/train/weights/best.pt
# - 출력: runs/detect/train/weights/best.onnx
# 서버/도커에서 CPU 추론(ONNX Runtime)으로 서빙할 때 사용

set -euo pipefail

WEIGHTS="runs/detect/train/weights/best.pt"

if [ ! -f "$WEIGHTS" ]; then
  echo "[ERROR] 학습 가중치를 찾을 수 없습니다: $WEIGHTS"
  echo "        먼저 scripts/train_detect.sh로 학습을 진행하세요."
  exit 1
fi

echo "[INFO] Exporting YOLOv8 model to ONNX..."
# opset=17은 최신 ORT와 호환성 양호. 필요시 16~19에서 조정
yolo export model="$WEIGHTS" format=onnx opset=17

echo "[DONE] Exported:"
echo "  runs/detect/train/weights/best.onnx"
echo "→ 서빙 시 app.py에서 이 ONNX를 로드하여 '탐지 → 크롭' 단계에 사용할 수 있습니다."
