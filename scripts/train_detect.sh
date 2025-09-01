#!/usr/bin/env bash
# YOLOv8 검출(예: dog_face, dog_body, cat_face, cat_body) 파인튜닝 스크립트
# ---------------------------------------------------------------------------
# 사전 준비:
#  1) ultralytics 패키지 설치 (requirements.txt에 포함)
#  2) dataset/detect/data.yaml 생성 (train/val 경로, 클래스 names 지정)
#     예시:
#       path: dataset/detect
#       train: images/train
#       val: images/val
#       names: [dog_face, dog_body, cat_face, cat_body]
#  3) 이미지/라벨(YOLO txt) 구조:
#       images/train/*.jpg
#       labels/train/*.txt   # 각 라벨 파일: class cx cy w h (normalized)
#       images/val/*.jpg
#       labels/val/*.txt
#
# 출력:
#  - runs/detect/train/weights/best.pt  (학습된 검출 모델)
#  - export_detect.sh로 ONNX 변환 가능
# ---------------------------------------------------------------------------

set -euo pipefail

# 경량 모델로 시작 후, 필요시 s/m 등급으로 상향
MODEL="yolov8n.pt"      # yolov8s.pt 로 변경하면 성능↑(시간/VRAM↑)
IMGSZ=640               # 입력 크기 (속도/정확도 트레이드오프)
EPOCHS=50               # 에폭 수 (데이터 양/품질에 맞춰 조정)
BATCH=16                # 배치 크기 (GPU 메모리에 맞춰 조정)
DEVICE=0                # GPU index (단일GPU=0)
DATA="dataset/detect/data.yaml"  # 데이터 설정 파일

echo "[INFO] Training YOLOv8 detect model..."
yolo detect train \
  model="$MODEL" \
  data="$DATA" \
  imgsz="$IMGSZ" \
  epochs="$EPOCHS" \
  batch="$BATCH" \
  device="$DEVICE" \
  amp=True              # 자동 혼합정밀(AMP)로 속도/VRAM 최적화

echo "[DONE] 학습 완료. 체크포인트 위치:"
echo "  runs/detect/train/weights/best.pt"
echo "→ ONNX로 내보내려면 scripts/export_detect.sh 실행"
