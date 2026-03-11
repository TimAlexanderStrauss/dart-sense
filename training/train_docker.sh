#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-yolov8n.pt}"
DATA="${DATA:-data/d1_to_d7.docker.yaml}"
EPOCHS="${EPOCHS:-100}"
IMGSZ="${IMGSZ:-800}"
BATCH="${BATCH:-16}"
WORKERS="${WORKERS:-8}"
DEVICE="${DEVICE:-0}"
PROJECT="${PROJECT:-runs}"
NAME="${NAME:-baseline_n}"

docker build -f docker/Dockerfile.train -t dart-sense-train .

docker run --rm --gpus all \
  -e YOLO_CONFIG_DIR=/workspace/.yolo \
  -v "$(pwd)":/workspace \
  -w /workspace \
  dart-sense-train \
  yolo detect train \
    data="$DATA" \
    model="$MODEL" \
    epochs="$EPOCHS" \
    imgsz="$IMGSZ" \
    batch="$BATCH" \
    workers="$WORKERS" \
    device="$DEVICE" \
    project="$PROJECT" \
    name="$NAME"
