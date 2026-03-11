param(
    [string]$Model = "yolov8n.pt",
    [string]$Data = "data/d1_to_d7.docker.yaml",
    [int]$Epochs = 100,
    [int]$ImgSz = 800,
    [string]$Batch = "16",
    [int]$Workers = 8,
    [string]$Device = "0",
    [string]$Project = "runs",
    [string]$RunName = "baseline_n"
)

$ErrorActionPreference = "Stop"

docker build -f docker/Dockerfile.train -t dart-sense-train .

docker run --rm --gpus all `
  -e YOLO_CONFIG_DIR=/workspace/.yolo `
  -v "${PWD}:/workspace" `
  -w /workspace `
  dart-sense-train `
  yolo detect train `
    data=$Data `
    model=$Model `
    epochs=$Epochs `
    imgsz=$ImgSz `
    batch=$Batch `
    workers=$Workers `
    device=$Device `
    project=$Project `
    name=$RunName
