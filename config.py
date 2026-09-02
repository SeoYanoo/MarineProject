import os
from pathlib import Path


def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(os.getenv(name, default))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(value, maximum))


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, default))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(value, maximum))

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

# 학습 완료 후 아래 경로에 모델 파일을 배치하세요.
YOLO_MODEL_PATH = MODELS_DIR / "yolov5_ship.pt"
VIT_MODEL_PATH = MODELS_DIR / "vit_ship_classifier"

# 탐지 후처리. Streamlit Cloud에서는 환경변수로 조절할 수 있다.
YOLO_CONF_THRESHOLD = _env_float("DETECTION_CONFIDENCE", 0.30, 0.05, 0.95)
NMS_IOU_THRESHOLD = _env_float("DETECTION_NMS_IOU", 0.45, 0.05, 0.95)
MAX_DETECTIONS_PER_FRAME = _env_int("MAX_DETECTIONS_PER_FRAME", 40, 1, 300)

# 영상 처리량 제한. 원본 재생 시간은 유지하면서 출력/추론 프레임만 줄인다.
VIDEO_INFERENCE_FPS = _env_float("VIDEO_INFERENCE_FPS", 4.0, 0.2, 15.0)
VIDEO_OUTPUT_FPS = _env_float("VIDEO_OUTPUT_FPS", 15.0, 1.0, 30.0)
VIDEO_MAX_INFERENCE_CALLS = _env_int("VIDEO_MAX_INFERENCE_CALLS", 240, 10, 1000)
VIDEO_MAX_DIMENSION = _env_int("VIDEO_MAX_DIMENSION", 1280, 480, 1920)
INFERENCE_MAX_DIMENSION = _env_int("INFERENCE_MAX_DIMENSION", 1280, 480, 1920)
INFERENCE_JPEG_QUALITY = _env_int("INFERENCE_JPEG_QUALITY", 82, 60, 95)

# 함정 크기 클래스
SHIP_SIZE_CLASSES = ("대형", "중형", "소형")

# ViT 세부 톤급 클래스 (데모 / 학습 레이블 예시)
FINE_CLASSES = {
    "대형": ("1,000톤급", "1,500톤급", "3,000톤급", "5,000톤급"),
    "중형": ("300톤급", "500톤급", "800톤급"),
    "소형": ("50톤급", "100톤급", "200톤급"),
}
