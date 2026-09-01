from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

# 학습 완료 후 아래 경로에 모델 파일을 배치하세요.
YOLO_MODEL_PATH = MODELS_DIR / "yolov5_ship.pt"
VIT_MODEL_PATH = MODELS_DIR / "vit_ship_classifier"

# YOLO 탐지 confidence threshold
YOLO_CONF_THRESHOLD = 0.25

# 함정 크기 클래스
SHIP_SIZE_CLASSES = ("대형", "중형", "소형")

# ViT 세부 톤급 클래스 (데모 / 학습 레이블 예시)
FINE_CLASSES = {
    "대형": ("1,000톤급", "1,500톤급", "3,000톤급", "5,000톤급"),
    "중형": ("300톤급", "500톤급", "800톤급"),
    "소형": ("50톤급", "100톤급", "200톤급"),
}
