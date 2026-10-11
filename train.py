"""Fine-tune YOLO on your own (preprocessed and augmented) dataset.

1. Put your images/labels in YOLO format (see data.yaml).
2. Run: python train.py
The best weights are saved to runs/detect/train/weights/best.pt
"""
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # start from pretrained weights
model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    # built-in augmentation
    fliplr=0.5,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=10,
    scale=0.5,
    mosaic=1.0,
)
metrics = model.val()
print("mAP50:", metrics.box.map50)
