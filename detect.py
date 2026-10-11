"""Real-time object detection and tracking with YOLO + OpenCV.

Usage:
    python detect.py                      # default webcam
    python detect.py --source video.mp4   # a video file
    python detect.py --model best.pt      # your own fine-tuned weights
Press 'q' to quit.
"""
import argparse
import time

import cv2
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Real-time object detection with YOLO")
    p.add_argument("--source", default="0", help="webcam index (0) or path to a video file")
    p.add_argument("--model", default="yolov8n.pt", help="path to YOLO weights")
    p.add_argument("--conf", type=float, default=0.4, help="confidence threshold")
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.model)

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")

    prev = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # track() keeps a persistent ID for every object across frames
        results = model.track(frame, persist=True, conf=args.conf, verbose=False)
        annotated = results[0].plot()

        # overlay object count and FPS
        now = time.time()
        fps = 1 / (now - prev) if now > prev else 0
        prev = now
        count = len(results[0].boxes)
        cv2.putText(annotated, f"Objects: {count}  FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Real-Time Object Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
