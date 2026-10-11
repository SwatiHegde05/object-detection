"""Backend API for the Real-Time Object Detection System (Flask + YOLO + OpenCV).

Endpoints
  GET  /health            -> service status and model info
  POST /detect            -> upload an image, get detections as JSON
  POST /detect/annotated  -> upload an image, get the image back with boxes drawn
  GET  /video_feed        -> live MJPEG stream with detection + tracking
                             (webcam by default, or set VIDEO_SOURCE)

Run:  python app.py
"""
import os
import threading

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from ultralytics import YOLO

MODEL_PATH = os.getenv("MODEL_PATH", "yolov8n.pt")
DEFAULT_CONF = float(os.getenv("CONF", "0.4"))
VIDEO_SOURCE = os.getenv("VIDEO_SOURCE", "0")  # webcam index or video file path

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB upload limit
CORS(app)  # lets a separate frontend (React, plain JS...) call this API

model = YOLO(MODEL_PATH)
model_lock = threading.Lock()  # one inference at a time on the shared model


# ---------- helpers ----------
def get_conf():
    """Read optional ?conf=0.5 from the query string."""
    try:
        return min(max(float(request.args.get("conf", DEFAULT_CONF)), 0.0), 1.0)
    except ValueError:
        return DEFAULT_CONF


def read_uploaded_image():
    """Return (image, None) or (None, (error_message, status_code))."""
    file = request.files.get("image")
    if file is None:
        return None, ("Send an image file in the 'image' form field.", 400)
    data = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        return None, ("Could not read the file as an image.", 400)
    return image, None


def to_detections(result):
    detections = []
    for box in result.boxes:
        x1, y1, x2, y2 = (round(float(v), 1) for v in box.xyxy[0])
        cls_id = int(box.cls[0])
        detections.append({
            "label": model.names[cls_id],
            "confidence": round(float(box.conf[0]), 3),
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        })
    return detections


def predict(image, conf):
    with model_lock:
        return model.predict(image, conf=conf, verbose=False)[0]


# ---------- routes ----------
@app.get("/health")
def health():
    return jsonify(status="ok", model=MODEL_PATH, classes=len(model.names))


@app.post("/detect")
def detect():
    image, err = read_uploaded_image()
    if err:
        return jsonify(error=err[0]), err[1]
    result = predict(image, get_conf())
    detections = to_detections(result)
    height, width = image.shape[:2]
    return jsonify(
        count=len(detections),
        image_size={"width": width, "height": height},
        detections=detections,
    )


@app.post("/detect/annotated")
def detect_annotated():
    image, err = read_uploaded_image()
    if err:
        return jsonify(error=err[0]), err[1]
    annotated = predict(image, get_conf()).plot()
    ok, buffer = cv2.imencode(".jpg", annotated)
    if not ok:
        return jsonify(error="Could not encode the result image."), 500
    return Response(buffer.tobytes(), mimetype="image/jpeg")


def generate_frames(source, conf):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        return
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            with model_lock:
                result = model.track(frame, persist=True, conf=conf, verbose=False)[0]
            annotated = result.plot()
            cv2.putText(annotated, f"Objects: {len(result.boxes)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            ok, buffer = cv2.imencode(".jpg", annotated)
            if not ok:
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + buffer.tobytes() + b"\r\n")
    finally:
        cap.release()


@app.get("/video_feed")
def video_feed():
    source = int(VIDEO_SOURCE) if VIDEO_SOURCE.isdigit() else VIDEO_SOURCE
    probe = cv2.VideoCapture(source)
    opened = probe.isOpened()
    probe.release()
    if not opened:
        return jsonify(error=f"Could not open video source: {VIDEO_SOURCE}"), 503
    return Response(generate_frames(source, get_conf()),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.errorhandler(413)
def too_large(_):
    return jsonify(error="File too large. Maximum size is 10 MB."), 413


if __name__ == "__main__":
    app.run(host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "5000")),
            threaded=True)
