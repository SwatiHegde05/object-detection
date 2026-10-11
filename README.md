# Real-Time Object Detection System

Detects and tracks multiple objects in live video using YOLO and OpenCV.
Each object keeps a persistent ID across frames, and the screen shows the live object count and FPS.

**Tech:** Python, YOLO (Ultralytics), OpenCV

## Setup
```bash
pip install -r requirements.txt
```

## Run
```bash
python detect.py                       # webcam
python detect.py --source video.mp4    # video file
python detect.py --model best.pt       # your fine-tuned model
```
Press `q` to quit.

## Fine-tuning
Prepare a dataset in YOLO format, edit `data.yaml`, then run:
```bash
python train.py
```
Training uses data augmentation (flip, colour shift, rotation, scale, mosaic).

## Demo
Add a screenshot or GIF here: `![demo](demo.gif)`

## Backend API (Flask)
Run the API server:
```bash
python app.py        # http://127.0.0.1:5000
```

| Method | Endpoint | What it does |
|---|---|---|
| GET | `/health` | Service status and model info |
| POST | `/detect` | Upload an image (`image` field), get detections as JSON |
| POST | `/detect/annotated` | Upload an image, get back the image with boxes drawn |
| GET | `/video_feed` | Live MJPEG stream with detection and tracking |

Optional `?conf=0.5` on any detection endpoint sets the confidence threshold.

Examples:
```bash
curl -F "image=@test.jpg" http://127.0.0.1:5000/detect
curl -F "image=@test.jpg" http://127.0.0.1:5000/detect/annotated -o result.jpg
```
Open `http://127.0.0.1:5000/video_feed` in a browser to see the live stream.

Settings (environment variables): `MODEL_PATH` (default `yolov8n.pt`), `VIDEO_SOURCE` (webcam index or video file, default `0`), `CONF`, `HOST`, `PORT`.
