<div align="center">

# ⚽ AI Football Analysis System

**Computer vision pipeline that turns raw broadcast footage into complete tactical intelligence.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8x-Ultralytics-orange?style=for-the-badge)](https://ultralytics.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-red?style=for-the-badge&logo=pytorch)](https://pytorch.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=for-the-badge&logo=opencv)](https://opencv.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

[Demo Video](#demo) · [Features](#features) · [Quick Start](#quick-start) · [Architecture](#architecture) · [Configuration](#configuration)

</div>

---

## 📽️ Demo

<!-- Replace with your actual output video GIF -->
<div align="center">
  <img src="assets/demo.gif" alt="Football Analysis Demo" width="900"/>
</div>

https://github.com/user-attachments/assets/e87c8d53-11e5-4aab-86e3-d5c21711ea5a

---

## 🖼️ What It Looks Like

<div align="center">

| Player Tracking + Team Colors | Radar Minimap | Speed & Distance |
|:---:|:---:|:---:|
| <img width="130" height="106" alt="image" src="https://github.com/user-attachments/assets/d2585e90-252f-48d2-b373-e92b9d80f24b" /> | <img width="376" height="226" alt="image" src="https://github.com/user-attachments/assets/ea16359b-f5fc-4be4-9fd3-a46df210f9e9" /> | <img width="152" height="187" alt="image" src="https://github.com/user-attachments/assets/16317caf-5d23-4152-a01f-5e317b4740c0" /> |

| KMeans Team Assignment | Ball Possession Stats | Top-Down View Transform |
|:---:|:---:|:---:|
| <img width="308" height="363" alt="image" src="https://github.com/user-attachments/assets/29ae5d16-fc2e-4275-9be2-f0a6e870b771" /> | <img width="495" height="128" alt="image" src="https://github.com/user-attachments/assets/28d74298-5b18-4f8d-8fe5-f8ce3b46b397" /> | ![vertices_visualization](https://github.com/user-attachments/assets/48f87d11-dfac-434f-ad7d-d6c05103dc33)


</div>

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎯 **Player Detection** | YOLOv8x detects players, goalkeepers, referees, and the ball every frame |
| 🔁 **Stable Tracking** | ByteTrack + custom stable ID remapper keeps 22 players at ~28 unique IDs |
| 👕 **Auto Team Assignment** | KMeans on kit colors — no manual labeling. Brightness-anchored for consistency |
| ⚽ **Ball Interpolation** | Fills up to 3 missed frames with spike rejection for seamless ball tracking |
| 📷 **Camera Compensation** | Lucas-Kanade optical flow removes camera pan/tilt from all positions |
| 📐 **Perspective Transform** | Warps broadcast trapezoid to real-world top-down coordinates (meters) |
| ⚡ **Speed Estimation** | Rolling window speed in km/h per player. Clamped at 42 km/h (world record) |
| 🗺️ **Radar Minimap** | Live top-down tactical view rendered bottom-center of every frame |
| 📊 **Possession Tracking** | Team 1%, Team 2%, Neutral% — updated live every frame |
| 📁 **CSV Export** | Player stats (speed, distance, touches) + team possession breakdown |

---

## 🚀 Quick Start

### 1. Clone

```bash
git clone https://github.com/yourusername/football-analysis.git
cd football-analysis
```

### 2. Install dependencies

```bash
pip install ultralytics supervision opencv-python numpy pandas scikit-learn torch torchvision
```

### 3. Download models

```python
# Run once — downloads all 3 official Roboflow YOLOv8x football models (~140MB each)
import os, urllib.request

os.makedirs("models", exist_ok=True)

models = {
    "models/football-player-detection.pt": "https://github.com/roboflow/sports/releases/download/v0.1.0/football-player-detection.pt",
    "models/football-ball-detection.pt":   "https://github.com/roboflow/sports/releases/download/v0.1.0/football-ball-detection.pt",
    "models/football-pitch-detection.pt":  "https://github.com/roboflow/sports/releases/download/v0.1.0/football-pitch-detection.pt",
}

for path, url in models.items():
    print(f"Downloading {path}...")
    urllib.request.urlretrieve(url, path)
    print(f"  ✓ Done")
```

### 4. Calibrate perspective transform (one-time)

```bash
python find_vertices.py
```

Click the 4 pitch corners in order: **top-left → top-right → bottom-right → bottom-left**. Copy the output into `view_transformer/view_transformer.py`.

### 5. Run

```bash
python main.py
```

| Output | Location |
|---|---|
| Annotated video | `output_videos/output_video.avi` |
| Player statistics | `Statistics/player_stats.csv` |
| Team statistics | `Statistics/team_stats.csv` |

---

## 🏗️ Architecture

```
football-analysis/
│
├── main.py                              # Pipeline entry point
│
├── trackers/
│   └── tracker.py                       # YOLOv8 inference + ByteTrack + ball filtering
│
├── stable_tracker/
│   └── stable_tracker.py                # Position-based ID stability layer
│
├── team_assigner/
│   └── team_assigner.py                 # KMeans kit color clustering
│
├── player_ball_assigner/
│   └── player_ball_assigner.py          # Nearest player → ball assignment
│
├── camera_movement_estimator/
│   └── camera_movement_estimator.py     # Lucas-Kanade optical flow compensation
│
├── view_transformer/
│   └── view_transformer.py              # Perspective warp → real-world meters
│
├── speed_and_distance_estimator/
│   └── speed_and_distance_estimator.py  # Rolling window speed (km/h)
│
├── radar/
│   └── radar.py                         # Live tactical minimap renderer
│
├── export_csv_stats/
│   └── export_csv_stats.py              # CSV statistics export
│
├── models/                              # YOLOv8x weights (download separately)
├── stubs/                               # Cached detections (auto-generated)
├── input_videos/                        # Your match footage goes here
├── output_videos/                       # Annotated output
└── Statistics/                          # CSV exports
```

---

## ⚙️ Pipeline

```
Video Frames
     │
     ▼
YOLOv8x Detection (players + ball)
     │
     ▼
ByteTrack → Stable ID Remapper (~28 IDs for 22 players)
     │
     ▼
Ball Interpolation (fills gaps ≤3 frames, rejects spikes)
     │
     ▼
Foot Position Extraction
     │
     ▼
Camera Movement Subtraction (Lucas-Kanade optical flow)
     │
     ▼
Perspective Transform → Real-World Coordinates (meters)
     │
     ▼
Speed & Distance Estimation (rolling 10-frame window)
     │
     ▼
KMeans Team Assignment (brightness-anchored)
     │
     ▼
Ball → Nearest Player Assignment + Possession Tracking
     │
     ▼
Annotated Video + Radar + Stats Overlay → output_video.avi
     │
     ▼
CSV Export (player_stats.csv + team_stats.csv)
```

---

## 📊 Output Statistics

### `player_stats.csv`

| player_id | team | total_distance_covered (m) | avg_speed_km_per_hr | ball_touches |
|-----------|------|---------------------------|---------------------|--------------|
| 4 | 1 | 487.3 | 16.2 | 12 |
| 7 | 2 | 312.8 | 11.4 | 3 |
| 15 | 1 | 198.4 | 8.7 | 0 |

### `team_stats.csv`

| team | possession_frames | possession_percent |
|------|------------------|--------------------|
| Team 1 | 1842 | 52.3% |
| Team 2 | 1356 | 38.5% |
| Neutral | 324 | 9.2% |

---

## 🔧 Configuration

### Ball detection filter
```python
# tracker.py — tune to your video resolution
if area < 150 or area > 900:   # bbox area in pixels²
    continue
if conf < 0.5:                  # minimum confidence
    continue
```

### Ball interpolation
```python
# main.py
tracker.interpolate_ball_positions(tracks['ball'],
    max_gap=3,      # max consecutive missing frames to fill
    max_jump=250    # max pixel jump before treating as spike
)
```

### Stable tracker
```python
# main.py — tune if players lose IDs too often
stable_tracker = StableTracker(
    max_distance=120,   # max pixel distance to match same player
    max_lost=90         # frames to keep track alive when off-screen
)
```

### Speed estimation
```python
# speed_and_distance_estimator.py
self.frame_window = 10   # rolling window size in frames
self.frame_rate  = 25    # match your video fps
# All speeds > 42 km/h are rejected as measurement errors
```

### Perspective transform
```python
# view_transformer.py — calibrate once per camera angle
court_width  = 68     # always 68m (FIFA standard pitch width)
court_length = 52.5   # meters visible in your camera frame
```

---

## 🧠 Key Technical Decisions

**Why two separate YOLO models?**
The ball is 10×10 pixels in broadcast footage — a combined model trained to detect both 10px balls and 200px players degrades at both. Separate YOLOv8x models, each specialized, dramatically outperform any combined approach.

**Why a custom stable ID layer on top of ByteTrack?**
ByteTrack alone generates 800+ unique IDs for 22 players across a full match due to occlusion and off-screen events. Speed calculation requires the same player to hold the same ID across the rolling window. The stable tracker brings this to ~28 IDs using position continuity matching.

**Why subtract camera movement before speed calculation?**
Without compensation, a player standing still while the camera pans appears to be sprinting. Optical flow on background features (pitch lines) isolates true player movement from camera movement.

---

## 🛠️ Known Limitations

- Team assignment assumes two teams with visually distinct kits — white vs cream may cause occasional misclassification
- Speed accuracy depends on accurate `court_length` calibration per video
- Players outside the perspective trapezoid have no speed/distance data
- Goalkeeper re-identification can fail after long off-screen periods

---

## 📦 Models Used

All models are **YOLOv8x** (~140MB) trained by [Roboflow](https://roboflow.com) on broadcast football footage.

| Model | Detects | Notes |
|-------|---------|-------|
| `football-player-detection.pt` | player, goalkeeper, referee | mAP50 ~0.98 |
| `football-ball-detection.pt` | ball | mAP50 ~0.85 |
| `football-pitch-detection.pt` | pitch keypoints | optional |

---

## 🙏 Credits

- [Roboflow Sports](https://github.com/roboflow/sports) — pretrained football detection models
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — detection framework
- [supervision](https://github.com/roboflow/supervision) — ByteTrack and detection utilities

---

## 📄 License

MIT — use it, fork it, build on it.

---

<div align="center">

**If this project helped you, drop a ⭐ — it means a lot.**

</div>
