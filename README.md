# ⚽ AI Football Analysis System

A production-grade computer vision pipeline that analyzes broadcast football footage end-to-end — detecting players, tracking the ball, estimating real-world speed, assigning team possession, and rendering a live tactical radar minimap.

> Built with YOLOv8x · ByteTrack · OpenCV · PyTorch · KMeans · Optical Flow · Perspective Transform

---

## 📽️ Demo


https://github.com/user-attachments/assets/7b5eec68-17be-46fd-81c2-32234e078af4


---

## 🖼️ Screenshots

<!-- Add your screenshots here -->
| Team Color Detection (KMeans) | Radar Minimap | Speed Overlay |
|:---:|:---:|:---:|
|<img width="305" height="369" alt="image" src="https://github.com/user-attachments/assets/c4fbde6c-1027-4e7a-a910-065afbb5b260" />
| <img width="380" height="227" alt="image" src="https://github.com/user-attachments/assets/82d6dabd-ccff-4b53-adc1-410d92bb795a" />
| <img width="164" height="132" alt="image" src="https://github.com/user-attachments/assets/d8d44274-80e8-4922-b8e0-fc60d9b35b99" />

---

## 🧠 Features

- **Player & Ball Detection** — YOLOv8x pretrained on broadcast football footage. Detects players, goalkeepers, referees, and the ball with high accuracy
- **Multi-Object Tracking** — ByteTrack assigns stable IDs to all players across frames
- **Team Assignment** — KMeans clustering on kit colors automatically separates teams. Handles color stability across frames with brightness-anchored cluster ordering
- **Ball Interpolation** — Smoothly fills missing ball detections up to 3 frames using pandas interpolation with spike rejection
- **Camera Movement Compensation** — Lucas-Kanade optical flow tracks background features to remove camera pan/tilt from player positions
- **Perspective Transform** — Warps the camera trapezoid into a top-down real-world coordinate system (meters)
- **Speed & Distance Estimation** — Rolling window speed calculation per player in km/h with realistic clamping (max 42 km/h)
- **Ball Possession Tracking** — Per-team possession % including neutral state when no player is near the ball
- **Radar Minimap** — Live top-down tactical view rendered bottom-center of the output video
- **CSV Export** — Player stats (distance, avg speed, ball touches) and team stats (possession %) exported automatically

---

## 🏗️ Architecture

```
input_videos/
│
├── main.py                          # Entry point — full pipeline
│
├── trackers/
│   └── tracker.py                   # YOLOv8 detection + ByteTrack + ball filtering
│
├── team_assigner/
│   └── team_assigner.py             # KMeans kit color clustering
│
├── player_ball_assigner/
│   └── player_ball_assigner.py      # Nearest player to ball assignment
│
├── camera_movement_estimator/
│   └── camera_movement_estimator.py # Optical flow camera compensation
│
├── view_transformer/
│   └── view_transformer.py          # Perspective transform to top-down view
│
├── speed_and_distance_estimator/
│   └── speed_and_distance_estimator.py  # Rolling window speed calculation
│
├── radar/
│   └── radar.py                     # Live tactical minimap renderer
│
├── export_csv_stats/
│   └── export_csv_stats.py          # CSV statistics export
│
├── models/
│   ├── football-player-detection.pt # YOLOv8x — players, goalkeeper, referee
│   ├── football-ball-detection.pt   # YOLOv8x — ball only
│   └── football-pitch-detection.pt  # YOLOv8x — pitch keypoints (optional)
│
├── stubs/                           # Cached detection results (auto-generated)
├── input_videos/                    # Place your input video here
├── output_videos/                   # Annotated output video
└── Statistics/                      # Exported CSV files
```

---

## ⚙️ Pipeline Order

```python
# 1. Load video frames
# 2. Run detection + tracking (YOLOv8 + ByteTrack) — cached to stubs
# 3. Interpolate ball positions
# 4. Add foot/center positions to all tracks
# 5. Estimate & subtract camera movement (optical flow)
# 6. Perspective transform → real-world coordinates (meters)
# 7. Calculate speed & distance per player
# 8. Assign team colors (KMeans)
# 9. Assign ball to nearest player
# 10. Draw annotations + radar + stats overlay → write video
# 11. Export CSV statistics
```

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/football-analysis.git
cd football-analysis
```

### 2. Install dependencies

```bash
pip install ultralytics supervision opencv-python numpy pandas scikit-learn torch
```

### 3. Download models

```python
# Run this once — downloads the 3 official Roboflow sports models
import os, gdown

os.makedirs("models", exist_ok=True)

models = {
    "models/football-player-detection.pt": "https://github.com/roboflow/sports/releases/download/v0.1.0/football-player-detection.pt",
    "models/football-ball-detection.pt":   "https://github.com/roboflow/sports/releases/download/v0.1.0/football-ball-detection.pt",
    "models/football-pitch-detection.pt":  "https://github.com/roboflow/sports/releases/download/v0.1.0/football-pitch-detection.pt",
}

for path, url in models.items():
    gdown.download(url, path, quiet=False)
```

### 4. Add your video

```
input_videos/your_match.mp4
```

### 5. Calibrate the perspective transform

Run `find_vertices.py` once to click the 4 pitch corners on your specific video:

```bash
python find_vertices.py
```

Click in this order: **top-left → top-right → bottom-right → bottom-left** along the visible touchlines. Copy the output into `view_transformer/view_transformer.py`.

### 6. Run

```bash
python main.py
```

Output video will be saved to `output_videos/output_video.avi`. CSV stats to `Statistics/`.

---

## 📊 Output Statistics

### player_stats.csv

| player_id | team | total_distance_covered (m) | avg_speed_km_per_hr | ball_touches |
|-----------|------|---------------------------|---------------------|--------------|
| 5 | 1 | 312.4 | 14.2 | 7 |
| 12 | 2 | 289.1 | 12.8 | 3 |

### team_stats.csv

| team | possession_frames | possession_percent |
|------|------------------|--------------------|
| Team 1 | 842 | 48.3% |
| Team 2 | 756 | 43.4% |
| Neutral | 144 | 8.3% |

---

## 🔧 Configuration

### Ball detection filter (tracker.py)
```python
# Tune based on your video resolution
if area < 150 or area > 900:   # bbox area in pixels²
    continue
if conf < 0.5:                  # confidence threshold
    continue
```

### Interpolation (tracker.py)
```python
tracker.interpolate_ball_positions(tracks['ball'], max_gap=3, max_jump=250)
# max_gap  — max consecutive missing frames to fill
# max_jump — max pixel distance before treating as spike
```

### Speed estimation (speed_and_distance_estimator.py)
```python
self.frame_window = 10   # frames for rolling speed window
self.frame_rate = 25     # match your video fps
# Speed clamped at 42 km/h (world record sprint)
```

### Perspective transform (view_transformer.py)
```python
court_width = 68     # full pitch width in meters (always 68)
court_length = 52.5  # visible pitch length in meters (adjust per video)
```

---

## 🛠️ Known Limitations

- Team color assignment assumes two teams with visually distinct kits. White vs light gray may cause issues
- Speed accuracy depends on correct `court_length` calibration
- Players fully outside the perspective trapezoid won't have speed data
- Ball detection degrades when the ball is occluded or in the air for extended periods

---

## 📦 Models

All three models are YOLOv8x (~140MB each) trained by [Roboflow](https://roboflow.com) specifically on broadcast football footage.

| Model | Classes | mAP50 |
|-------|---------|-------|
| football-player-detection.pt | player, goalkeeper, referee | ~0.98 |
| football-ball-detection.pt | ball | ~0.85 |
| football-pitch-detection.pt | pitch keypoints | — |

---

## 🙏 Acknowledgements

- [Roboflow Sports](https://github.com/roboflow/sports) — pretrained football models
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — detection framework
- [supervision](https://github.com/roboflow/supervision) — detection utilities and ByteTrack

---

## 📄 License

MIT License — feel free to use, modify, and build on this.
