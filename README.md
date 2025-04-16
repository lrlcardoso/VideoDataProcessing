# VRRehab_UQ-MyTurn

A modular Python pipeline for processing, filtering, and visualizing pose tracking data in virtual reality (VR) neurorehabilitation sessions. This project supports multi-step analysis workflows using YOLO-based detection, person Re-ID with deep embeddings, camera motion filtering, and dynamic video annotation.

---

## 📂 Project Structure

```
VRRehab_UQ-MyTurn/
├── main.py                         # Main script for orchestrating the pipeline
├── config.py                       # Configuration settings for filtering, paths, etc.
├── utils/
│   ├── data_extraction.py         # Initial YOLO pose extraction and detection
│   ├── filter_detection.py        # Target embedding, cosine similarity, filtering
│   ├── logfile_utils.py           # Logging system
│   ├── video_processing.py        # Pre-processing and trimming with FFmpeg
│   ├── visualization.py           # Annotates video with bounding boxes, skeletons, midpoints
│   └── file_utils.py              # File management helpers
├── pose_models/                   # YOLO-Pose models (e.g., yolo11n-pose.pt)
├── README.md
└── requirements.txt
```

---

## 🧠 Features

- 🔍 **YOLO-Pose Detection** with model selection
- 🧬 **Deep Re-Identification** using torchreid embeddings
- 🧠 **Target Filtering** via cosine similarity and dynamic shoulder midpoints
- 🎥 **Camera Movement Detection** using ORB + affine transform magnitude
- 🧰 **Modular Visualization** including midpoints and dynamic search boxes
- ⚙️ **Configurable Pipeline** by patient, session, and time range

---

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/VRRehab_UQ-MyTurn.git
cd VRRehab_UQ-MyTurn
```

2. Set up a virtual environment (optional but recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

> Note: TorchReID requires some system dependencies and may be installed from source. See: https://github.com/KaiyangZhou/deep-person-reid

---

## 🚀 Usage

### Step 1: Run Multiple Detection

```bash
python main.py --multipleDetection
```

Outputs:
- `.pkl` file with detection results
- Downsampled video
- Initial annotated video

---

### Step 2: Run Target Filtering

```bash
python main.py --filterDetection
```

Outputs:
- Filtered `.pkl` with `is_target`, `mid_points`, `avg_mid`, `search_box`
- Annotated video with only the target and key spatial cues

---

## 🧪 Data Format

Each frame in the `.pkl` contains:
```python
{
  "boxes": [...],
  "ids": [...],
  "keypoints": [...],
  "is_target": [...],
  "mid_points": [...],
  "avg_mid": [...],
  "search_box": [[x1, y1], [x2, y2]],
  "transform_magnitude": float
}
```

---

## 📝 License

This project is part of an academic research initiative at The University of Queensland. Contact [Lucas Cardoso](mailto:your.email@uq.edu.au) for usage or collaboration inquiries.

---

## 🤝 Acknowledgments

- [YOLO-Pose](https://github.com/itsyb/YOLOv7-Pose)
- [TorchReID](https://github.com/KaiyangZhou/deep-person-reid)
- OpenCV, FFmpeg, PyTorch, tqdm, and others
