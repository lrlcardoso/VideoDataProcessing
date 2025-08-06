# RehabTrack_Workflow – Video Data Processing

This is part of the [RehabTrack_Workflow](https://github.com/lrlcardoso/RehabTrack_Workflow): a modular Python pipeline for **tracking and analysing physiotherapy movements**, using video and IMU data.  
This stage prepares cleaned, annotated, and structured video data for subsequent analysis.

---

## 📌 Overview

This stage performs:
- **Pose estimation** using YOLO-Pose
- **Person re-identification** via TorchReID embeddings
- **Target filtering** using cosine similarity and shoulder midpoints
- **Camera movement detection** using ORB features
- **Dynamic video annotation** with bounding boxes and skeleton overlays

**Inputs:**
- Raw session video recordings (e.g., in .mkv)
- Model weights (YOLO-Pose, TorchReID)

**Outputs:**
- `.pkl` file with per-frame detections & metadata
- Downsampled and trimmed video
- Annotated video showing detected people and keypoints

---

## 📂 Repository Structure

```
Video_Data_Processing/
├── main.py               # Main entry point
├── config.py             # Configurable parameters & paths
├── utils/                # Processing and helper modules
│   ├── data_extraction.py
│   ├── filter_detection.py
│   ├── logfile_utils.py
│   ├── video_processing.py
│   ├── visualization.py
│   └── file_utils.py
├── requirements.txt
├── segmentation_example.txt
└── README.md
```

---

## 🛠 Installation

```bash
git clone https://github.com/yourusername/VideoDataProcessing.git
cd VideoDataProcessing
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> TorchReID may require additional system dependencies – see [TorchReID documentation](https://github.com/KaiyangZhou/deep-person-reid).

---

## 🚀 Usage

### 1️⃣ Multiple Detection
Extract detections (all individuals) from raw video.
```bash
python main.py --multipleDetection
```
**Inputs:**
- Recorded videos (no limit to length)

**Outputs:**
- `.pkl` with all detections
- Downsampled video
- Initial annotated video

### 2️⃣ Target Filtering
Filter detections to isolate the target person.
```bash
python main.py --filterDetection
```
**Inputs:**  
- **Recorded videos** – no restriction on length.  
- **Detections file** (`.pkl`) – containing all detections (output from the `--multipleDetection` step).  
- **Segmentation file** – specifies:  
  1. The time segments of the video to process.  
  2. A reference segment (60–90 s) where the target person is clearly visible with no obstructions.  
  3. The ID assigned to that person during this reference segment.  

Refer to the `segmentation_example.txt` file for detailed formatting and examples.  

**Outputs:**
- Filtered `.pkl` with `is_target`, `mid_points`, `avg_mid`, `search_box`
- Annotated video showing only the target

---

## 📦 Data Format

Each frame entry in the `.pkl` contains:
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

## 📖 Citation

If you use this stage in your research, please cite:
```
Cardoso, L. R. L. (2025). RehabTrack_Workflow: Video Data Processing. 
GitHub. https://doi.org/XXXX/zenodo.XXXXX
```

---

## 📝 License

Code: [MIT License](LICENSE)  
Documentation & figures: [CC BY 4.0](LICENSE-docs)

---

## 🤝 Acknowledgments

- [YOLO-Pose](https://github.com/itsyb/YOLOv7-Pose)  
- [TorchReID](https://github.com/KaiyangZhou/deep-person-reid)  
- OpenCV, FFmpeg, PyTorch, tqdm
