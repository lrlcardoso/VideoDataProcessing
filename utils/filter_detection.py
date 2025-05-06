"""
==============================================================================
Title:          Filter Detection and Target Similarity Estimation
Description:    Filters tracking results by computing cosine similarity and 
                color histogram similarity between each person and a target 
                embedding. Also detects camera motion, manages bounding box 
                constraints, and saves midpoints and flags to a .pkl file for 
                downstream visualization.
Author:         Lucas R. L. Cardoso
Project:        VRRehab_UQ-MyTurn
Date:           2025-04-15
Version:        1.3
==============================================================================
Usage:
    python filter_detection.py

Dependencies:
    - Python >= 3.x
    - Required libraries: cv2, torch, torchvision, torchreid, numpy,
                          scikit-learn, tqdm, pickle

Changelog:
    - v1.0: [2025-04-15] Initial release
    - v1.1: [2025-04-17] Extended logfile entries to include filtering 
                         parameters such as HISTORY_WINDOW, 
                         SHOULDER_WIDTH_SCALE, VERTICAL_SCALE_FACTOR, 
                         MIN_BOX_WIDTH, MAX_BOX_WIDTH, and 
                         THRESHOLD_CAMERA_MOV.
    - v1.2: [2025-04-24] Run the filter for each segment. Target embedding is 
                         computed outside the filter method. Also updated the 
                         status display.
    - v1.3: [2025-05-06] Added target color histogram computation to the 
                         embedding definition. Combined cosine and histogram 
                         similarities for better re-identification accuracy. 
                         Also fixed a bug where incorrect frame positions were 
                         being processed.
==============================================================================
"""

import os
import cv2
import pickle
import numpy as np
import torch
import torchreid
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from utils.logfile_utils import update_logfile
from scipy.signal import filtfilt, butter
import matplotlib.pyplot as plt

# === Constants ===
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
HISTORY_WINDOW = 15
SHOULDER_WIDTH_SCALE = 0.8
VERTICAL_SCALE_FACTOR = 1.3
MIN_BOX_WIDTH = 100
MAX_BOX_WIDTH = 280
THRESHOLD_CAMERA_MOV = 10
SMOOTH_WINDOW = 15
MIN_MOVEMENT_DURATION = 10
MIN_STATIC_DURATION = 120
ALPHA = 0.8  # weight for cosine sim
BETA = 0.2   # weight for color histogram

# === Helper definition to print (on the logfile) the intervals in which camera movement was detected ===
def get_camera_movement_intervals_from_flags(flags, fps, start_seg):
    """
    Returns a list of (start_time, end_time) for all intervals in which
    camera_movement_flags are True. Times are formatted as hh:mm:ss.hh.
    """
    intervals = []
    in_motion = False
    for i, flag in enumerate(flags):
        if flag and not in_motion:
            in_motion = True
            start = i
        elif not flag and in_motion:
            in_motion = False
            end = i - 1
            intervals.append((start, end))
    if in_motion:
        intervals.append((start, len(flags) - 1))

    def format_time(idx):
        total_seconds = (idx / fps) - start_seg
        if total_seconds < 0:
            total_seconds = 0
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        hundredths = int((total_seconds - int(total_seconds)) * 100)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{hundredths:02d}"

    return [(format_time(start), format_time(end)) for start, end in intervals]

# === Device setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True

# === Utility Functions ===
def get_crop(frame, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    return frame[y1:y2, x1:x2]

def get_embedding(image, model, transform):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(image)
    return emb.cpu().numpy()

def load_model():
    # model = torchreid.models.build_model('resnet50', num_classes=1000, pretrained=True)

    model = torchreid.models.build_model(
        name='pcb_p6', #'osnet_ain_x1_0',
        num_classes=1,         # use 1 for inference, or set accordingly when fine-tuning
        loss='softmax_triplet',        # or 'softmax_triplet' if training
        pretrained=True
    )
    model.eval()
    model.to(device)
    return model

model = load_model()
transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# === Embedding Computation for Target Person ===
def create_target_embedding(video_path, results, embedding_info):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    embedding_sum = None
    hist_sum = None
    count = 0

    for start, stop, target_id in embedding_info:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))
        for frame_idx in tqdm(range(int(start * fps), int(stop * fps)), desc=f"Creating embedding for ID {target_id}"):
            if frame_idx >= len(results):
                break
            ret, frame = cap.read()
            if not ret:
                break
            frame_data = results[frame_idx]
            boxes, ids = frame_data.get("boxes"), frame_data.get("ids")
            if boxes is None or len(boxes) == 0:
                continue
            for box, id_ in zip(boxes, ids):
                if id_ == target_id:
                    emb = get_embedding(get_crop(frame, box), model, transform)
                    embedding_sum = emb if embedding_sum is None else embedding_sum + emb
                    hist = compute_histogram(get_crop(frame, box))
                    hist_sum = hist if hist_sum is None else hist_sum + hist
                    count += 1

    cap.release()
    if count == 0:
        raise ValueError("No embeddings found for the provided intervals and IDs.")

    return embedding_sum / count, hist_sum / count

def compute_histogram(image, bins=(8, 8, 8)):
    # Convert BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Compute a 3D HSV histogram
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist, norm_type=cv2.NORM_L1)
    return hist.flatten()

def compare_histograms(hist1, hist2):

    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

def compute_shoulder_midpoint(left, right):
    left = np.array(left)
    right = np.array(right)
    if (left[0] == 0 or left[1] == 0) or (right[0] == 0 or right[1] == 0):
        return None
    return ((left + right) / 2).tolist()

# === Camera Movement Detection ===
def detect_camera_movements(video_path, output_pkl, start_sec, end_sec):
    transform_magnitudes = []
    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    end_sec = min(end_sec, total_frames / fps)
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret, prev_frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to read the starting frame for movement detection.")
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    kp1, des1 = orb.detectAndCompute(prev_gray, None)

    frame_number = start_frame + 1

    for _ in tqdm(range(start_frame + 1, end_frame), desc=f"🔄 [1/4] Detecting camera movement"):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp2, des2 = orb.detectAndCompute(gray, None)

        transform_magnitude = 0
        if des1 is not None and des2 is not None:
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            if len(matches) > 10:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
                if M is not None:
                    dx, dy = M[0, 2], M[1, 2]
                    transform_magnitude = np.sqrt(dx**2 + dy**2)

        transform_magnitudes.append(transform_magnitude)
        kp1, des1 = kp2, des2
        frame_number += 1

    cap.release()

    # === Smooth the transform magnitudes ===
    if len(transform_magnitudes) >= SMOOTH_WINDOW:
        b, a = butter(N=2, Wn=1/SMOOTH_WINDOW, btype='low')
        smoothed = filtfilt(b, a, transform_magnitudes)
    else:
        smoothed = transform_magnitudes.copy()  # not enough frames to smooth

    # === Threshold and create initial movement flags ===
    flags = (np.array(smoothed) > THRESHOLD_CAMERA_MOV).astype(int)

    # === Post-process to remove short spikes ===
    def remove_short_segments(arr, min_len, target_val):
        arr = arr.copy()
        in_segment = False
        start_idx = 0
        for i, val in enumerate(arr):
            if val == target_val and not in_segment:
                in_segment = True
                start_idx = i
            elif val != target_val and in_segment:
                if i - start_idx < min_len:
                    arr[start_idx:i] = 1 - target_val
                in_segment = False
        if in_segment and len(arr) - start_idx < min_len:
            arr[start_idx:] = 1 - target_val
        return arr

    flags = remove_short_segments(flags, MIN_MOVEMENT_DURATION, target_val=1)
    flags = remove_short_segments(flags, MIN_STATIC_DURATION, target_val=0)

    # Pad flags to match full video length (if needed)
    camera_movement_flags = [False] * total_frames
    for i, val in enumerate(flags):
        camera_movement_flags[start_frame + 1 + i] = bool(val)

    # === Save movement plot ===
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    plot_output_path = os.path.join(os.path.dirname(output_pkl), f"{base_name}_camera_movement_plot.png")

    plt.figure(figsize=(12, 6))
    plt.plot(transform_magnitudes, label="Original", color='lightgray')
    plt.plot(smoothed, label="Smoothed", color='blue')
    plt.plot(flags * max(smoothed), label="Movement Flag", color='red', alpha=0.5)
    plt.xlabel("Frame Index")
    plt.ylabel("Transform Magnitude")
    plt.title("Detected Camera Movement Segments")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_output_path)
    plt.close()

    return camera_movement_flags, transform_magnitudes

# === Main Detection Logic ===
def run_filter_detection(video_path, pickle_file, output_pkl, start_video_time=None, end_video_time=None, target_embedding=None, target_hist=None):
    
    if target_embedding is None:
        raise ValueError("target_embedding must be provided!")
    
    with open(pickle_file, "rb") as f:
        results = pickle.load(f)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps
    cap.release()

    if start_video_time is None:
        start_video_time = 0
    if end_video_time is None:
        end_video_time = video_duration

    camera_movement_flags, transform_magnitudes = detect_camera_movements(video_path, output_pkl, start_video_time, end_video_time)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame, end_frame = int(start_video_time * fps), int(end_video_time * fps)
    recent_mids = []
    avg_mid = None
    tracking_output = []
    prev_box_width = MIN_BOX_WIDTH
    prev_box_height = MIN_BOX_WIDTH * VERTICAL_SCALE_FACTOR
    mag_idx = 0

    for frame_idx in tqdm(range(start_frame, end_frame), desc=f"🔄 [2/4] Filtering frames"):

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        if frame_idx >= len(results):
            break

        ret, frame = cap.read()
        camera_moving = camera_movement_flags[frame_idx] if frame_idx < len(camera_movement_flags) else False

        if not ret or camera_moving:
            if frame_idx < len(results):
                frame_data = results[frame_idx]
                frame_data["transform_magnitude"] = transform_magnitudes[mag_idx] if mag_idx < len(transform_magnitudes) else 0

                # Set is_target = [False, False, ..., False] to match box count
                num_boxes = len(frame_data.get("boxes", []))
                frame_data["is_target"] = [False] * num_boxes

            mag_idx += 1
            avg_mid = None
            recent_mids.clear()
            tracking_output.append([False] * len(frame_data.get("boxes", [])))
            continue

        frame_data = results[frame_idx]
        frame_data["transform_magnitude"] = transform_magnitudes[mag_idx] if mag_idx < len(transform_magnitudes) else 0
        mag_idx += 1

        boxes = frame_data.get("boxes", [])
        keypoints = frame_data.get("keypoints")
        if boxes is None or len(boxes) == 0 or keypoints is None:
            tracking_output.append([])
            continue

        crops = [get_crop(frame, box) for box in boxes]
        images = torch.stack([transform(c) for c in crops]).to(device)
        with torch.no_grad():
            embs = model(images).cpu().numpy()
        # Compute embeddings and similarities for ALL persons
        sims = cosine_similarity(target_embedding, embs)[0]

        # Now enhance with color histogram:
        hist_scores = []
        for crop in crops:
            hist = compute_histogram(crop)
            hist_score = compare_histograms(target_hist, hist)
            hist_scores.append(hist_score)

        # Combine both scores (you can tune the weights later)
        combined_scores = ALPHA * sims + BETA * np.array(hist_scores)

        frame_data["similarities"] = combined_scores.tolist()

        # Compute midpoints for all persons
        mids = [compute_shoulder_midpoint(kp[LEFT_SHOULDER], kp[RIGHT_SHOULDER]) for kp in keypoints]
        frame_data["mid_points"] = mids

        # Select best match (target) based on max similarity
        best_idx = np.argmax(combined_scores)
        target_kps = keypoints[best_idx]

        mid = compute_shoulder_midpoint(target_kps[LEFT_SHOULDER], target_kps[RIGHT_SHOULDER])

        if mid is not None:
            shoulder_dist = abs(target_kps[LEFT_SHOULDER][0] - target_kps[RIGHT_SHOULDER][0])
            box_width = min(max(SHOULDER_WIDTH_SCALE * shoulder_dist, MIN_BOX_WIDTH), MAX_BOX_WIDTH)
            box_height = VERTICAL_SCALE_FACTOR * box_width
            prev_box_width = box_width
            prev_box_height = box_height
        else:
            box_width = prev_box_width
            box_height = prev_box_height

        if (
            mid is not None and avg_mid is not None and
            abs(mid[0] - avg_mid[0]) <= box_width / 2 and
            abs(mid[1] - avg_mid[1]) <= box_height / 2
        ) or avg_mid is None:
            recent_mids.append(mid)
        else:
            found = False
            for i, kp in enumerate(keypoints):
                if i == best_idx:
                    continue
                m = compute_shoulder_midpoint(kp[LEFT_SHOULDER], kp[RIGHT_SHOULDER])
                if m is None:
                    continue
                shoulder_dist = abs(kp[LEFT_SHOULDER][0] - kp[RIGHT_SHOULDER][0])
                candidate_box_width = min(max(SHOULDER_WIDTH_SCALE * shoulder_dist, MIN_BOX_WIDTH), MAX_BOX_WIDTH)
                candidate_box_height = VERTICAL_SCALE_FACTOR * candidate_box_width
                if abs(m[0] - avg_mid[0]) <= candidate_box_width / 2 and abs(m[1] - avg_mid[1]) <= candidate_box_height / 2:
                    best_idx = i
                    recent_mids.append(m)
                    prev_box_width = candidate_box_width
                    prev_box_height = candidate_box_height
                    found = True
                    break
            if not found:
                best_idx = None

        if len(recent_mids) > HISTORY_WINDOW:
            recent_mids.pop(0)
        valid_mids = [m for m in recent_mids if m is not None]
        avg_mid = np.mean(valid_mids, axis=0) if valid_mids else None
        frame_data["avg_mid"] = avg_mid

        # Define search box
        search_box = None
        if best_idx is not None and avg_mid is not None:
            target_kps = keypoints[best_idx]
            left_shoulder = target_kps[LEFT_SHOULDER]
            right_shoulder = target_kps[RIGHT_SHOULDER]
            if (
                left_shoulder is not None and right_shoulder is not None and
                (left_shoulder[0] != 0 and left_shoulder[1] != 0) and
                (right_shoulder[0] != 0 and right_shoulder[1] != 0)
            ):
                shoulder_dist = abs(left_shoulder[0] - right_shoulder[0])
                box_width = min(max(SHOULDER_WIDTH_SCALE * shoulder_dist, MIN_BOX_WIDTH), MAX_BOX_WIDTH)
                box_height = VERTICAL_SCALE_FACTOR * box_width
                half_w = int(box_width / 2)
                half_h = int(box_height / 2)
                top_left = (int(avg_mid[0] - half_w), int(avg_mid[1] - half_h))
                bottom_right = (int(avg_mid[0] + half_w), int(avg_mid[1] + half_h))
                search_box = [top_left, bottom_right]

        flags = [(i == best_idx) for i in range(len(boxes))]
        frame_data["is_target"] = flags
        frame_data["search_box"] = [list(top_left), list(bottom_right)] if search_box else None
        mids = [compute_shoulder_midpoint(kp[LEFT_SHOULDER], kp[RIGHT_SHOULDER]) for kp in keypoints]
        frame_data["mid_points"] = mids
        tracking_output.append(flags)

    cap.release()

    filtered_results = results[start_frame:end_frame]
    for i, flags in enumerate(tracking_output):
        if i < len(filtered_results):
            filtered_results[i]["is_target"] = flags

    with open(output_pkl, "wb") as f:
        pickle.dump(filtered_results, f)

    # Call the update_logfile function
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    log_file = os.path.join(os.path.dirname(output_pkl), f"{base_name}_logfile.txt")

    intervals = get_camera_movement_intervals_from_flags(camera_movement_flags, fps, start_video_time)

    # Create a nicely formatted string for the logfile
    if intervals:
        camera_movement_log = "  - Camera Movement Intervals:\n"
        for i, (start, end) in enumerate(intervals, 1):
            camera_movement_log += f"      {i:02d}. {start} → {end}\n"
    else:
        camera_movement_log = "  - Camera Movement Intervals: None detected\n"

    update_logfile(
        logfile=log_file,
        routine_name="filter_detection.py",
        history_window=HISTORY_WINDOW,
        shoulder_width_scale=SHOULDER_WIDTH_SCALE,
        vertical_scale_factor=VERTICAL_SCALE_FACTOR,
        min_box_width=MIN_BOX_WIDTH,
        max_box_width=MAX_BOX_WIDTH,
        threshold_camera_mov=THRESHOLD_CAMERA_MOV,
        smooth_window=SMOOTH_WINDOW,
        min_movement_duration=MIN_MOVEMENT_DURATION,
        min_static_duration=MIN_STATIC_DURATION,
        alpha_combined_scores=ALPHA,
        beta_combined_scores=BETA,
        camera_movement_log=camera_movement_log
    )

# === Optional standalone run ===
if __name__ == "__main__":
    run_filter_detection(
        video_path="example_video.mp4",
        pickle_file="example_results.pkl",
        output_pkl="filtered_results.pkl",
        start_video_time=0,
        end_video_time=None,
        embedding_info=[(0, 10, 1)]  # Example: use first 10s for target with ID=1
    )
