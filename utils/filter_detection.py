"""
==============================================================================
Title:          Filter Detection and Target Similarity Estimation
Description:    Filters tracking results by computing cosine similarity between
                each person and a target embedding. Also detects camera motion,
                manages bounding box constraints, and saves midpoints and flags
                to a .pkl file for downstream visualization.
Author:         Lucas R. L. Cardoso
Project:        VRRehab_UQ-MyTurn
Date:           2025-04-15
Version:        1.1
==============================================================================
Usage:
    python filter_detection.py

Dependencies:
    - Python >= 3.x
    - Required libraries: cv2, torch, torchvision, torchreid, numpy,
                          scikit-learn, tqdm, pickle

Changelog:
    - v1.0: [2025-04-15] Initial release
    - v1.1: [2025-04-17] Extended logfile entries to include filtering parameters 
                         such as HISTORY_WINDOW, SHOULDER_WIDTH_SCALE, 
                         VERTICAL_SCALE_FACTOR, MIN_BOX_WIDTH, MAX_BOX_WIDTH, 
                         and THRESHOLD_CAMERA_MOV.
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

def compute_shoulder_midpoint(left, right):
    left = np.array(left)
    right = np.array(right)
    if (left[0] == 0 or left[1] == 0) or (right[0] == 0 or right[1] == 0):
        return None
    return ((left + right) / 2).tolist()

# === Camera Movement Detection ===
def detect_camera_movements(video_path, start_sec, end_sec, threshold,
                            smooth_window=15, min_movement_duration=10, min_static_duration=120):
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

    for _ in tqdm(range(start_frame + 1, end_frame), desc="Detecting camera movement"):
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
    if len(transform_magnitudes) >= smooth_window:
        b, a = butter(N=2, Wn=1/smooth_window, btype='low')
        smoothed = filtfilt(b, a, transform_magnitudes)
    else:
        smoothed = transform_magnitudes.copy()  # not enough frames to smooth

    # === Threshold and create initial movement flags ===
    flags = (np.array(smoothed) > threshold).astype(int)

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

    flags = remove_short_segments(flags, min_movement_duration, target_val=1)
    flags = remove_short_segments(flags, min_static_duration, target_val=0)

    # Pad flags to match full video length (if needed)
    camera_movement_flags = [False] * total_frames
    for i, val in enumerate(flags):
        camera_movement_flags[start_frame + 1 + i] = bool(val)

    # === Save movement plot ===
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    plot_output_path = os.path.join(os.path.dirname(video_path), f"{base_name}_camera_movement_plot.png")

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

# === Embedding Computation for Target Person ===
def create_target_embedding(video_path, model, transform, results, embedding_info):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    embedding_sum = None
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
                    count += 1

    cap.release()
    if count == 0:
        raise ValueError("No embeddings found for the provided intervals and IDs.")

    return embedding_sum / count

# === Main Detection Logic ===
def run_filter_detection(video_path, pickle_file, output_pkl, start_video_time=None, end_video_time=None, embedding_info=[]):
    
    print()
    print("-> Filtering Detection...")
    print()
    
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

    camera_movement_flags, transform_magnitudes = detect_camera_movements(
        video_path, start_video_time, end_video_time, THRESHOLD_CAMERA_MOV
    )

    model = load_model()
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    target_embedding = create_target_embedding(video_path, model, transform, results, embedding_info)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame, end_frame = int(start_video_time * fps), int(end_video_time * fps)
    recent_mids = []
    avg_mid = None
    tracking_output = []
    prev_box_width = MIN_BOX_WIDTH
    prev_box_height = MIN_BOX_WIDTH * VERTICAL_SCALE_FACTOR
    mag_idx = 0

    for frame_idx in tqdm(range(start_frame, end_frame), desc="Filtering frames"):
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
        frame_data["similarities"] = sims.tolist()

        # Compute midpoints for all persons
        mids = [compute_shoulder_midpoint(kp[LEFT_SHOULDER], kp[RIGHT_SHOULDER]) for kp in keypoints]
        frame_data["mid_points"] = mids

        # Select best match (target) based on max similarity
        best_idx = np.argmax(sims)
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
    log_file = os.path.splitext(video_path)[0] + "_logfile.txt"

    update_logfile(
        logfile=log_file,
        routine_name="filter_detection.py",
        history_window=HISTORY_WINDOW,
        shoulder_width_scale=SHOULDER_WIDTH_SCALE,
        vertical_scale_factor=VERTICAL_SCALE_FACTOR,
        min_box_width=MIN_BOX_WIDTH,
        max_box_width=MAX_BOX_WIDTH,
        threshold_camera_mov=THRESHOLD_CAMERA_MOV
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
