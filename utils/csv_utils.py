"""
==============================================================================
Title:          Filtered Skeleton Marker CSV Exporter
Description:    Extracts the target skeleton markers from filtered tracking 
                results (.pkl) and exports them as a CSV. Each row = one frame 
                (where a target is present), and columns include absolute 
                (global) and segment-relative times plus all target marker 
                coordinates and confidence scores.
Author:         Lucas R. L. Cardoso
Project:        VRRehab_UQ-MyTurn
Date:           2025-04-24
Version:        1.0
==============================================================================
Usage:
    from utils.export_filtered_skeleton import generate_csv
    generate_csv(
        video_path=video_path,               # Path to original video file
        pkl_file=output_pickle_file,         # Path to filtered .pkl file
        output_csv="filtered_markers.csv",   # Desired output CSV path
        start_video_time=0                   # (float) Segment start time (sec) in full video
    )

Dependencies:
    - Python >= 3.x
    - Required libraries: pickle, csv, tqdm, cv2

Notes:
    - Only frames where the target is present are exported to the CSV.
    - 'Global Time': time relative to start of the original video (formatted and in seconds).
    - 'Segment Time': time relative to start of the segment (formatted and in seconds).
    - Marker order is 1, 2, ..., n and columns are labeled as: 1_x, 1_y, 1_conf, ..., n_x, n_y, n_conf.
    - Assumes 17 keypoints per skeleton. Adjust 'n_keypoints' as needed.
    - Adjust extraction of the target if your data structure differs.

Changelog:
    - v1.0: [2025-04-24] Initial release.
==============================================================================
"""

import csv
import cv2
import pickle
from tqdm import tqdm

def format_time_hms(seconds):
    """Convert seconds (float) to 'hh:mm:ss.ddd' string for easy reading."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02}:{minutes:02}:{secs:06.3f}"

def generate_csv(video_path, pkl_file, output_csv, start_video_time):
    """
    Export filtered target skeleton markers to a CSV file.
    
    Parameters:
        video_path (str): Path to the original video file (to get FPS).
        pkl_file (str): Path to filtered results .pkl file.
        output_csv (str): Path to output CSV file.
        start_video_time (float): Segment start time (in seconds) in the full video.
    """
    # Get video FPS using OpenCV to calculate timing
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # Load filtered results from pickle file
    with open(pkl_file, 'rb') as f:
        results = pickle.load(f)

    n_keypoints = 17

    # Build CSV header: times and all marker coordinates/confidences, marker indices start at 1
    header = (
        ['Global Time', 'Global Time (sec)', 'Segment Time', 'Segment Time (sec)'] +
        [f"{i+1}_{axis}" for i in range(n_keypoints) for axis in ("x", "y", "conf")]
    )

    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        # tqdm progress bar for extraction
        for frame_idx, frame_data in tqdm(
            enumerate(results),
            total=len(results),
            desc="🔄 [4/4] Extracting CSV file"
        ):
            is_target = frame_data.get("is_target")
            keypoints = frame_data.get("keypoints")
            if is_target is None or keypoints is None:
                continue
            # Only export data for the person in this frame who is marked as the target
            for i, flag in enumerate(is_target):
                if flag and keypoints[i] is not None:
                    global_time_sec = start_video_time + frame_idx/fps
                    segment_time_sec = frame_idx / fps
                    global_time_str = format_time_hms(global_time_sec)
                    segment_time_str = format_time_hms(segment_time_sec)
                    row = [
                        global_time_str,      # Human-readable global time
                        f"{global_time_sec:.3f}",  # Absolute global time in seconds
                        segment_time_str,     # Human-readable segment-relative time
                        f"{segment_time_sec:.3f}", # Segment-relative time in seconds
                    ]
                    # Flatten all marker coordinates/confidences in marker order
                    for kpt in keypoints[i]:
                        row.extend(list(kpt) if len(kpt) == 3 else list(kpt) + [1.0])
                    writer.writerow(row)
                    break  # Only one target per frame is written
    print(f"✅ CSV saved: {output_csv}")
