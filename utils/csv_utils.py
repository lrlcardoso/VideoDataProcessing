"""
==============================================================================
Title:          Filtered Skeleton Marker CSV Exporter
Description:    Extracts the target skeleton markers from filtered tracking 
                results (.pkl) and exports them as a CSV. Each row = one frame 
                (where a target is present), and columns include unix time, 
                segment-relative time, plus all target marker coordinates and 
                confidence scores.
Author:         Lucas R. L. Cardoso
Project:        VRRehab_UQ-MyTurn
Date:           2025-04-24
Version:        1.1
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
    - 'Unix Time': Absolute unix timestamp (in seconds) for each frame, 
      based on the video filename's start time, assuming Brisbane timezone.
    - 'Segment Time': time relative to the start of the segment (formatted as hh:mm:ss.ddd).
    - Marker order is 1, 2, ..., n and columns are labeled as: 1_x, 1_y, 1_conf, ..., n_x, n_y, n_conf.
    - Assumes 17 keypoints per skeleton. Adjust 'n_keypoints' as needed.
    - Adjust extraction of the target if your data structure differs.

Changelog:
    - v1.0: [2025-04-24] Initial release.
    - v1.1: [2025-04-25] Changed CSV to output only two time columns: 
            'Unix Time' (in seconds, Brisbane timezone, from video filename) 
            and 'Segment Time' (formatted as hh:mm:ss.ddd from segment start).
==============================================================================
"""

import csv
import cv2
import pickle
import os
import re
from tqdm import tqdm
from datetime import datetime, timezone, timedelta

def format_time_hms(seconds):
    """Convert seconds (float) to 'hh:mm:ss.ddd' string for easy reading."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02}:{minutes:02}:{secs:06.3f}"

def extract_unix_start_time_from_filename(video_path):
    """
    Extracts the datetime from the video filename (format: YYYY-MM-DD HH-MM-SS)
    and returns the unix timestamp in UTC seconds, adjusted for Brisbane time (UTC+10).
    """
    basename = os.path.basename(video_path)
    # Accepts space or underscore between date and time; dash or colon in time
    match = re.search(
        r'(\d{4}-\d{2}-\d{2})[ _](\d{2})[-:](\d{2})[-:](\d{2})',
        basename
    )
    if not match:
        # Search in the full path just in case
        match = re.search(
            r'(\d{4}-\d{2}-\d{2})[ _](\d{2})[-:](\d{2})[-:](\d{2})',
            video_path
        )
    if not match:
        raise ValueError("No valid timestamp (YYYY-MM-DD HH-MM-SS) found in video_path.")

    date_str, hour, minute, second = match.groups()
    brisbane = timezone(timedelta(hours=10))  # UTC+10, no DST
    dt = datetime.strptime(f"{date_str} {hour}:{minute}:{second}", "%Y-%m-%d %H:%M:%S")
    dt = dt.replace(tzinfo=brisbane)
    return dt.timestamp()

def generate_csv(video_path, pkl_file, output_csv, start_video_time):
    """
    Export filtered target skeleton markers to a CSV file.
    Each row: [Unix Time, Segment Time, keypoints...]
    
    Parameters:
        video_path (str): Path to the original video file (to get FPS & start time).
        pkl_file (str): Path to filtered results .pkl file.
        output_csv (str): Path to output CSV file.
        start_video_time (float): Segment start time (in seconds) in the full video.
    """
    # Get video FPS using OpenCV to calculate timing
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # Get the unix start time of the video (seconds since epoch, at first frame of the video)
    unix_video_start = extract_unix_start_time_from_filename(video_path)
    # Adjust for the start of this segment
    unix_segment_start = unix_video_start + start_video_time

    # Load filtered results from pickle file
    with open(pkl_file, 'rb') as f:
        results = pickle.load(f)

    n_keypoints = 17

    # Build CSV header: unix and segment time, marker coordinates/confidences, marker indices start at 1
    header = (
        ['Unix Time', 'Segment Time'] +
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
            for i, flag in enumerate(is_target):
                if flag and keypoints[i] is not None:
                    # For this frame, unix time is segment_start + frame_idx/fps
                    unix_time = unix_segment_start + (frame_idx / fps)
                    segment_time_str = format_time_hms(frame_idx / fps)
                    row = [
                        f"{unix_time:.3f}",
                        segment_time_str,
                    ]
                    for kpt in keypoints[i]:
                        row.extend(list(kpt) if len(kpt) == 3 else list(kpt) + [1.0])
                    writer.writerow(row)
                    break  # Only one target per frame is written
    print(f"✅ CSV saved: {output_csv}")

