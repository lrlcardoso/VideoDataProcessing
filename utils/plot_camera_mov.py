"""
==============================================================================
Title:          Transform Magnitude Plotter
Description:    Loads a pickle file containing per-frame camera motion data 
                (transform magnitudes) and visualizes it using Matplotlib.
                Also prints the last modified timestamp of the file.
Author:         Lucas R. L. Cardoso
Project:        VRRehab_UQ-MyTurn
Date:           2025-04-15
Version:        1.0
==============================================================================
Usage:
    python plot_camera_mov.py

Dependencies:
    - Python >= 3.x
    - Required libraries: pickle, matplotlib, os, datetime

Changelog:
    - v1.0: [2025-04-15] Initial release
==============================================================================
"""

import pickle
import matplotlib.pyplot as plt
import os
from datetime import datetime
from scipy.signal import filtfilt, butter
import numpy as np


def detect_movement_segments(pkl_path, threshold=10, smooth_window=15,
                              min_movement_duration=10, min_static_duration=120, fps=30):
    # Load transform magnitudes
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    magnitudes = [frame.get("transform_magnitude", 0) or 0 for frame in data]
    magnitudes = np.array(magnitudes)

    # Smooth signal using zero-lag Butterworth filter
    b, a = butter(N=2, Wn=1/smooth_window, btype='low')
    smoothed = filtfilt(b, a, magnitudes)

    # Initial movement flags based on threshold
    movement_flags = (smoothed > threshold).astype(int)

    # Helper to remove short segments
    def remove_short_segments(flags, min_len, target_val):
        in_segment = False
        start_idx = 0
        for i, val in enumerate(flags):
            if val == target_val and not in_segment:
                in_segment = True
                start_idx = i
            elif val != target_val and in_segment:
                if i - start_idx < min_len:
                    flags[start_idx:i] = 1 - target_val
                in_segment = False
        if in_segment and len(flags) - start_idx < min_len:
            flags[start_idx:] = 1 - target_val
        return flags

    # Post-process
    movement_flags = remove_short_segments(movement_flags, min_movement_duration, target_val=1)
    movement_flags = remove_short_segments(movement_flags, min_static_duration, target_val=0)

    # Print movement intervals in mm:ss
    def format_time(frame_idx):
        total_seconds = frame_idx / fps
        minutes = int(total_seconds // 60)
        seconds = int(total_seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"

    print("Detected camera movement intervals:")
    in_motion = False
    for i, flag in enumerate(movement_flags):
        if flag == 1 and not in_motion:
            in_motion = True
            start = i
        elif flag == 0 and in_motion:
            in_motion = False
            end = i - 1
            print(f"→ Movement from {format_time(start)} to {format_time(end)}")
    if in_motion:
        print(f"→ Movement from {format_time(start)} to {format_time(len(movement_flags) - 1)}")

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(magnitudes, label="Original", color='lightgray')
    plt.plot(smoothed, label="Smoothed", color='blue')
    plt.plot(movement_flags * max(smoothed), label="Movement Flag", color='red', alpha=0.5)
    plt.xlabel("Frame Index")
    plt.ylabel("Transform Magnitude")
    plt.title("Detected Camera Movement Segments")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return movement_flags

# Example usage:
pkl_file_path = r"C:\Users\s4659771\Documents\MyTurn_Project\Data\Processed\P01\Session2_20250210\Video\VR\Camera1\2025-02-10 10-25-27_kinematic_data_filtered.pkl"

mod_time = os.path.getmtime(pkl_file_path)
formatted_time = datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M:%S")

print("Last modified:", formatted_time)

detect_movement_segments(pkl_file_path)
