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

def plot_transform_magnitude(pkl_path):
    # Load the pickle file
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    # Extract transform magnitudes
    magnitudes = []
    for i, frame_data in enumerate(data):
        mag = frame_data.get("transform_magnitude", None)
        magnitudes.append(mag if mag is not None else 0)

    # Plot the transform magnitude per frame
    plt.figure(figsize=(12, 6))
    plt.plot(magnitudes, label="Transform Magnitude", color='blue')
    plt.xlabel("Frame Index")
    plt.ylabel("Transform Magnitude")
    plt.title("Camera Movement (Transform Magnitude per Frame)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Example usage:
pkl_file_path = r"C:\Users\s4659771\Documents\MyTurn_Project\Data\Processed\P01\Session3_20250217\Video\VR\Camera1\2025-02-17 10-08-21_kinematic_data.pkl"

mod_time = os.path.getmtime(pkl_file_path)
formatted_time = datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M:%S")

print("Last modified:", formatted_time)

plot_transform_magnitude(pkl_file_path)
