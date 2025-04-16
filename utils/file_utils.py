"""
==============================================================================
Title:          Video File Finder
Description:    Finds all video files for the selected patients and sessions 
                from the specified directories and subfolders, with dynamic 
                control over whether to look in Raw or Processed folders.
Author:         Lucas R. L. Cardoso
Project:        VRRehab_UQ-MyTurn
Date:           2025-03-25
Version:        1.2
==============================================================================
Usage:
    Called as a utility function from other scripts (not run directly).

Dependencies:
    - Python >= 3.x
    - Required libraries: os, glob

Changelog:
    - v1.0: [2025-03-25] Initial release
    - v1.1: [2025-04-15] Updated to support dynamic Raw/Processed folder structure using unified ROOT_DIR
    - v1.2: [2025-04-16] Support for camera selection via argument
==============================================================================
"""

import os
import glob

def find_videos(root_dir, patients, sessions, target_subfolders, selected_cameras, mode):
    """
    Finds all video files for the selected patients and sessions.

    Parameters:
    - root_dir: str, base directory where 'Raw' or 'Processed' folders exist.
    - patients: list of str, patient identifiers to look for.
    - sessions: list of str, session identifiers to look for.
    - target_subfolders: list of str, subfolders where videos are stored (e.g., ['Video']).
    - selected_cameras: list of str, e.g., ['Camera1'], ['Camera1', 'Camera2'].
    - mode: str, either 'Raw' or 'Processed'.

    Returns:
    - video_paths: list of str, paths to the found video files.
    """
    video_paths = []

    for patient in patients:
        patient_path = os.path.join(root_dir, mode, patient)
        if not os.path.exists(patient_path):
            print(f"Skipping {patient} (folder not found in {mode})")
            continue

        for session in sessions:
            session_dirs = glob.glob(os.path.join(patient_path, f"{session}_*"))

            for session_dir in session_dirs:
                video_dir = os.path.join(session_dir, "Video")
                if os.path.exists(video_dir):
                    for subdir in target_subfolders:
                        subfolder_path = os.path.join(video_dir, subdir)
                        if os.path.exists(subfolder_path):
                            for cam in selected_cameras:
                                cam_path = os.path.join(subfolder_path, cam)
                                if os.path.exists(cam_path):
                                    for ext in ["*.mp4", "*.mkv"]:
                                        for video_file in glob.glob(os.path.join(cam_path, ext)):
                                            if mode == "Processed":
                                                filename = os.path.basename(video_file)
                                                name_without_ext = os.path.splitext(filename)[0]
                                                if "_" in name_without_ext:
                                                    continue
                                            video_paths.append(video_file)

    return video_paths
