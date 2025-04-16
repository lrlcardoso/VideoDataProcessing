"""
==============================================================================
Title:          Video Processing Pipeline
Description:    Processes raw videos by adjusting resolution/FPS, extracting 
                movement data, and generating annotated outputs. Also supports
                filtering detections and regenerating annotated results.
Author:         Lucas R. L. Cardoso
Project:        VRRehab_UQ-MyTurn
Date:           2025-03-25
Version:        1.1
==============================================================================
Usage:
    python main.py [--multiple_detection] [--filter_detection]
    
Dependencies:
    - Python >= 3.x
    - Required libraries: os, time, argparse
    - Local modules: utils (file_utils, video_processing, data_extraction, 
      visualization, logfile_utils, filter_detection), config

Notes:
    - Ensure the `utils` package and `config.py` file are in the working directory.
    - Use --multiple_detection to process raw videos.
    - Use --filter_detection to apply filtering to previously processed videos.
    
Changelog:
    - v1.0: [2025-03-25] Initial release
    - v1.1: [2025-04-15] Added the --filter_detection logic
==============================================================================
"""

import os
import time
import argparse
from utils.file_utils import find_videos
from utils.video_processing import change_video_fps_and_res
from utils.data_extraction import extract_movement_data
from utils.visualization import generate_video
from utils.logfile_utils import update_logfile
from utils.filter_detection import run_filter_detection
from config import ROOT_DIR, SELECTED_PATIENTS, SELECTED_SESSIONS, TARGET_SUBFOLDERS, SELECTED_CAMERAS, FILTER_CONFIG


def parse_video_info(video_path):
    """
    Extracts metadata from the video path.

    Returns:
    - patient, session, camera: IDs parsed from folder structure
    - video_name: Full filename (with extension)
    - base_name: Filename without extension
    """
    patient = video_path.split("\\")[7]
    session = video_path.split("\\")[8]
    camera = video_path.split("\\")[11]
    video_name = os.path.basename(video_path)
    base_name = os.path.splitext(video_name)[0]
    return patient, session, camera, video_name, base_name


def process_video(video_path):
    """
    Handles full processing of a raw video:
    - Logs metadata
    - Converts FPS/resolution
    - Extracts movement data
    - Generates annotated output
    """
    patient, session, camera, video_name, base_name = parse_video_info(video_path)
    print(f"---\nProcessing: {patient} | {session} | {camera} | {video_name}\n---")

    output_dir = os.path.dirname(video_path).replace("Raw", "Processed")
    os.makedirs(output_dir, exist_ok=True)

    update_logfile(
        logfile=os.path.join(output_dir, f"{base_name}_logfile.txt"),
        routine_name="main.py",
        patient_number=patient,
        session_number=session,
        camera_number=camera,
        original_video_name=video_name,
    )

    processed_video = os.path.join(output_dir, f"{base_name}.mkv")
    pickle_file = os.path.join(output_dir, f"{base_name}_kinematic_data.pkl")
    annotated_video = os.path.join(output_dir, f"{base_name}_annotated.mp4")

    change_video_fps_and_res(
        video_path, 
        processed_video, 
        1920, 
        30
    )

    extract_movement_data(
        processed_video, 
        pickle_file
    )

    generate_video(
        pickle_file, 
        processed_video, 
        annotated_video,
        show_only_targets=False, 
        start_video_time=None,
        end_video_time=None
    )

    print()


def filter_video(video_path):
    """
    Handles filtering of already processed video:
    - Runs detection filtering on existing pickle data
    - Generates new annotated video with filtered targets
    """
    patient, session, camera, video_name, base_name = parse_video_info(video_path)
    print(f"---\nProcessing: {patient} | {session} | {camera} | {video_name}\n---")

    dir = os.path.dirname(video_path)
    pickle_file = os.path.join(dir, f"{base_name}_kinematic_data.pkl")
    output_video = os.path.join(dir, f"{base_name}_annotated_filtered.mp4")

    run_filter_detection(
        video_path, pickle_file, pickle_file,
        start_video_time=FILTER_CONFIG["start_video_time"],
        end_video_time=FILTER_CONFIG["end_video_time"],
        embedding_info=FILTER_CONFIG["embedding_info"]
    )

    generate_video(
        pickle_file, 
        video_path, 
        output_video,
        show_only_targets=True, 
        start_video_time=FILTER_CONFIG["start_video_time"],
        end_video_time=FILTER_CONFIG["end_video_time"]
    )


def main(multiple_detection=False, filter_detection=False):
    """
    Entry point for batch video processing.
    Selects mode based on args, finds videos, and applies processing.
    """
    start_time = time.time()

    if multiple_detection or filter_detection:
        mode = "Raw" if multiple_detection else "Processed"
        process_func = process_video if multiple_detection else filter_video

        videos = find_videos(ROOT_DIR, SELECTED_PATIENTS, SELECTED_SESSIONS, TARGET_SUBFOLDERS, SELECTED_CAMERAS, mode=mode)
        for video in videos:
            process_func(video)
    else:
        print('Error: Please choose one of the following options: --multiple_detection or --filter_detection.')

    print(f"---\nAll videos processed in {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}.\n---\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process videos and extract data.')
    parser.add_argument('--multiple_detection', action='store_true', help='Enable multiple detection.')
    parser.add_argument('--filter_detection', action='store_true', help='Enable filter detection.')
    args = parser.parse_args()

    main(args.multiple_detection, args.filter_detection)
