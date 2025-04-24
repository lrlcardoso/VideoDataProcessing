"""
==============================================================================
Title:          Video Processing Pipeline
Description:    Processes raw videos by adjusting resolution/FPS, extracting 
                movement data, and generating annotated outputs. Also supports
                filtering detections and regenerating annotated results.
Author:         Lucas R. L. Cardoso
Project:        VRRehab_UQ-MyTurn
Date:           2025-03-25
Version:        1.2
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
    - v1.2: [2025-04-24] Run the filter for each segment and create the 
                         embedding only once, always using the Camera1 video 
                         and pkl file. Also updated the status display for the 
                         filter detection. Still need to update for he multiple 
                         detection.
==============================================================================
"""

import os
import time
import argparse
import pickle
from utils.file_utils import find_videos
from utils.video_processing import change_video_fps_and_res
from utils.data_extraction import extract_movement_data
from utils.visualization import generate_video
from utils.logfile_utils import update_logfile
from utils.filter_detection import run_filter_detection, create_target_embedding
from utils.segmentation_info import parse_segmentation_info
from utils.csv_utils import generate_csv
from config import ROOT_DIR, SELECTED_PATIENTS, SELECTED_SESSIONS, TARGET_SUBFOLDERS, SELECTED_CAMERAS


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

def format_time_for_log(timestr):
    """
    Converts float seconds or 'HH:MM:SS.mmm' string to 'hh:mm:ss.mmm'
    Always returns 2-digit hours, 2-digit minutes, 2-digit seconds, 3-digit ms.
    """
    if isinstance(timestr, (float, int)):
        h = int(timestr // 3600)
        m = int((timestr % 3600) // 60)
        s = int(timestr % 60)
        ms = int(round((timestr - int(timestr)) * 1000))
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
    elif isinstance(timestr, str):
        # Try to parse if possible
        import re
        match = re.match(r'(\d{1,2}):(\d{2}):(\d{2})(?:[.,](\d{1,6}))?', timestr)
        if match:
            h, m, s, ms = match.groups()
            h, m, s = int(h), int(m), int(s)
            ms = ms or "0"
            # Pad/truncate ms to 3 digits
            ms = ms.ljust(3, '0')[:3]
            return f"{h:02d}:{m:02d}:{s:02d}.{ms}"
        else:
            # Return as-is if not matched (fallback)
            return timestr
    else:
        return str(timestr)


def process_video(video_path):
    """
    Handles full processing of a raw video:
    - Logs metadata
    - Converts FPS/resolution
    - Extracts movement data
    - Generates annotated output
    """
    patient, session, camera, video_name, base_name = parse_video_info(video_path)
    print("="*100)
    print(f"📝 Processing: {patient} | {session} | {camera} | {video_name}")
    print("="*100)

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
    Handles filtering of already annotated video:
    - Runs detection filtering on existing pickle data
    - Generates new annotated video and pkl file with filtered targets for each segment of the session
    (uses the txt file from the manually segmentation to define the segments)
    """

    # Parse info
    patient, session, camera, video_name, base_name = parse_video_info(video_path)
    print("="*100)
    print(f"📝 Processing: {patient} | {session} | {camera} | {video_name}")
    print("="*100)

    # Always look for segmentation file in Camera1 folder
    # Split the path and find Raw/Processed and VR/NoVR folder dynamically
    dir_components = video_path.split(os.sep)
    try:
        # Find 'Raw' or 'Processed' in path
        raw_proc_idx = [i for i, p in enumerate(dir_components) if p in ("Raw", "Processed")][0]
        raw_or_processed = dir_components[raw_proc_idx]
        patient = dir_components[raw_proc_idx + 1]
        session = dir_components[raw_proc_idx + 2]
        video_idx = dir_components.index("Video", raw_proc_idx)
        vr_folder = dir_components[video_idx + 1]
    except (ValueError, IndexError):
        raise RuntimeError(f"Failed to determine Raw/Processed or 'FMA_and_VR', 'VR' or 'CT' folder in: {video_path}")

    camera1_dir = os.path.join(
        ROOT_DIR,
        raw_or_processed,
        patient,
        session,
        "Video",
        vr_folder,
        "Camera1"
    )

    segmentation_file = os.path.join(camera1_dir, f"{base_name}_segmentation.txt")
    if not os.path.isfile(segmentation_file):
        raise FileNotFoundError(f"Segmentation file not found: {segmentation_file}")

    # Parse embedding info and segments
    embedding_info, segments = parse_segmentation_info(segmentation_file)

    # Define the input file (processed pickle for the video)
    dir = os.path.dirname(video_path)
    pickle_file = os.path.join(dir, f"{base_name}_kinematic_data.pkl")

    # Compute the embedding ONCE, using the video on Camera1 and the respective pkl file
    embedding_video = os.path.join(camera1_dir, video_name)
    # Load kinematic data for embedding computation
    pickle_file_embedding = os.path.join(camera1_dir, f"{base_name}_kinematic_data.pkl")
    with open(pickle_file_embedding, 'rb') as f:
        results_embedding = pickle.load(f)
    target_embedding = create_target_embedding(embedding_video, results_embedding, embedding_info)

    # Prepare for duplicate segment names
    segment_count = {}
    for seg_name, seg_start, seg_end in segments:
        
        start_time = time.time()

        # "Session Duration" is skipped by the parser, but double check here
        if seg_name.lower() == "session duration":
            continue
        # Ensure unique folder per segment
        if seg_name not in segment_count:
            segment_count[seg_name] = 1
            folder_name = f"{seg_name}_{segment_count[seg_name]}"
        else:
            segment_count[seg_name] += 1
            folder_name = f"{seg_name}_{segment_count[seg_name]}"

        dir_segment = os.path.join(dir, "Segments", folder_name)
        os.makedirs(dir_segment, exist_ok=True)
        output_pickle_file = os.path.join(dir_segment, f"{base_name}_kinematic_data_filtered.pkl")
        output_video = os.path.join(dir_segment, f"{base_name}_annotated_filtered.mp4")
        output_csv = os.path.join(dir_segment, f"{base_name}_markers_filtered.csv")

        print(f"\n📂 Segment: {folder_name}")
        print("-"*100)

        print()

        update_logfile(
            logfile=os.path.join(dir_segment, f"{base_name}_logfile.txt"),
            routine_name="main.py",
            patient_number=patient,
            session_number=session,
            camera_number=camera,
            original_video_name=video_name,
            segment=folder_name,
            start_time=format_time_for_log(seg_start),
            end_time=format_time_for_log(seg_end),
        )

        # Call filtering for the segment
        run_filter_detection(
            video_path=video_path,
            pickle_file=pickle_file,
            output_pkl=output_pickle_file,
            start_video_time=seg_start,
            end_video_time=seg_end,
            target_embedding=target_embedding
        )

        # Generate the annotated video for the filtered segment
        generate_video(
            pkl_file=output_pickle_file,
            video_source=video_path,
            output_video=output_video,
            show_only_targets=True,
            start_video_time=seg_start,
            end_video_time=seg_end
        )
        
        generate_csv(
            video_path=video_path,
            pkl_file=output_pickle_file,
            output_csv=output_csv,
            start_video_time=seg_start
        )

    # End the timer after processing the video
    end_time = time.time()

    print()  # To move to the next line after the progress
    print(f"Execution time for the segment: {end_time - start_time:.2f} seconds", flush=True)
    print()

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
