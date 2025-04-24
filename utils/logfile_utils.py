"""
==============================================================================
Title:          Logfile Updater
Description:    Logs or updates structured entries containing video processing 
                parameters, including runtime information, session context, and 
                inference and Re-ID configuration details.
Author:         Lucas R. L. Cardoso
Project:        VRRehab_UQ-MyTurn
Date:           2025-04-24
Version:        1.2
==============================================================================
Usage:
    Called as a utility function from other scripts (not run directly).

Dependencies:
    - Python >= 3.x
    - Required libraries: os, time, socket

Changelog:
    - v1.0: [2025-03-25] Initial release
    - v1.1: [2025-04-15] Added support for Re-ID parameters (HISTORY_WINDOW, 
            SHOULDER_WIDTH_SCALE, VERTICAL_SCALE_FACTOR, MIN_BOX_WIDTH, 
            MAX_BOX_WIDTH, THRESHOLD_CAMERA_MOV)
    - v1.2: [2025-04-24] Added support for segment information, start/end time, 
            and smoothing parameters (SMOOTH_WINDOW, MIN_MOVEMENT_DURATION, 
            MIN_STATIC_DURATION)
==============================================================================
"""

import os
import socket
from datetime import datetime
from zoneinfo import ZoneInfo

def update_logfile(
    logfile,
    routine_name,
    patient_number=None,
    session_number=None,
    camera_number=None,
    original_video_name=None,
    fps=None,
    resolution=None,
    yolo_model_name=None,
    device=None,
    classes=None,
    batch_size=None,
    stream_flag=None,
    precision_flag=None,
    iou=None,
    max_detections=None,
    time_to_process=None,
    history_window=None,
    shoulder_width_scale=None,
    vertical_scale_factor=None,
    min_box_width=None,
    max_box_width=None,
    threshold_camera_mov=None,
    # New fields for 1.2
    segment=None,
    start_time=None,
    end_time=None,
    smooth_window=None,
    min_movement_duration=None,
    min_static_duration=None,
):
    """
    Appends a structured log entry for a processing routine, capturing session,
    video, and parameter information for reproducibility and tracking.
    """

    pc_name = socket.gethostname()

    # Ensure the log file exists
    if not os.path.exists(logfile):
        with open(logfile, 'w') as file:
            file.write(f"""==============================================================================
                            Video Data Processing Log
==============================================================================
Description:    Logs the details of video processing, kinematic data extraction, 
                and model parameters for each session and patient.
Author:         Lucas R. L. Cardoso
Project:        VRRehab_UQ-MyTurn
Version:        1.0
PC:             {pc_name}
==============================================================================

LOG ENTRIES:
==============================================================================
""")

    # Get the current timestamp
    timestamp = datetime.now(ZoneInfo("Australia/Brisbane")).strftime("%Y-%m-%d %H:%M:%S")

    # Prepare log entry
    log_entry = f"\n{timestamp} - [{routine_name}]:\n"

    # General video info
    if patient_number is not None:
        log_entry += f"  - Patient Number: {patient_number}\n"
    if session_number is not None:
        log_entry += f"  - Session Number: {session_number}\n"
    if camera_number is not None:
        log_entry += f"  - Camera Number: {camera_number}\n"
    if original_video_name is not None:
        log_entry += f"  - Original Video Name: {original_video_name}\n"
    if segment is not None:
        log_entry += f"  - Segment: {segment}\n"
    if start_time is not None:
        log_entry += f"  - Start Time: {start_time}\n"
    if end_time is not None:
        log_entry += f"  - End Time: {end_time}\n"
    if fps is not None:
        log_entry += f"  - FPS Used: {fps}\n"
    if resolution is not None:
        log_entry += f"  - Resolution Used: {resolution}\n"

    # Model / processing configuration
    if yolo_model_name is not None:
        log_entry += f"  - YOLO Model Used: {yolo_model_name}\n"
    if device is not None:
        log_entry += f"  - Device: {device}\n"
    if classes is not None:
        log_entry += f"  - Classes: {classes}\n"
    if batch_size is not None:
        log_entry += f"  - Batch Size: {batch_size}\n"
    if stream_flag is not None:
        log_entry += f"  - Stream: {stream_flag}\n"
    if precision_flag is not None:
        log_entry += f"  - Half Precision: {precision_flag}\n"
    if iou is not None:
        log_entry += f"  - IOU Threshold: {iou}\n"
    if max_detections is not None:
        log_entry += f"  - Max Detections: {max_detections}\n"
    if time_to_process is not None:
        log_entry += f"  - Time to Process: {time_to_process:.2f} seconds\n"

    # Re-ID / Filtering parameters
    if history_window is not None:
        log_entry += f"  - HISTORY_WINDOW: {history_window}\n"
    if shoulder_width_scale is not None:
        log_entry += f"  - SHOULDER_WIDTH_SCALE: {shoulder_width_scale}\n"
    if vertical_scale_factor is not None:
        log_entry += f"  - VERTICAL_SCALE_FACTOR: {vertical_scale_factor}\n"
    if min_box_width is not None:
        log_entry += f"  - MIN_BOX_WIDTH: {min_box_width}\n"
    if max_box_width is not None:
        log_entry += f"  - MAX_BOX_WIDTH: {max_box_width}\n"
    if threshold_camera_mov is not None:
        log_entry += f"  - THRESHOLD_CAMERA_MOV: {threshold_camera_mov}\n"
    if smooth_window is not None:
        log_entry += f"  - SMOOTH_WINDOW: {smooth_window}\n"
    if min_movement_duration is not None:
        log_entry += f"  - MIN_MOVEMENT_DURATION: {min_movement_duration}\n"
    if min_static_duration is not None:
        log_entry += f"  - MIN_STATIC_DURATION: {min_static_duration}\n"
    
    # Append new log entry to file
    with open(logfile, 'a') as file:
        file.write(log_entry)
