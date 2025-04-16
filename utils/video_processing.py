"""
==============================================================================
Title:          Change Video FPS and Resolution
Description:    Changes the resolution and frame rate of a video using FFmpeg
                with NVIDIA GPU acceleration and provides status updates.
Author:         Lucas R. L. Cardoso
Project:        VRRehab_UQ-MyTurn
Date:           2025-03-25
Version:        1.0
==============================================================================
Usage (as standalone):
    python -m utils.video_processing
    
Dependencies:
    - Python >= 3.x
    - Required libraries: subprocess, time, ffmpeg

Changelog:
    - v1.0: [2025-03-25] Initial release
==============================================================================
"""

import os
import subprocess
import time
from utils.logfile_utils import update_logfile

def get_video_metadata(video_file):
    """
    Extracts the FPS and resolution from a video file using FFmpeg's ffprobe.

    Parameters:
    - video_file: str, path to the video file.

    Returns:
    - fps: float, frames per second of the video.
    - resolution: str, resolution in 'WxH' format.
    """
    try:
        # FFprobe command to extract frame rate and resolution
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate,width,height",
            "-of", "csv=p=0",
            video_file
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        width, height, frame_rate = result.stdout.strip().split(',')

        # Convert frame rate to float
        num, denom = map(int, frame_rate.split('/'))
        fps = num / denom if denom != 0 else float(num)

        return fps, f"{width}x{height}"

    except Exception as e:
        print(f"Error extracting metadata: {e}")
        return None, None

def change_video_fps_and_res(video_source, output_file, width, fps):
    """
    Changes the resolution and frame rate of a video using FFmpeg with NVIDIA GPU acceleration.
    Displays the status line during processing.

    Parameters:
    - video_source: str, path to the input video file to be processed.
    - output_file: str, path to the saved output video file.
    - width: int, the desired width of the output video (height will be adjusted to maintain aspect ratio).
    - fps: int, the desired frames per second for the output video.

    Returns:
    - output_file: str, path to the saved output video file.
    """
    print("\n-> Changing FPS/Resolution...\n")

    # FFmpeg command with GPU acceleration for video processing
    command = [
        "ffmpeg", "-hwaccel", "cuda",  # Enable hardware acceleration
        "-i", video_source,
        "-vf", f"scale={width}:-1",  # Resize while maintaining aspect ratio
        "-r", str(fps),              # Set frame rate
        "-c:v", "h264_nvenc",        # Use NVIDIA encoder (NVENC)
        "-preset", "p1",             # Fastest preset (p1 = fastest, p7 = slowest/best quality)
        "-b:v", "5M",                # Set video bitrate (adjust as needed)
        "-c:a", "aac",               # Use AAC for audio
        "-b:a", "128k",              # Set audio bitrate
        "-y", output_file,           # Overwrite existing file
        "-hide_banner",              # Hide unnecessary FFmpeg banner
        "-nostdin"                   # Prevent interactive input issues
    ]

    # Start a timer to measure execution time
    start_time = time.time()

    # Execute the FFmpeg command as a subprocess
    process = subprocess.Popen(command, stderr=subprocess.PIPE, universal_newlines=True)
    
    # Monitor and print the FFmpeg status lines (e.g., frame, fps, time)
    for line in process.stderr:
        if "frame=" in line or "fps=" in line or "time=" in line:
            print("\r" + line.strip(), end="", flush=True)  # Print only the status line

    process.wait()

    # End the timer after processing the video
    end_time = time.time()

    # Print the total execution time for processing the video
    print("\nExecution time: {:.2f} seconds".format(end_time - start_time), flush=True)

    # Extract metadata from the output video
    extracted_fps, extracted_resolution = get_video_metadata(output_file)

    if extracted_fps is not None and extracted_resolution is not None:
        log_file = os.path.splitext(output_file)[0] + "_logfile.txt"

        # Call the update_logfile function
        update_logfile(
            logfile=log_file,
            routine_name="video_processing.py",
            fps=extracted_fps,
            resolution=extracted_resolution
        )

if __name__ == "__main__":
    # Example usage of the change_video_fps_and_res function
    video_source = "VRsession_60fps.mkv"  # Path to the input video
    output_file = "VRsession_30fps3.mp4"  # Path for the output video
    desired_width = 1920                  # Desired video width
    desired_fps = 30                      # Desired frames per second

    # Change the video's FPS and resolution
    change_video_fps_and_res(video_source, output_file, desired_width, desired_fps)