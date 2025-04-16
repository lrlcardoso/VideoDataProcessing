"""
==============================================================================
Title:          Annotate Video with Bounding Boxes and Skeletons
Description:    Processes a video by overlaying bounding boxes, person IDs, 
                and skeletons on the frames using data from a pickle file. 
                It also compresses the output video with NVIDIA GPU 
                acceleration using FFmpeg and provides progress updates.
Author:         Lucas R. L. Cardoso
Project:        VRRehab_UQ-MyTurn
Date:           2025-03-25
Version:        1.1
==============================================================================
Usage:
    python <script_name>.py
    
Dependencies:
    - Python >= 3.x
    - Required libraries: cv2, subprocess, pickle, time

Changelog:
    - v1.0: [2025-03-25] Initial release
    - v1.1: [2025-04-15] Added support for trimming the video using start and 
    stop seconds
==============================================================================
"""

import cv2
import subprocess
import pickle
import time

# Method to draw the bounding box and person ID on the frame
def draw_bounding_box_with_id(frame, box, person_id, box_color):
    """
    Draws a bounding box and person ID on the given frame.

    Parameters:
    - frame: The image frame where the bounding box and ID will be drawn.
    - box: The coordinates of the bounding box (x1, y1, x2, y2).
    - person_id: The ID of the person associated with the bounding box.
    - box_color: The color to draw the bounding box and text.

    Returns:
    - None
    """
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)  # Draw the bounding box
    
    # Add the person ID above the bounding box
    cv2.putText(frame, f"ID: {person_id}", 
                (x1, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 
                1.0, box_color, 2)

# Method to draw the skeleton for each person in the frame
def draw_skeleton(frame, person, keypoints_color):
    """
    Draws the skeleton for a person in the given frame.

    Parameters:
    - frame: The image frame where the skeleton will be drawn.
    - person: List of keypoints representing the person's body parts.
    - keypoints_color: The color to draw the skeleton.

    Returns:
    - None
    """
    connections = [
        (0, 2),  # Right Eye ↔ Nose
        (0, 1),  # Left Eye ↔ Nose
        (2, 4),  # Right Eye ↔ Right Ear
        (1, 3),  # Left Eye ↔ Left Ear
        (5, 6),  # Right Shoulder ↔ Left Shoulder
        (6, 12), # Right Shoulder ↔ Right Hip
        (6, 8),  # Right Shoulder ↔ Right Elbow
        (8, 10), # Right Elbow ↔ Right Wrist
        (5, 11), # Left Shoulder ↔ Left Hip
        (5, 7),  # Left Shoulder ↔ Left Elbow
        (7, 9),  # Left Elbow ↔ Left Wrist
        (12, 11), # Right Hip ↔ Left Hip
        (12, 14), # Right Hip ↔ Right Knee
        (14, 16), # Right Knee ↔ Right Ankle
        (11, 13), # Left Hip ↔ Left Knee
        (13, 15), # Left Knee ↔ Left Ankle
    ]
     
    keypoint_positions = []  # Initialize list to hold valid keypoint positions

    # Extract valid keypoints
    for i, keypoint in enumerate(person):
        if keypoint is not None and len(keypoint) == 2:  # Check if the keypoint is valid
            x, y = map(int, keypoint)
            keypoint_positions.append((x, y))

    # Draw lines between connected keypoints if the positions are valid
    for connection in connections:
        if connection[0] < len(keypoint_positions) and connection[1] < len(keypoint_positions):
            keypoint1 = keypoint_positions[connection[0]]
            keypoint2 = keypoint_positions[connection[1]]
            if keypoint1 != (0, 0) and keypoint2 != (0, 0):  # Avoid drawing lines between invalid keypoints
                cv2.line(frame, keypoint1, keypoint2, keypoints_color, 2)

    # Draw keypoints as circles if the positions are valid
    for keypoint in keypoint_positions:
        if keypoint != (0, 0):  # Avoid drawing circles on invalid keypoints
            cv2.circle(frame, keypoint, 6, keypoints_color, -1)

# Add this line at the top of your script (or somewhere globally)
last_valid_search_box = [None]  # Wrapped in list for mutability

def process_frame(frame, data, show_only_targets):
    """
    Processes a single frame to draw bounding boxes, IDs, and skeletons.
    Handles persistent search box for target mode when camera is not moving.
    """
    boxes = data.get("boxes", [])
    ids = data.get("ids", [])
    keypoints = data.get("keypoints", [])

    raw_flags = data.get("is_target", None)
    if raw_flags is not None and len(raw_flags) == len(boxes):
        flags = raw_flags
    else:
        flags = [True] * len(boxes) if not show_only_targets else [False] * len(boxes)

    mids = data.get("mid_points", [None] * len(boxes))
    avg_mid = data.get("avg_mid", None)
    similarities = data.get("similarities", [])
    search_box = data.get("search_box", None)

    # ✅ Update or keep persistent search box
    if show_only_targets:
        if search_box:
            last_valid_search_box[0] = search_box  # Update if valid
        else:
            search_box = last_valid_search_box[0]  # Use previous if missing

    # ✅ Show all midpoints and similarities
    for i, mid in enumerate(mids):
        if mid is not None:
            x, y = map(int, mid)
            cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)
            cv2.putText(frame, "Mid", (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            if i < len(similarities):
                sim_score = similarities[i]
                cv2.putText(frame, f"Sim: {sim_score:.2f}", (x + 5, y + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # ✅ Draw bounding boxes and skeletons
    for i, box in enumerate(boxes):
        if not show_only_targets or flags[i]:
            draw_bounding_box_with_id(frame, box, ids[i] if i < len(ids) else "?", (0, 255, 0))
            if i < len(keypoints):
                draw_skeleton(frame, keypoints[i], (0, 0, 255))

    # ✅ Draw avg_mid and persistent search_box in target-only mode
    if show_only_targets:
        if avg_mid is not None:
            x, y = map(int, avg_mid)
            cv2.circle(frame, (x, y), 7, (255, 0, 0), -1)
            cv2.putText(frame, "AvgMid", (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        if search_box:
            top_left, bottom_right = search_box
            cv2.rectangle(frame, tuple(top_left), tuple(bottom_right), (255, 255, 0), 2)
            cv2.putText(frame, "SearchBox", (top_left[0], top_left[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

# Main function to process video and overlay bounding boxes and skeletons
def process_video(pkl_file, video_path, output_video_path, start_video_time, end_video_time, show_only_targets=False):
    """
    Processes a video by overlaying bounding boxes and skeletons from a pickle file.

    Parameters:
    - pkl_file: Path to the pickle file containing extracted data (boxes, IDs, keypoints).
    - video_path: Path to the input video file.
    - output_video_path: Path to save the annotated output video.

    Returns:
    - None
    """
    # Load the extracted data from the pickle file
    with open(pkl_file, "rb") as f:
        extracted_data = pickle.load(f)

    # Open the video file using OpenCV
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # Get frames per second from the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps

    # Set up the FFmpeg process for H.264 compression
    command = [
        'ffmpeg',
        '-y',  # Overwrite the output file without asking
        '-f', 'rawvideo',  # Input format
        '-vcodec', 'rawvideo',  # Input codec
        '-pix_fmt', 'bgr24',  # Input pixel format
        '-s', f'{int(cap.get(3))}x{int(cap.get(4))}',  # Resolution (width x height)
        '-r', str(fps),  # Frame rate
        '-i', '-',  # Input comes from stdin
        '-c:v', 'libx264',  # Use H.264 codec for output
        '-pix_fmt', 'yuv420p',  # Convert output to yuv420p for better compatibility
        '-crf', '23',  # Set quality/compression (lower is better quality)
        '-preset', 'medium',  # Compression speed (medium balances quality and speed)
        '-movflags', '+faststart',  # Enable streaming optimization
        '-profile:v', 'main',  # Use main profile for better compatibility
        '-level:v', '3.1',  # Set H.264 level (compatible with most players)
        output_video_path  # Path to save the output video
    ]

    # Start a timer to measure execution time
    start_time = time.time()

    # Open the FFmpeg process to write video frames
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    # Use full video if times are not provided
    if start_video_time is None:
        start_video_time = 0
    if end_video_time is None:
        end_video_time = video_duration

    start_frame, end_frame = int(start_video_time * fps), int(end_video_time * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx = 0

    while cap.isOpened():
        current_frame = start_frame + frame_idx
        if current_frame >= end_frame or frame_idx >= len(extracted_data):
            break

        ret, frame = cap.read()
        if not ret:
            break

        data = extracted_data[frame_idx]
        # data = extracted_data[current_frame]
        process_frame(frame, data, show_only_targets)
        process.stdin.write(frame.tobytes())

        print(f"\rProcessing frame {current_frame + 1}/{end_frame}", end="", flush=True)
        frame_idx += 1

    # Release the video capture and finish processing
    cap.release()
    process.stdin.close()
    process.wait()

    # End the timer after processing the video
    end_time = time.time()

    # Print the total execution time for processing the video
    print()  # To move to the next line after the progress
    print(f"Execution time: {end_time - start_time:.2f} seconds", flush=True)


def generate_video(pkl_file, video_source, output_video, start_video_time, end_video_time, show_only_targets):

    print()
    print("-> Generating Annotated Video...")
    print()

    process_video(pkl_file, video_source, output_video, start_video_time, end_video_time, show_only_targets)

if __name__ == "__main__":
    pkl_file = "VRsession_30fps_largerModel.pkl"
    video_source = "VRsession_30fps.mp4"
    output_video = "VRsession_30fps_largerModel_annotated.mp4"
    show_only_targets = False

    generate_video(pkl_file, video_source, output_video, show_only_targets)
