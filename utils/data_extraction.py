"""
==============================================================================
Title:          Movement Data Extraction with YOLO
Description:    Processes a video file with the YOLO model to track movement and 
                save the results (bounding boxes, confidence scores, object IDs, 
                keypoints) to a pickle file.
Author:         Lucas R. L. Cardoso
Project:        VRRehab_UQ-MyTurn
Date:           2025-03-25
Version:        1.0
==============================================================================
Usage:
    python <script_name>.py
    
Dependencies:
    - Python >= 3.x
    - Required libraries: pickle, time, cv2, tqdm, ultralytics (YOLO)
    
Notes:
    - Ensure the path to the YOLO model is correct in `model_path`.
    - The `video_source` should point to the video to be processed.
    
Changelog:
    - v1.0: [2025-03-25] Initial release
==============================================================================
"""

import os
import pickle
import time
import cv2
from tqdm import tqdm
from ultralytics import YOLO
from utils.logfile_utils import update_logfile

def extract_movement_data(video_source, output_pickle):
    """
    Processes a video file with the YOLO model to track movement and save the results to a pickle file.

    This uses the YOLO model to track key points and objects in the video and saves the tracking data to a pickle file.

    Parameters:
    - video_source: str, path to the input video file.
    - output_pickle: str, path to save the results in pickle format.
    
    The output pickle file will contain movement data including bounding boxes, confidence scores, 
    object IDs, and keypoints for each frame in the video.
    """

    print()
    print("-> Extracting Movement Data...")
    print()

    # Load the YOLO model for pose estimation (extra large model)
    model_path = r"C:\Users\s4659771\Documents\MyTurn_Project\Video_Data_Processing2\pose_models\yolo11x-pose.pt"
    model = YOLO(model_path)

    # Start a timer to measure execution time
    start_time = time.time()

    # Initialize an empty list to store the tracking data
    results_cpu = []

    # Use OpenCV to get the total number of frames in the video
    cap = cv2.VideoCapture(video_source)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Extract total frame count
    cap.release()  # Close the video file

    # Initialize tqdm progress bar with the correct number of frames
    progress_bar = tqdm(total=total_frames, desc="Processing", unit="frame", dynamic_ncols=True)

    DEVICE="cuda"
    CLASSES=0
    BATCH=64 
    STREAM=True
    HALF=True
    IOU=0.2
    MAX_DET=3

    # Process the video frame by frame using the YOLO model for tracking
    for result in model.track(source=video_source, show=False, save=False, device=DEVICE, classes=CLASSES, batch=BATCH, 
                              stream=STREAM, half=HALF, iou=IOU, max_det=MAX_DET, verbose=False):
        
        # Extract the relevant tracking data from the result
        key_data = {
            "boxes": result.boxes.xyxy.cpu().numpy() if result.boxes is not None else None,  # Bounding box coordinates
            "conf": result.boxes.conf.cpu().numpy() if result.boxes is not None else None,  # Confidence scores
            "ids": result.boxes.id.cpu().numpy() if result.boxes.id is not None else None,  # Object IDs
            "keypoints": result.keypoints.xy.cpu().numpy() if result.keypoints is not None else None,  # Keypoints
        }
        results_cpu.append(key_data)  # Append the data for this frame to the results list

        # Update the progress bar
        progress_bar.update(1)

    # Close the progress bar after processing is complete
    progress_bar.close()

    # End the timer after processing the video
    end_time = time.time()

    # Print the total execution time for processing the video
    print(f"Execution time: {end_time - start_time:.2f} seconds")

    # Save the collected tracking data to the specified pickle file
    with open(output_pickle, "wb") as f:
        pickle.dump(results_cpu, f)

    # Call the update_logfile function
    log_file = os.path.splitext(video_source)[0] + "_logfile.txt"

    update_logfile(
        logfile=log_file,
        routine_name="data_extraction.py",
        yolo_model_name=os.path.splitext(os.path.basename(model_path))[0],
        device=DEVICE, 
        classes=CLASSES, 
        batch_size=BATCH, 
        stream_flag=STREAM, 
        precision_flag=HALF, 
        iou=IOU, 
        max_detections=MAX_DET,
        time_to_process = end_time - start_time
    )


# Allow the script to be run as a standalone program or imported as a module
if __name__ == "__main__":
    
    # Example usage of the extract_movement_data function
    video_source = "VRsession_30fps.mp4"  # Path to the video file to be processed
    output_pickle = "VRsession_30fps_largerModel2.pkl"  # Path to save the tracking results as a pickle file
    
    # Call the function to process the video and save the results
    extract_movement_data(video_source, output_pickle)
