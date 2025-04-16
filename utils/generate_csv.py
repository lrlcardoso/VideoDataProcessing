import csv
import datetime
import ffmpeg
import pickle
import numpy as np


def create_csv_for_person(track_id, csv_writers):
    """
    Creates a new CSV file for a given person (track_id).
    
    Parameters:
    - track_id (int): The ID of the tracked person.
    - csv_writers (dict): Dictionary to store CSV file handles and writers.
    
    Returns:
    - None
    """
    filename = f"person_{track_id}.csv"
    
    # Create headers (17 markers, each with x, y, confidence)
    headers = ["System Time", "Time"] + [f"{i}_{axis}" for i in range(17) for axis in ["x", "y", "conf"]]
    
    # Open the file and create a CSV writer
    file = open(filename, mode="w", newline="")
    writer = csv.writer(file)
    writer.writerow(headers)
    
    # Store the writer and file handle
    csv_writers[track_id] = (file, writer)
    print(f"Created new CSV file: {filename}")


def process_tracking_data(pickle_file, video_file):
    """
    Processes tracking data from a pickle file and saves it as CSV files.
    
    Parameters:
    - pickle_file (str): Path to the pickle file containing YOLO tracking data.
    - video_file (str): Path to the original video file to extract FPS.
    
    Returns:
    - None
    """
    # Load YOLO results from pickle file
    with open(pickle_file, "rb") as f:
        results = pickle.load(f)

    # Get video metadata (to extract FPS)
    probe = ffmpeg.probe(video_file)
    fps = eval(probe['streams'][0]['r_frame_rate'])

    # Dictionary to store CSV writers
    csv_writers = {}

    # Process frames
    for frame_idx, frame_data in enumerate(results):
        system_time = f'{datetime.datetime.now().strftime("%y/%m/%d_%H:%M:%S.%f")[:-3]}'
        time_value = frame_idx * (1 / fps)

        # Skip frames with no valid data
        if frame_data["boxes"] is None or frame_data["ids"] is None or frame_data["keypoints"] is None:
            continue

        boxes = frame_data["boxes"]
        ids = frame_data["ids"]
        keypoints = frame_data["keypoints"]

        for person_idx in range(len(ids)):
            track_id = int(ids[person_idx])  # Extract track_id

            # Create a new CSV file for new track IDs
            if track_id not in csv_writers:
                create_csv_for_person(track_id, csv_writers)

            # Extract keypoints for the current person
            a = []
            for marker in range(17):
                if keypoints.shape[1] > marker:  # Ensure marker index is within bounds
                    x, y = keypoints[person_idx, marker]
                    conf = frame_data["conf"][person_idx] if frame_data["conf"] is not None else 0.0  # Extract confidence
                else:
                    x, y, conf = np.nan, np.nan, 0.0  # Use NaN for missing keypoints
                a.extend([x, y, conf])  # Append all three values

            # Write row to corresponding CSV file
            _, writer = csv_writers[track_id]
            writer.writerow([system_time, time_value] + a)

    # Close all open files
    for file, _ in csv_writers.values():
        file.close()

    print("CSV files generated for all tracked persons.")


def generate_csv(pickle_file, video_file):
    
    process_tracking_data(pickle_file, video_file)


if __name__ == "__main__":

    pickle_file = "sample_video_25fps.pkl"
    video_file = "sample_video_25fps.mp4"
    
    generate_csv(pickle_file, video_file)
