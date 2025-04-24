"""
==============================================================================
Title:          Segmentation Info Parser
Description:    Parses a segmentation text file to extract embedding intervals
                and segment information, including start/end times and IDs.
                Designed for integration in data processing pipelines but can
                also be run standalone for direct inspection of parsed data.
Author:         Lucas R. L. Cardoso
Project:        VRRehab_UQ-MyTurn
Date:           2025-04-24
Version:        1.0
==============================================================================
Usage:
    python parse_segmentation_info.py --txt_path path/to/segmentation.txt

Dependencies:
    - Python >= 3.x
    - Required libraries: re, datetime

Changelog:
    - v1.0: [2025-04-24] Initial release
==============================================================================
"""

import re
import argparse
from datetime import datetime

def parse_segmentation_info(txt_path):
    """
    Parses a segmentation info text file to extract embedding intervals (ID, start, end)
    and general segment intervals (name, start, end).
    
    Parameters:
        txt_path (str): Path to the segmentation info text file.
    
    Returns:
        embedding_info (list): List of [start_sec, end_sec, id_val] for embedding intervals.
        segments (list): List of (segment_name, start_sec, end_sec) for each segment.
    """
    # Regex patterns for parsing IDs, times, and segments
    id_pattern = re.compile(r"ID:(\d+(?:\.\d+)?)")
    time_pattern = re.compile(r"(\d{2}:\d{2}:\d{2}\.\d{3})")
    segment_pattern = re.compile(r"([^\t]+)\t+\t(\d{2}:\d{2}:\d{2}\.\d{3})\t(\d{2}:\d{2}:\d{2}\.\d{3})")

    time_format = "%H:%M:%S.%f"
    embedding_info = []
    segments = []

    # Read and clean non-empty lines from the file
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    for line in lines:
        if line.startswith("Session Duration"):
            continue  # Skip any session duration lines
        if line.startswith("ID:"):
            # Parse embedding info lines: expect ID and two timestamps
            match_id = id_pattern.search(line)
            match_times = time_pattern.findall(line)
            if not match_id or len(match_times) < 2:
                continue  # Skip invalid lines
            id_val = int(float(match_id.group(1)))
            start_time = datetime.strptime(match_times[0], time_format)
            end_time = datetime.strptime(match_times[1], time_format)
            start_sec = start_time.hour * 3600 + start_time.minute * 60 + start_time.second + start_time.microsecond / 1e6
            end_sec = end_time.hour * 3600 + end_time.minute * 60 + end_time.second + end_time.microsecond / 1e6
            embedding_info.append([start_sec, end_sec, id_val])
        else:
            # Parse segment annotation lines: expect name and two timestamps (tab-separated)
            match = segment_pattern.match(line)
            if match:
                seg_name = match.group(1).strip()
                start_time = datetime.strptime(match.group(2), time_format)
                end_time = datetime.strptime(match.group(3), time_format)
                start_sec = start_time.hour * 3600 + start_time.minute * 60 + start_time.second + start_time.microsecond / 1e6
                end_sec = end_time.hour * 3600 + end_time.minute * 60 + end_time.second + end_time.microsecond / 1e6
                segments.append((seg_name, start_sec, end_sec))
    return embedding_info, segments

if __name__ == "__main__":
    # CLI for standalone usage
    parser = argparse.ArgumentParser(description="Parse segmentation info from a text file.")
    parser.add_argument('--txt_path', type=str, required=True, help='Path to the segmentation info .txt file')
    args = parser.parse_args()

    embedding_info, segments = parse_segmentation_info(args.txt_path)
    
    print("=== Embedding Intervals (start_sec, end_sec, id) ===")
    for emb in embedding_info:
        print(emb)
    print("\n=== Segments (name, start_sec, end_sec) ===")
    for seg in segments:
        print(seg)