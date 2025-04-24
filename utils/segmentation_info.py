import re
from datetime import datetime

def parse_segmentation_info(txt_path):
    # Patterns for parsing
    id_pattern = re.compile(r"ID:(\d+(?:\.\d+)?)")
    time_pattern = re.compile(r"(\d{2}:\d{2}:\d{2}\.\d{3})")
    segment_pattern = re.compile(r"([^\t]+)\t+\t(\d{2}:\d{2}:\d{2}\.\d{3})\t(\d{2}:\d{2}:\d{2}\.\d{3})")

    time_format = "%H:%M:%S.%f"
    embedding_info = []
    segments = []

    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    # Parse segments
    for line in lines:
        if line.startswith("Session Duration"):
            continue  # Skip session duration lines
        if line.startswith("ID:"):
            # Parse embedding info
            match_id = id_pattern.search(line)
            match_times = time_pattern.findall(line)
            if not match_id or len(match_times) < 2:
                continue
            id_val = int(float(match_id.group(1)))
            start_time = datetime.strptime(match_times[0], time_format)
            end_time = datetime.strptime(match_times[1], time_format)
            start_sec = start_time.hour * 3600 + start_time.minute * 60 + start_time.second + start_time.microsecond / 1e6
            end_sec = end_time.hour * 3600 + end_time.minute * 60 + end_time.second + end_time.microsecond / 1e6
            embedding_info.append([start_sec, end_sec, id_val])
        else:
            match = segment_pattern.match(line)
            if match:
                seg_name = match.group(1).strip()
                start_time = datetime.strptime(match.group(2), time_format)
                end_time = datetime.strptime(match.group(3), time_format)
                start_sec = start_time.hour * 3600 + start_time.minute * 60 + start_time.second + start_time.microsecond / 1e6
                end_sec = end_time.hour * 3600 + end_time.minute * 60 + end_time.second + end_time.microsecond / 1e6
                segments.append((seg_name, start_sec, end_sec))
    return embedding_info, segments