import re
from datetime import datetime

def parse_embedding_info(txt_path):
    id_pattern = re.compile(r"ID:(\d+(?:\.\d+)?)")
    time_pattern = re.compile(r"(\d{2}:\d{2}:\d{2}\.\d{3})")  # Matches HH:MM:SS.mmm
    time_format = "%H:%M:%S.%f"

    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        if line.strip().startswith("ID:"):
            match_id = id_pattern.search(line)
            match_times = time_pattern.findall(line)

            if not match_id or len(match_times) < 2:
                raise ValueError("Failed to parse ID or time from line: " + line)

            id_val = int(float(match_id.group(1)))
            start_time_str, end_time_str = match_times[:2]

            start_time = datetime.strptime(start_time_str, time_format)
            end_time = datetime.strptime(end_time_str, time_format)

            start_sec = start_time.minute * 60 + start_time.second + start_time.microsecond / 1e6 + start_time.hour * 3600
            end_sec = end_time.minute * 60 + end_time.second + end_time.microsecond / 1e6 + end_time.hour * 3600

            return [[start_sec, end_sec, id_val]]

    raise ValueError("No valid 'ID:' line found in the file.")
