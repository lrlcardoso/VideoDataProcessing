"""
==============================================================================
Title:          Configuration Settings
Description:    Defines the root directory and selected patients, sessions, 
                and target subfolders for the VRRehab_UQ-MyTurn project.
Author:         Lucas R. L. Cardoso
Project:        VRRehab_UQ-MyTurn
Date:           2025-03-25
Version:        1.1

Changelog:
    - v1.0: [2025-03-25] Initial release
    - v1.1: [2025-04-15] Added the variable FILTER_CONFIG
==============================================================================
"""

ROOT_DIR = r"C:\Users\s4659771\Documents\MyTurn_Project\Data"
# SELECTED_PATIENTS = ["P01"]
# SELECTED_SESSIONS = ["Session3"]
# TARGET_SUBFOLDERS = ["FMA_and_VR", "VR", "CT"]
# SELECTED_CAMERAS = ["Camera1"]


# # Example 1
# SELECTED_PATIENTS = ["P01"]
# SELECTED_SESSIONS = ["Session3"]
# TARGET_SUBFOLDERS = ["VR"]
# SELECTED_CAMERAS = ["Camera1"]
# FILTER_CONFIG = {
#     "start_video_time": 38*60,
#     "end_video_time": 41*60,
#     "embedding_info": [[120, 180, 12]]
# }

# Example 2
SELECTED_PATIENTS = ["P02"]
SELECTED_SESSIONS = ["Session3"]
TARGET_SUBFOLDERS = ["VR"]
SELECTED_CAMERAS = ["Camera1"]
FILTER_CONFIG = {
    "start_video_time": 23*60, #0
    "end_video_time": 23*60 + 15, #15
    "embedding_info": [[20, 40, 2]] #[[31*60 + 10, 32*60 + 10, 152]] #
}

# # Example 3
# SELECTED_PATIENTS = ["P03"]
# SELECTED_SESSIONS = ["Session3"]
# TARGET_SUBFOLDERS = ["VR"]
# SELECTED_CAMERAS = ["Camera1"]
# FILTER_CONFIG = {
#     "start_video_time": 23*60 + 30.2,
#     "end_video_time": 23*60 + 50,
#     "embedding_info": [[0, 20, 2]]
# }