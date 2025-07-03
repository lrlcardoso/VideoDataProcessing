"""
==============================================================================
Title:          Configuration Settings
Description:    Defines the root directory and selected patients, sessions, 
                target subfolders, camera and segments for the VRRehab_UQ-MyTurn 
                project.
Author:         Lucas R. L. Cardoso
Project:        VRRehab_UQ-MyTurn
Date:           2025-03-25
Version:        1.4

Changelog:
    - v1.0: [2025-03-25] Initial release
    - v1.1: [2025-04-15] Added the variable FILTER_CONFIG
    - v1.2: [2025-04-24] Removed the variable FILTER_CONFIG that is now 
                         obtained from the segmentation txt file. 
    - v1.3: [2025-05-06] Added the variable SELECTED_SEGMENTS to allow filtering 
                         and processing of specific segments by index.
    - v1.4: [2025-06-19] Added the variables TARGET_ID and IGNORE_IDS to allow 
                         for manual adjustments of the filtering process.
==============================================================================
"""

ROOT_DIR = r"C:\Users\s4659771\Documents\MyTurn_Project\Data"
SELECTED_PATIENTS = ["P02"]
SELECTED_SESSIONS = ["Session1"]
TARGET_SUBFOLDERS = ["FMA_and_VR", "VR", "CT"]
SELECTED_CAMERAS = ["Camera1","Camera2"]
SELECTED_SEGMENTS = [0,1,2,3,4,5,6,7,8,9,10,11,12]
TARGET_ID = None
IGNORE_IDS = None


# python main.py --filter_detection