"""
==============================================================================
Title:          Configuration Settings
Description:    Defines the root directory and selected patients, sessions, 
                target subfolders, camera and segments for the VRRehab_UQ-MyTurn 
                project.
Author:         Lucas R. L. Cardoso
Project:        VRRehab_UQ-MyTurn
Date:           2025-03-25
Version:        1.3

Changelog:
    - v1.0: [2025-03-25] Initial release
    - v1.1: [2025-04-15] Added the variable FILTER_CONFIG
    - v1.2: [2025-04-24] Removed the variable FILTER_CONFIG that is now 
                         obtained from the segmentation txt file. 
    - v1.3: [2025-05-06] Added the variable SELECTED_SEGMENTS to allow filtering 
                         and processing of specific segments by index.
==============================================================================
"""

ROOT_DIR = r"C:\Users\s4659771\Documents\MyTurn_Project\Data"
SELECTED_PATIENTS = ["P04"]
SELECTED_SESSIONS = ["Session3"]
TARGET_SUBFOLDERS = ["FMA_and_VR", "VR", "CT"]
SELECTED_CAMERAS = ["Camera1"]
SELECTED_SEGMENTS = [5]