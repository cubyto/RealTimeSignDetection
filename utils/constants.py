import os

import cv2

# SETTINGS
MIN_LENGTH_FRAMES = 5
LENGTH_KEYPOINTS = 1662
MODEL_NUMS = [7, 12, 18]

# PATHS
ROOT_PATH = os.getcwd()
DATA_PATH = os.path.join(ROOT_PATH, "data")
RAW_DATA_PATH = os.path.join(DATA_PATH, "raw")
FRAME_ACTIONS_PATH = os.path.join(RAW_DATA_PATH, "frame_actions")
PROCESSED_DATA_PATH = os.path.join(DATA_PATH, "processed")
KERAS_DATA_PATH = os.path.join(PROCESSED_DATA_PATH, "keras")
KEYPOINTS_DATA_PATH = os.path.join(PROCESSED_DATA_PATH, "keypoints")

# SHOW IMAGE PARAMETERS
FONT = cv2.FONT_HERSHEY_PLAIN
FONT_SIZE = 1.5
FONT_POS = (10, 45)
RED_COLOR = (0, 0, 255)
BLUE_COLOR = (255, 0, 0)
