import os

import cv2

# PATHS
ROOT_PATH = os.getcwd()
DATA_PATH = os.path.join(ROOT_PATH, "data")
RAW_DATA_PATH = os.path.join(DATA_PATH, "raw")
PROCESSED_DATA_PATH = os.path.join(DATA_PATH, "processed")
FRAME_ACTIONS_PATH = os.path.join(RAW_DATA_PATH, "frame_actions")
MODELS_PATH = os.path.join(ROOT_PATH, "models")

MAX_LENGTH_FRAMES = 15
LENGTH_KEYPOINTS = 1662
MIN_LENGTH_FRAMES = 5
MODEL_NAME = f"actions_{MAX_LENGTH_FRAMES}.keras"

# SHOW IMAGE PARAMETERS
FONT = cv2.FONT_HERSHEY_PLAIN
FONT_SIZE = 1.5
FONT_POS = (10, 45)
RED_COLOR = (0, 0, 255)
BLUE_COLOR = (255, 0, 0)
