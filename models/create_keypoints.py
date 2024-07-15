import os

import pandas as pd
from mediapipe.python.solutions.holistic import Holistic

from utils.constants import KEYPOINTS_DATA_PATH, MIN_LENGTH_FRAMES
from utils.model_utils import get_keypoints, insert_keypoints_sequence


def create_keypoints(word_path):
    data = pd.DataFrame([])
    name_word = os.path.basename(word_path)
    save_path = os.path.join(KEYPOINTS_DATA_PATH, f"{name_word}.h5")
    if f"{name_word}.h5" in os.listdir(KEYPOINTS_DATA_PATH):
        return False
    with Holistic() as model_holistic:
        print(f'Creando keypoints de "{name_word}"...')
        sample_list = os.listdir(word_path)
        sample_count = len(sample_list)
        n_sample = 1

        for sample_name in sample_list:
            sample_path = os.path.join(word_path, sample_name)
            frames_count = len(os.listdir(sample_path))

            model_num = (
                "7"
                if MIN_LENGTH_FRAMES <= frames_count <= 7
                else "12" if frames_count <= 12 else "18"
            )

            keypoints_sequence = get_keypoints(model_holistic, sample_path)
            data = insert_keypoints_sequence(
                data, n_sample, model_num, keypoints_sequence
            )

            print(f"{n_sample}/{sample_count}", end="\r")
            n_sample += 1

    data.to_hdf(save_path, key="data", mode="w")
    return True
