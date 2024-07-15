import os

from models.train_model import training_model
from utils.constants import KERAS_DATA_PATH, MODEL_NUMS


def train():
    for model_num in MODEL_NUMS:
        model_name_path = os.path.join(KERAS_DATA_PATH, f"actions_{model_num}.keras")
        training_model(model_name_path, model_num)
    return True
