import numpy as np
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

from models.model import get_model
from utils.constants import KEYPOINTS_DATA_PATH
from utils.model_utils import get_sequences_and_labels, get_word_ids


def training_model(model_path, model_num: int, epochs=70):
    word_ids = get_word_ids(KEYPOINTS_DATA_PATH)  # ['word1', 'word2', 'word3]

    sequences, labels = get_sequences_and_labels(word_ids, model_num)

    sequences = pad_sequences(
        sequences,
        maxlen=int(model_num),
        padding="pre",
        truncating="post",
        dtype="float32",
    )

    X = np.array(sequences)
    y = to_categorical(labels).astype(int)

    model = get_model(int(model_num), len(word_ids))
    model.fit(X, y, epochs=epochs)
    model.summary()
    model.save(model_path)
