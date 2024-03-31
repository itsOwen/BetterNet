import os
import json
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.metrics import Recall, Precision, MeanIoU
from tensorflow.keras.optimizers import Adam
from metrics import intersection_over_union, dice_coefficient, dice_loss, binary_crossentropy_dice_loss

def create_directory(directory_path):
    """ Create a directory. """
    try:
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
    except OSError:
        print(f"Error: creating directory with name {directory_path}")

def shuffle_data(features, labels):
    shuffled_features, shuffled_labels = shuffle(features, labels, random_state=42)
    return shuffled_features, shuffled_labels

def load_model(model_path):
    with CustomObjectScope({
            'intersection_over_union': intersection_over_union,
            'dice_coefficient': dice_coefficient,
            'dice_loss': dice_loss,
            'binary_crossentropy_dice_loss': binary_crossentropy_dice_loss
        }):
        loaded_model = tf.keras.models.load_model(model_path)
        return loaded_model