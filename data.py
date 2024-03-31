import os
import numpy as np
import cv2
import glob
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import CustomObjectScope
from metrics import intersection_over_union

# Constants for image dimensions
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

def load_file_names(base_path, file_name):
    file_path = os.path.join(base_path, file_name)
    print(f"Loading names from file: {file_path}")
    
    with open(file_path, "r") as file:
        data = file.read().split("\n")[:-1]

    image_files = []
    mask_files = []

    for name in data:
        # Use glob to find all files matching the pattern (image and mask for any extension)
        image_files_match = glob.glob(os.path.join(base_path, "images", name + ".*"))
        mask_files_match = glob.glob(os.path.join(base_path, "masks", name + ".*"))

        if image_files_match:
            image_files.append(image_files_match[0]) 
        if mask_files_match:
            mask_files.append(mask_files_match[0])

    print(f"Loaded {len(image_files)} images and {len(mask_files)} masks.")
    return image_files, mask_files

def load_dataset(dataset_paths):
    train_images, train_masks, valid_images, valid_masks = [], [], [], []

    for dataset_path in dataset_paths:
        print(f"Loading data from directory: {dataset_path}")
        train_images_data, train_masks_data = load_file_names(dataset_path, "train.txt")
        valid_images_data, valid_masks_data = load_file_names(dataset_path, "val.txt")
        train_images.extend(train_images_data)
        train_masks.extend(train_masks_data)
        valid_images.extend(valid_images_data)
        valid_masks.extend(valid_masks_data)

    print(f"Total training images: {len(train_images)}")
    print(f"Total training masks: {len(train_masks)}")
    print(f"Total validation images: {len(valid_images)}")
    print(f"Total validation masks: {len(valid_masks)}")

    return (train_images, train_masks), (valid_images, valid_masks)

def load_test_dataset(test_dataset_path):
    test_images, test_masks = [], []

    # Load names from the val.txt file
    test_images_data, test_masks_data = load_file_names(test_dataset_path, "val.txt")

    for image_path, mask_path in zip(test_images_data, test_masks_data):
        test_images.append(image_path)
        test_masks.append(mask_path)

    print(f"Loaded {len(test_images)} test images and {len(test_masks)} test masks.")
    return test_images, test_masks

def read_image(image_path):
    image_path = image_path.decode()
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_LANCZOS4)
    image = image / 255.0
    return image.astype(np.float32)

def read_mask(mask_path):
    mask_path = mask_path.decode()
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (IMAGE_WIDTH, IMAGE_HEIGHT))
    mask = mask / 255.0
    return np.expand_dims(mask, axis=-1).astype(np.float32)

def parse_image_and_mask(image_path, mask_path):
    def _parse(image_path, mask_path):
        return read_image(image_path), read_mask(mask_path)

    image, mask = tf.numpy_function(_parse, [image_path, mask_path], [tf.float32, tf.float32])
    image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, 3])
    mask.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, 1])
    return image, mask

def create_dataset(image_paths, mask_paths, batch_size=8):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    dataset = dataset.map(parse_image_and_mask)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset