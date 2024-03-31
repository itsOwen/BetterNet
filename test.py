import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import time
import numpy as np
import tensorflow as tf
import cv2
from operator import add
from metrics import intersection_over_union, dice_coefficient, weighted_f_score, s_score, e_score, max_e_score, mean_absolute_error
from utils import create_directory, load_model
from data import load_test_dataset
import argparse

def compute_metrics(y_true, y_pred):
    y_pred = y_pred > 0.5
    y_pred_flat = y_pred.reshape(-1).astype(np.uint8)

    y_true = y_true > 0.5
    y_true_flat = y_true.reshape(-1).astype(np.uint8)

    iou_score = intersection_over_union(y_true_flat, y_pred_flat).numpy()
    dice_score = dice_coefficient(y_true_flat, y_pred_flat).numpy()
    f_score = weighted_f_score(y_true_flat, y_pred_flat)
    s_measure_score = s_score(y_true_flat, y_pred_flat)
    e_measure_score = e_score(y_true_flat, y_pred_flat)
    max_e_measure_score = max_e_score(y_true_flat, y_pred_flat)
    mae_score = mean_absolute_error(y_true_flat, y_pred_flat)

    return [iou_score, dice_score, f_score, s_measure_score, e_measure_score, max_e_measure_score, mae_score]

def parse_mask(mask):
    mask = np.squeeze(mask)
    mask = np.stack([mask, mask, mask], axis=-1)
    return mask

if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)

    parser = argparse.ArgumentParser(description='Test model on a dataset.')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset directory in the "Dataset" folder.')
    args = parser.parse_args()

    dataset_name = args.dataset
    print(f"Testing on {dataset_name}")

    dataset_path = os.path.join("Dataset", dataset_name)
    
    test_images, test_masks = load_test_dataset(dataset_path)

    image_size = (224, 224)

    model_path = f"model/model.keras"
    model = load_model(model_path)

    dummy_image = np.zeros((1, 224, 224, 3))
    _ = model.predict(dummy_image)

    metrics_scores = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    inference_times = []

    for i, (image_path, mask_path) in enumerate(zip(test_images, test_masks)):
        name = os.path.basename(mask_path).split(".")[0]
    
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            print(f"Failed to load image: {image_path}. Skipping.")
            continue
        image = cv2.resize(image, image_size)
        original_image = image.copy()
        image = image / 255.0
        image = np.expand_dims(image, axis=0).astype(np.float32)
    
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Failed to load mask: {mask_path}. Skipping.")
            continue
        mask = cv2.resize(mask, image_size)
        original_mask = mask.copy()
        mask = np.expand_dims(mask, axis=0) / 255.0
        mask = mask.astype(np.float32)
    
        start_time = time.time()
        predicted_mask = model.predict(image)
        inference_time = time.time() - start_time
        inference_times.append(inference_time)
        print(f"{name}: {inference_time:1.5f}")
    
        scores = compute_metrics(mask, predicted_mask)
        metrics_scores = list(map(add, metrics_scores, scores))
    
        predicted_mask = (predicted_mask[0] > 0.5) * 255
        predicted_mask = np.array(predicted_mask, dtype=np.uint8)
    
        original_mask = parse_mask(original_mask)
        predicted_mask = parse_mask(predicted_mask)
        separator_line = np.ones((image_size[0], 10, 3)) * 255
    
        concatenated_images = np.concatenate([original_image, separator_line, original_mask, separator_line, predicted_mask], axis=1)
        cv2.imwrite(f"results/{name}.png", concatenated_images)
    
    average_scores = [score_sum / len(test_images) for score_sum in metrics_scores]
    metric_labels = ["mIoU", "mDice", "Fw", "Sm", "Em", "maxEm", "MAE"]
    
    print("\nAverage Scores:")
    for label, score in zip(metric_labels, average_scores):
        print(f"{label}: {score:1.4f}")
    
    mean_inference_time = np.mean(inference_times)
    mean_fps = 1 / mean_inference_time
    print(f"Mean FPS: {mean_fps}")