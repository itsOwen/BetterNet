import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
from metrics import dice_coefficient, intersection_over_union, binary_crossentropy_dice_loss, weighted_f_score, s_score, e_score
from model import BetterNet
from utils import create_directory
from data import create_dataset, load_dataset
import gc
import cv2
import pydensecrf.densecrf as dcrf

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train the BetterNet model on multiple datasets.')
    parser.add_argument('-bs', '--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('-e', '--num_epochs', type=int, default=100, help='Number of epochs for training')
    parser.add_argument('--dataset_paths', nargs='+', default=["Dataset/Kvasir-SEG", "Dataset/CVC-ClinicDB"], help='List of dataset paths')
    return parser.parse_args()

def apply_crf(image, segmentation_mask):
    d = dcrf.DenseCRF2D(image.shape[1], image.shape[0], num_classes)

    unary = segmentation_mask.squeeze().reshape((num_classes, -1))
    d.setUnaryEnergy(-unary.astype(np.float32))

    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=image.astype(np.uint8), compat=10)

    refined_segmentation = np.argmax(d.inference(10), axis=0).reshape((image.shape[0], image.shape[1]))
    
    return refined_segmentation

def apply_morphological_operations(segmentation_mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    segmented_image_dilated = cv2.dilate(segmentation_mask, kernel, iterations=1)
    segmented_image_eroded = cv2.erode(segmented_image_dilated, kernel, iterations=1)
    
    return segmented_image_eroded

if __name__ == "__main__":
    args = parse_arguments()
    np.random.seed(42)
    tf.random.set_seed(42)

    input_shape = (224, 224, 3)
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    dataset_paths = args.dataset_paths

    model_save_path = "model/model.keras"
    csv_log_path = "model/model_data.csv"
    tensorboard_log_path = "logs"

    create_directory("model")

    loaded_dataset = load_dataset(dataset_paths)

    combined_images = np.concatenate((loaded_dataset[0][0], loaded_dataset[1][0]))
    combined_masks = np.concatenate((loaded_dataset[0][1], loaded_dataset[1][1]))

    indices = np.arange(len(combined_images))
    np.random.shuffle(indices)
    shuffled_images = combined_images[indices]
    shuffled_masks = combined_masks[indices]

    split_index = int(0.9 * len(shuffled_images))
    train_images, val_images = shuffled_images[:split_index], shuffled_images[split_index:]
    train_masks, val_masks = shuffled_masks[:split_index], shuffled_masks[split_index:]

    train_dataset = create_dataset(train_images, train_masks, batch_size)
    val_dataset = create_dataset(val_images, val_masks, batch_size)

    model = BetterNet(input_shape=input_shape, num_classes=1, dropout_rate=0.5)
    model.compile(loss=binary_crossentropy_dice_loss, optimizer=Adam(learning_rate), metrics=[dice_coefficient, intersection_over_union, weighted_f_score, s_score, e_score])
    model.summary()

    print('------------------------------------------------------------------------')
    print('Training on two combined datasets...')
    training_history = model.fit(
        train_dataset,
        epochs=num_epochs,
        validation_data=val_dataset,
        steps_per_epoch=len(train_images) // batch_size,
        validation_steps=len(val_images) // batch_size,
        callbacks=[
            ModelCheckpoint(model_save_path, monitor='val_loss', verbose=1, save_best_only=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=15, min_lr=1e-7, verbose=1),
            CSVLogger(csv_log_path),
            TensorBoard(log_dir=tensorboard_log_path),
            EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
        ]
    )

    # Applying post-processing to validation predictions
    val_predictions = model.predict(val_images)
    val_predictions_crf = [apply_crf(image, mask) for image, mask in zip(val_images, val_predictions)]
    val_predictions_final = [apply_morphological_operations(mask) for mask in val_predictions_crf]

    del model, training_history, train_dataset, val_dataset
    tf.keras.backend.clear_session()
    gc.collect()