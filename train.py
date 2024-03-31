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

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train the BetterNet model on multiple datasets.')
    parser.add_argument('-bs', '--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('-e', '--num_epochs', type=int, default=200, help='Number of epochs for training')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    np.random.seed(42)
    tf.random.set_seed(42)

    input_shape = (224, 224, 3)
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs

    model_save_path = "model/model.keras"
    csv_log_path = "model/data.csv"
    tensorboard_log_path = "logs"

    create_directory("model")

    # Paths to both datasets
    dataset_paths = ["Dataset/Kvasir-SEG", "Dataset/CVC-ClinicDB"]

    loaded_dataset = load_dataset(dataset_paths)
    print(f"Training images: {len(loaded_dataset[0][0])}")
    print(f"Training masks: {len(loaded_dataset[0][1])}")
    print(f"Validation images: {len(loaded_dataset[1][0])}")
    print(f"Validation masks: {len(loaded_dataset[1][1])}")

    combined_images = np.concatenate((loaded_dataset[0][0], loaded_dataset[1][0]))
    combined_masks = np.concatenate((loaded_dataset[0][1], loaded_dataset[1][1]))
    print(f"Total images after combining datasets: {len(combined_images)}")
    print(f"Total masks after combining datasets: {len(combined_masks)}")

    # Shuffle the combined dataset to ensure a good mix
    indices = np.arange(len(combined_images))
    np.random.shuffle(indices)
    shuffled_images = combined_images[indices]
    shuffled_masks = combined_masks[indices]
    print(f"Total images after shuffling: {len(shuffled_images)}")
    print(f"Total masks after shuffling: {len(shuffled_masks)}")

    # Split dataset into training and validation
    split_index = int(0.8 * len(shuffled_images))
    train_images, val_images = shuffled_images[:split_index], shuffled_images[split_index:]
    train_masks, val_masks = shuffled_masks[:split_index], shuffled_masks[split_index:]
    print(f"Training images after splitting: {len(train_images)}")
    print(f"Training masks after splitting: {len(train_masks)}")
    print(f"Validation images after splitting: {len(val_images)}")
    print(f"Validation masks after splitting: {len(val_masks)}")

    train_dataset = create_dataset(train_images, train_masks, batch_size)
    val_dataset = create_dataset(val_images, val_masks, batch_size)
    print(f"Training dataset: {train_dataset}")
    print(f"Validation dataset: {val_dataset}")

    model = BetterNet(input_shape=input_shape, num_classes=1, dropout_rate=0.5)
    model.compile(loss=binary_crossentropy_dice_loss, optimizer=Adam(learning_rate), metrics=[dice_coefficient, intersection_over_union, weighted_f_score, s_score, e_score])
    model.summary()

    print('------------------------------------------------------------------------')
    print('Training on combined datasets...')
    training_history = model.fit(
        train_dataset,
        epochs=num_epochs,
        validation_data=val_dataset,
        steps_per_epoch=len(train_images) // batch_size,
        validation_steps=len(val_images) // batch_size,
        callbacks=[
            ModelCheckpoint(model_save_path, monitor='val_loss', verbose=1, save_best_only=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-7, verbose=1),
            CSVLogger(csv_log_path),
            TensorBoard(log_dir=tensorboard_log_path),
            EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
        ]
    )

    del model, training_history, train_dataset, val_dataset
    tf.keras.backend.clear_session()
    gc.collect()
