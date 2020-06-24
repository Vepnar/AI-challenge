#!/usr/bin/env python3
import time
import numpy
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Activation,
    Flatten,
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
)

# Limit gpu usage. zero when there is no limit
GPU_LIMIT = 1152

# Amound of epochs.
EPOCH = 50

# Amount of data used for validation
VALIDATION_SPLIT = 0.1

# Size of training batches
BATCH_SIZE = 200

DATASET_LABELS = "./dataset/labels.npy"
DATASET_FEATURES = "./dataset/features.npy"
LOG_DIR = "./logs/{}"


def setup_gpu_limiter(vram):
    """limit the amound of VRAM the gpu could use.
    This is from stackoverflow
    Source: https://stackoverflow.com/questions/53698035/failed-to-get-convolution-algorithm-this-is-probably-because-cudnn-failed-to-in
    """
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        # Restrict TensorFlow to only allocate 1*X GB of memory on the first GPU
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [
                    tf.config.experimental.VirtualDeviceConfiguration(
                        memory_limit=(vram)
                    )
                ],
            )
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)


def load_dataset():
    """Load the features & labels."""
    x = numpy.load(DATASET_FEATURES)
    y = numpy.load(DATASET_LABELS)

    # The amount of classes in the dataset.
    classes = len(y[0])

    return x, y, classes


def get_tensorboard(name):
    return TensorBoard(log_dir=f"logs/{name}")


def create_model(x, classes):
    """Created model Convolutional & Dense with layers."""
    model = Sequential(
        [
            Conv2D(64, 3, padding="same", activation="relu", input_shape=x.shape[1:],),
            MaxPooling2D(),
            Conv2D(64, 3, padding="same", activation="relu",),
            MaxPooling2D(),
            Flatten(),
            Dense(128, activation="relu"),
            Dense(classes),
        ]
    )

    # Compile the model with the adam optimizer and categorial cross entropy.
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def _get_data(features, labels, batch_size):
    """Only load the batch into memory and not the whole dataset"""
    data_size = len(labels)

    # Never stop sending data
    while 1:
        batch_start = 0
        batch_end = batch_size

        # Loop through all samples
        while batch_start < data_size:
            limit = min(batch_end, data_size)

            # Return one batch back to the trainer
            x = features[batch_start:limit]
            y = labels[batch_start:limit]
            yield (x, y)

            # Increment counter
            batch_start += batch_size
            batch_end += batch_size


def train_generator_model(model, x, y, tensorboard):
    """Train the already generated model with the labels and features."""

    # Take some samples and turn them into validation data
    validation_count = int(len(x) * VALIDATION_SPLIT)
    validation_features = x[:validation_count]
    validation_labels = y[:validation_count]

    # Remove taken items from the test data
    x = x[validation_count:]
    y = y[validation_count:]

    model.fit(
        _get_data(x, y, BATCH_SIZE),
        epochs=EPOCH,
        steps_per_epoch=int(len(x) / BATCH_SIZE),
        validation_data=(validation_features, validation_labels),
        callbacks=[tensorboard],
    )


def train_model(model, x, y, tensorboard):
    """Train the already generated model with the labels and features."""
    model.fit(
        x,
        y,
        batch_size=BATCH_SIZE,
        epochs=EPOCH,
        validation_split=VALIDATION_SPLIT,
        callbacks=[tensorboard],
    )


if __name__ == "__main__":

    # Only limit the gpu usage when this is enabled
    if GPU_LIMIT > 0:
        setup_gpu_limiter(GPU_LIMIT)

    # Name of the current train session.
    name = f"Training-{str(time.time())}"

    # Get tensorboard.
    tensorboard = get_tensorboard(name)

    # Load npy files with the training data.
    x, y, classes = load_dataset()

    model = create_model(x, classes)

    # Train the model with our labels and features.
    train_model(model, x, y, tensorboard)
