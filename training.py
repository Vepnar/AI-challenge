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
)

# This is from stackoverflow
# Source: https://stackoverflow.com/questions/53698035/failed-to-get-convolution-algorithm-this-is-probably-because-cudnn-failed-to-in
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    # Restrict TensorFlow to only allocate 1*X GB of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=(784))],
        )
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

# The smaller the images the faster the neuralnetwork learns.
IMG_HEIGHT = 50

# The deeplearning neuralnetwork always need images from the same size.
IMG_WIDTH = 50

# Amound of epochs.
EPOCH = 25

# Amount of data used for validation
VALIDATION_SPLIT = 0.1

# Size of training batches
BATCH_SIZE = 256

DATASET_LABELS = "./dataset/labels.npy"
DATASET_FEATURES = "./dataset/features.npy"
LOG_DIR = "./logs/{}"

NAME = f"bin-counter-{int(time.time())}"

# Load the features & labels.
x = numpy.load(DATASET_FEATURES)
y = numpy.load(DATASET_LABELS)

# The amount of classes in the dataset.
CLASSES = len(y[0])

# Load tensorboard
tensorboard = TensorBoard(log_dir="logs/{NAME}")

# Created model wigh layers
model = Sequential(
    [
        Conv2D(64, 3, padding="same", activation="relu", input_shape=x.shape[1:],),
        MaxPooling2D(),
        Conv2D(64, 3, padding="same", activation="relu",),
        MaxPooling2D(),
        Flatten(),
        Dense(128, activation="relu"),
        Dense(CLASSES),
    ]
)

# Compile the model with the adam optimizer and categorial cross entropy.
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# Train the model
model.fit(
    x,
    y,
    batch_size=BATCH_SIZE,
    epochs=EPOCH,
    validation_split=VALIDATION_SPLIT,
    callbacks=[tensorboard],
)
