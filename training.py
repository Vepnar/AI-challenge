#!/usr/bin/env python3
import numpy
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Activation,
    Flatten,
    Conv2D,
    MaxPooling2D,
)

# The smaller the images the faster the neuralnetwork learns.
IMG_HEIGHT = 50

# The deeplearning neuralnetwork always need images from the same size.
IMG_WIDTH = 50

# Amound of epochs.
EPOCH = 25

# Amount of data used for validation
VALIDATION_SPLIT = 0.1

# Size of training batches
BATCH_SIZE = 32

# The amount of classes
CLASSES = 5

DATASET_LABELS = "./dataset/labels.npy"
DATASET_FEATURES = "./dataset/features.npy"

# Load the features & labels.
x = numpy.load(DATASET_FEATURES)
y = numpy.load(DATASET_LABELS)

# Created model wigh layers
model = Sequential(
    [
        Conv2D(64, 3, padding="same", activation="relu", input_shape=x.shape[1:],),
        MaxPooling2D(),
        Conv2D(32, 3, padding="same", activation="relu",),
        MaxPooling2D(),
        Conv2D(128, 3, padding="same", activation="relu",),
        MaxPooling2D(),
        Flatten(),
        Dense(512, activation="relu"),
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
model.fit(x, y, batch_size=BATCH_SIZE, epochs=EPOCH, validation_split=VALIDATION_SPLIT)
