#!/usr/bin/env python3
import os
import cv2
import json
import random
import numpy as np

# The smaller the images the faster the neuralnetwork learns.
IMG_HEIGHT = 50
# The deeplearning neuralnetwork always need images from the same size.
IMG_WIDTH = 50

CLASSES = 6

# The directory where we downloaded our images.
DATASET_IMAGES = "./dataset/images"
DATASET_METADATA = "./dataset/metadata"

DATASET_LABELS = "./dataset/labels.npy"
DATASET_FEATURES = "./dataset/features.npy"


def process_image(path):

    # Grayscale the images will greatly reduce the size of the data.
    img_array = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # Resizing the images will also reduce the size of the image (makes sense).
    # Plus not all images are equal size and our neural net will require images of equal size.
    img_array = cv2.resize(img_array, (IMG_WIDTH, IMG_HEIGHT))

    # Parse the filename to find the metadata file.
    file_base = os.path.basename(path)
    file_name = os.path.splitext(file_base)[0]
    metadata_path = f"{DATASET_METADATA}/{file_name}.json"

    # Open the meta.
    with open(metadata_path) as metadata_file:
        metadata = json.load(metadata_file)
        expected_quantity = metadata["EXPECTED_QUANTITY"]

    # Convert expected_quantity to an one hot encoding.
    label = [0] * CLASSES
    label[expected_quantity] = 1

    # Return processed image and the expected quantity.
    return img_array, label


if __name__ == "__main__":

    # Store all input data and labels in one list.
    training_data = []

    # Go through all images listed in the dataset directory
    print("Resizing and recolouring images")
    for img in os.listdir(DATASET_IMAGES):
        real_path = os.path.join(DATASET_IMAGES, img)

        # Process our images.
        x, y = process_image(real_path)
        training_data.append((x, y))

    # Shuffle the images.
    random.shuffle(training_data)

    # Features
    x = []
    # Labels
    y = []

    # Add our shuffeld data to the feature & label array.
    for features, labels in training_data:
        x.append(features)
        y.append(labels)

    # Reshape the features array.
    x = np.array(x).reshape(-1, IMG_HEIGHT, IMG_WIDTH, 1)

    # Normalize the features to floats
    x = x / 255.0

    # Save our procesed data to a numpy file
    np.save(DATASET_LABELS, y)
    np.save(DATASET_FEATURES, x)
