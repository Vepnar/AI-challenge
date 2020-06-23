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

# The labels we want to learn to the AI.
SELECTED_LABEL = [2, 3, 4, 5, 6]

# The amount of samples per label
LABEL_SIZE = 2438

# The directory where we downloaded our images.
DATASET_IMAGES = "./dataset/images"
DATASET_METADATA = "./dataset/metadata"

DATASET_LABELS = "./dataset/labels.npy"
DATASET_FEATURES = "./dataset/features.npy"

# DONT CHANGE THE VARIABLES BELOW.
CLASSES = len(SELECTED_LABEL)
LABEL_DICT = {}

# Add empty lists to the selected label dictionary.
for selected in SELECTED_LABEL:
    LABEL_DICT.update({selected: 0})


def process_image(path):
    # Parse the filename to find the metadata file.
    file_base = os.path.basename(path)
    file_name = os.path.splitext(file_base)[0]
    metadata_path = f"{DATASET_METADATA}/{file_name}.json"

    # Open the meta.
    with open(metadata_path) as metadata_file:
        metadata = json.load(metadata_file)
        expected_quantity = metadata["EXPECTED_QUANTITY"]
        if not expected_quantity in SELECTED_LABEL:
            return
        if LABEL_DICT[expected_quantity] > LABEL_SIZE:
            return

        # Increment the label counter.
        LABEL_DICT[expected_quantity] += 1

    # Convert expected_quantity to an one hot encoding.
    label = [0] * CLASSES
    label[SELECTED_LABEL.index(expected_quantity)] = 1

    # Grayscale the images will greatly reduce the size of the data.
    img_array = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # Resizing the images will also reduce the size of the image (makes sense).
    # Plus not all images are equal size and our neural net will require images of equal size.
    img_array = cv2.resize(img_array, (IMG_WIDTH, IMG_HEIGHT))

    # Return processed image and the expected quantity.
    return img_array, label


if __name__ == "__main__":

    # Store all input data and labels in one list.
    training_data = []

    # Go through all images listed in the dataset directory.
    print("Resizing and recolouring images")
    for img in os.listdir(DATASET_IMAGES):
        real_path = os.path.join(DATASET_IMAGES, img)

        # Process the image.
        sample_with_label = process_image(real_path)
        if sample_with_label:
            training_data.append(sample_with_label)

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

    print(f"{len(training_data)} samples processed")
    print(f"lables: {json.dumps(LABEL_DICT, sort_keys=True, indent=4)}")
