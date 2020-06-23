#!/usr/bin/env python3

import os
import json

# Directory where we store the metadata files
DATASET_METADATA = "./dataset/metadata"

# Store the parsed data
label_list = {}
labels = 0

print("Started reading", end="")

# Find all images inside the DATASET_METADATA directory
for img in os.listdir(DATASET_METADATA):

    # Open the file and read the json
    real_path = os.path.join(DATASET_METADATA, img)
    with open(real_path) as metadata_file:
        metadata = json.load(metadata_file)

        # Parse the expected quantity
        expected_quantity = metadata["EXPECTED_QUANTITY"]

        # Update the expected quantity dictionary if it exists.
        # Or create a new key in the dictionary if it doesn't exist
        if expected_quantity in label_list:
            label_list[expected_quantity] += 1
        else:
            label_list.update({expected_quantity: 1})

        # Update the label counter.
        labels += 1

        # Print a dot every 1000 files.
        if labels % 1000 == 0:
            print(".", end="")

# Print the parsed data in json format
print("")
print(f"Total amount of labels {labels}")
print(f"lables: {json.dumps(label_list,sort_keys=True, indent=4)}")
