import os
import json
import multiprocessing

# Disable logging
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from training import *

CONFIG_FILE = "automated.json"


def create_layer(config, model, x, classes):
    """Create a layer from a dictionary"""
    options = {}

    # Get the type of activation
    option = config.get("activation")
    if option:
        options.update({"activation": option})

    # Get the type of padding
    option = config.get("padding")
    if option:
        options.update({"padding": option})

    # Set layer as input layer
    option = config.get("input_layer")
    if option:
        options.update({"input_shape": x.shape[1:]})

    # Set layer as output layer
    option = config.get("output_layer")
    if option:
        size = classes
    else:
        size = config.get("nodes")

    # Set the type of layer for example cov2d or dense.
    option = config.get("type")
    if option == "cov2d":
        kernel = config.get("kernel", 3)
        model.add(Conv2D(size, kernel, **options))
        model.add(MaxPooling2D())

    if option == "dense":
        model.add(Dense(size, **options))

    if option == "flatten":
        model.add(Flatten())

    # Add a dropout.
    if config.get("dropout"):
        model.add(Dropout(config["dropout"]))

    # Add a batch normalization layer.
    if config.get("normalization"):
        model.add(BatchNormalization())


def create_model(x, classes, config):
    """Automated function to generate models"""

    # Generate model with required the minimum amount of layers.
    model = Sequential()

    # Create layers for the model based on the settings in the configuration file
    for layer in config.get("layers"):
        create_layer(layer, model, x, classes)

    # Compile the model with the adam optimizer and categorial cross entropy.
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    return model


def tensorflow_process(config):
    # Only limit the gpu usage when this is enabled
    if GPU_LIMIT > 0:
        setup_gpu_limiter(GPU_LIMIT)

    # Name of the current train session.
    name = f"{config['name']}-{str(time.time())}"

    # Get tensorboard.
    tensorboard = get_tensorboard(name)

    # Load npy files with the training data.
    x, y, classes = load_dataset()

    model = create_model(x, classes, config)

    # Train the model with our labels and features.
    train_generator_model(model, x, y, tensorboard)


if __name__ == "__main__":
    # Load configuration from the json file
    config_file = open(CONFIG_FILE, "r")
    json_data = json.load(config_file)
    print(f"{len(json_data)} Types of configuration found in the config file.")
    for config in json_data:

        # Create a subproccess for tensorflow.
        # This is the only way to clear the memory when the training session finished
        process_eval = multiprocessing.Process(target=tensorflow_process, args=[config])
        process_eval.start()
        process_eval.join()

    # Close the configuration file.
    config_file.close()
