import os
import json
import multiprocessing

# Disable logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from training import *

CONFIG_FILE = "automated.json"


def create_model(x, classes, config):
    """Automated function to generate models"""

    # Generate model with required the minimum amount of layers.
    model = Sequential(
        [
            Conv2D(64, 3, padding="same", activation="relu", input_shape=x.shape[1:]),
            MaxPooling2D(),
        ]
    )

    # Create the amount of convolutional layers given in the configuration file.
    for _ in range(config["con"]):
        model.add(Conv2D(config["con_size"], 3, padding="same", activation="relu"))
        model.add(MaxPooling2D())

    # Turn 3D feature map into a 1D feature vector
    model.add(Flatten())

    # Create the amount of dense layers given in the configuration file.
    for _ in range(config["dense"]):
        model.add(Dense(config["dense_size"], activation="relu"))

    # Output dense layer.
    model.add(Dense(classes))

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
    name = f"Training-{config['name']}-{str(time.time())}"

    # Get tensorboard.
    tensorboard = get_tensorboard(name)

    # Load npy files with the training data.
    x, y, classes = load_dataset()

    model = create_model(x, classes, config)

    # Train the model with our labels and features.
    train_model(model, x, y, tensorboard)


if __name__ == "__main__":
    # Load configuration from the json file
    config_file = open(CONFIG_FILE, "r")
    json_data = json.load(config_file)
    for config in json_data:

        # Create a subproccess for tensorflow.
        # This is the only way to clear the memory when the training session finished
        process_eval = multiprocessing.Process(target=tensorflow_process, args=[config])
        process_eval.start()
        process_eval.join()
        
    # Close the configuration file.
    config_file.close()
