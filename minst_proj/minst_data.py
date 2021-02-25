# This is a file that downloads the minst data and then looks at it

import os

import tensorflow as tf
from tensorflow import keras

def load_images():
    training, testing = keras.datasets.mnist.load_data()
    # Each of these are tuples that contain ndarrays

    return training, testing




def reshape_data(data, num_to_use:int, ):
    images = None
    labels = None
    if isinstance(data, tuple):
        images, labels = data
       
    images = images[:num_to_use].reshape(-1, (28 * 28))
    images = images/255.0
    labels = labels[:num_to_use]
    return images, labels

def create_model():
    """
    The function to create a model
    """
    inputs = keras.layers.Input(shape=(784,))
    t = keras.layers.Dense(512, activation="relu", )(inputs)
    t = keras.layers.Dropout(.2)(t)
    out = keras.layers.Dense(10, )

    model = keras.Model(inputs=inputs, outputs=out)
    # doing the compilation of the model
    model.compile(optimizer='adam',
                loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.metrics.SparseCategoricalAccuracy()])
    return model

if __name__ == "__main__":
    print(f"This is the version of the tensorflow that is used {tf.version.VERSION}")
    # getting the training and the testing data
    training, testing = load_images()
    
    print(f"The traning and the testing are now loaded")

    train_images, train_labels = reshape_data(training, 1000)