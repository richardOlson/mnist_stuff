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
    out = keras.layers.Dense(10, )(t)

    model = keras.Model(inputs=inputs, outputs=out)
    # doing the compilation of the model
    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    return model


# This is the function that is the callback for checkpoints
def get_callback():
    path = "checkPoints/cp.ckpt"
    call_back = tf.keras.callbacks.ModelCheckpoint(filepath=path, save_weights_only=True,
                                        verbose=1)
    return call_back

if __name__ == "__main__":
    print(f"This is the version of the tensorflow that is used {tf.version.VERSION}")
    # getting the training and the testing data
    training, testing = load_images()
    
    print(f"The traning and the testing are now loaded")

    train_images, train_labels = reshape_data(training, 1000)
    test_images, test_labels = reshape_data(testing, 1000)
    
    # create the model
    model = create_model()

    # showing the model summary
    # model.summary()

    # getting a callback
    checkPointCallBack = get_callback()

    model.fit(x=train_images, y=train_labels, batch_size=128, epochs=10,
                validation_data=(test_images, test_labels), 
                callbacks=[checkPointCallBack])
    pred = model.predict(x=test_images,batch_size=128, verbose=1, )

    # this is the reloading of the model weights
    model2 = create_model()

    # getting the path
    path = os.path.join(os.path.dirname(__file__), ".." ,  "checkPoints/cp.ckpt")
    # now loading the weights again
    model2.load_weights(path)

    # evaluating the new Model
    loss, acc = model2.evaluate(x=test_images, y=test_labels)
    print(f"The accuracy is now {acc}")
    breakpoint()
    print(f"have done the prediction")

