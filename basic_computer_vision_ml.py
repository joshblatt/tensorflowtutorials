import tensorflow as tf
import numpy as np
from tensorflow import keras

fashion_mnist = keras.datasets.fashion_mnist

# Training images - set of images to train model
# Training label - number indicating the class of that type of clothing (ie shoe)
# Test images - set of images the model will try to classify
# Test labels - number indicating the class that the model classified the image as
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Takes in a 28x28 set of pixels
# Outputs 1 of 10 values
model = keras.Sequential([
    # input of shape 28 x 28 (size of image)
    keras.layers.Flatten(input_shape=(28, 28)),
    # Going to have 128 functions
    # When the pixels of the shoe get fed in, 1 by 1, the combination of all these functions will output correct value
    # Computer needs to figure out the paramets inside these functions to get that result
    # extends to all other items of clothing in the data set
    keras.layers.Dense(128, activation=tf.nn.relu),
    # number of different items represented in data set
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# relu & softmax - activation functions
# relu - rectified linear unit
# returns a value if > 0
"""
if (x > 0)
    return x;
else
    return 0;
"""
# softmax - picks biggest number in set
# sets max to 1, rest to 0
# all we have to do is find the

model.compile(optimizer=tf.train.AdamOptimizer(), loss='sparse_categorical_crossentropy')

model.fit(train_images, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images, test_labels)

predictions = model.predict(my_images)