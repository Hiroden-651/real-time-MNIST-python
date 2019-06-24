"""
    MNSIT Neural Network

    This file will implement a neural network on the MNIST dataset using Keras.
"""

import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils

# load data using Keras
def MNIST_normalized():
    # load data from keras
    (train_x, train_y), (test_x, test_y) = mnist.load_data()

    # reformat and normalize data
    train_x = train_x.reshape(train_x.shape[0], train_x.shape[1] * train_x.shape[2]).astype('float32')
    test_x = test_x.reshape(test_x.shape[0], test_x.shape[1] * test_x.shape[2]).astype('float32')

    train_x = train_x / 255.0
    test_x = test_x / 255.0

    # change labels to one-hot encoding
    train_y = np_utils.to_categorical(train_y)
    test_y = np_utils.to_categorical(test_y)

    # return datasets
    return (train_x, train_y), (test_x, test_y)


def create_network(input_layer, hidden_layer, output_layer):
    model = Sequential()
    model.add(Dense(hidden_layer, input_dim=input_layer, kernel_initializer='normal', activation='sigmoid'))
    model.add(Dense(output_layer, kernel_initializer='normal', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def get_trained_network(train_d, train_l, test_d, test_l, epochs):
    model = create_network(784, 100, 10)
    model.fit(train_d, train_l, validation_data=(test_d, test_l), epochs=epochs, verbose=1)
    return model

def single_prediction(img, model):
    return np.amax(model.predict(img.reshape(1, img.shape[0])))