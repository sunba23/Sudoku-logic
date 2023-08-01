import tensorflow as tf
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import to_categorical

def digit_recognition_sequential_model():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # the pixel values from images should be in range [0, 1]
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    # the results should be one-hot encoded (e.g. 2 -> [0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    #creating and training the model    
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # training the model
    history = model.fit(x_train, y_train, batch_size=128, epochs=10, validation_data=(x_test, y_test))

    return model


def predict_number(cell_image, model):
    # expand dimensions to fit the model's input shape
    prediction = model.predict(np.expand_dims(cell_image, axis=0))
    return np.argmax(prediction)
