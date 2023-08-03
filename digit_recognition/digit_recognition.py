import tensorflow as tf
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten
from keras.utils import to_categorical, normalize
import cv2
import matplotlib.pyplot as plt

def digit_recognition_sequential_model():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # the pixel values from images should be in range [0, 1]
    x_train = normalize(x_train, axis=1)
    x_test = normalize(x_test, axis=1)

    #creating and training the model    
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax')) # 10 outputs for 10 digits; softmax so that the sum of all outputs is 1 (probabilities)

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=10)
    model.save('digit_recognition_model_new_2.keras')


'''
the following was used to load, save and test the model

def preprocess_cell_image(cell_image):
        cell_image = cv2.resize(cell_image, (28, 28))
        cell_image = np.invert(cell_image)
        #normalize pixels to range [0, 1]
        cell_image = cell_image.astype('float32') / 255
        cell_image = np.expand_dims(cell_image, axis=0)
        return cell_image

if __name__ == '__main__':
    # digit_recognition_sequential_model()
    model = load_model('digit_recognition_model_new_2.keras')
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    for digit in [1, 3, 5, 6]:
        image = preprocess_cell_image(cv2.imread(f'digits/{digit}.png')[:, :, 0])
        print('digit is: ', np.argmax(model.predict(image)))
        plt.imshow(image[0], cmap=plt.cm.binary)
        plt.show()
'''
