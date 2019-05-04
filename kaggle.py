import numpy as np
import pickle
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, AveragePooling2D
import keras
import cv2
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, AveragePooling2D
import keras
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import TensorBoard
from sklearn.utils.class_weight import compute_class_weight


image_size = 48  # Pixels * pixels in image
emotions = ["neutral", "happy", "anger"] #"sadness", "surprise"]

with open("fer2013.csv") as f:
    content = f.readlines()

lines = np.array(content)

num_of_instances = lines.size
print("number of instances: ", num_of_instances)

num_classes = 7
x_train, y_train, x_test, y_test = [], [], [], []

for i in range(1, num_of_instances):
    try:
        emotion, img, type = lines[i].split(",")

        # emotion = '2'

        val = img.split(" ")
        pixels = np.array(val, 'float32')

        #emotion = keras.utils.to_categorical(emotion, 5)

        #if (type == 'Training' or type == 'PublicTest'):
        if 'Training' in type:
            y_train.append(emotion)
            x_train.append(pixels)
        elif 'PublicTest' in type:
            y_test.append(emotion)
            x_test.append(pixels)
    except:
        print("", end="")

y_train = keras.utils.to_categorical(y_train, num_classes)
x_train = np.array(x_train).reshape(-1, image_size, image_size, 1)
y_test = keras.utils.to_categorical(y_test, num_classes)
x_test = np.array(x_test).reshape(-1, image_size, image_size, 1)

####### Model ############
model = Sequential()

# 1st convolution layer
model.add(Conv2D(128, (3,3), input_shape= (image_size, image_size, 1))) #try 128***
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

# fully connected neural networks
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(num_classes, activation='softmax'))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

#x_train = keras.utils.normalize(x_train, axis=1) #important

model.fit(x_train, y_train, epochs=30, batch_size = 32)

model.save('evaluation.model')

train_score = model.evaluate(x_train, y_train, verbose=0)
print('Train accuracy:', 100 * train_score[1])

test_score = model.evaluate(x_test, y_test, verbose=0)
print('Test accuracy:', 100 * test_score[1])


