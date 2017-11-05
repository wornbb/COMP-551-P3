from __future__ import print_function
import os
import keras
from keras.datasets import mnist
from keras import optimizers
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pylab as plt

import numpy as np
import csv

batch_size = 64
num_classes = 40
epochs = 40
# input image dimensions
img_x, img_y = 64, 64

x_train = np.loadtxt("train_x.csv", delimiter=",")  # load from text
print("train_x is loaded")
y_train = np.loadtxt("train_y_preprocessed.csv", delimiter=",")
print("train_y is loaded")
x_pred = np.loadtxt("test_x.csv", delimiter=",")
print("test_x is loaded")

# split dataset to train dataset and test dataset
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.10)

# reshape the data into a 4D tensor - (sample_number, x_img_size, y_img_size, num_channels)
x_train = x_train.reshape(x_train.shape[0], img_x, img_y, 1)
x_test = x_test.reshape(x_test.shape[0], img_x, img_y, 1)
x_pred = x_pred.reshape(x_pred.shape[0], img_x, img_y, 1)
input_shape = (img_x, img_y, 1)

# convert the data to the right type
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_pred = x_pred.astype('float32')

x_train /= 255
x_test /= 255
x_pred /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print(x_pred.shape[0], 'predict samples')

# convert class vectors to binary class matrices - this is for use in the
# categorical_crossentropy loss below
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=input_shape))
BatchNormalization(axis=-1)
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
BatchNormalization(axis=-1)
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
BatchNormalization(axis=-1)
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
BatchNormalization(axis=-1)
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

# Fully connected layer
model.add(Dense(512))
BatchNormalization()
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes))

model.add(Activation('softmax'))
#sgd = optimizers.SGD(lr=1e-4, momentum=0.9)
#adam = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
rmsprop = keras.optimizers.RMSprop(lr=0.0015, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# class AccuracyHistory(keras.callbacks.Callback):
#     def on_train_begin(self, logs={}):
#         self.acc = []
#
#     def on_epoch_end(self, batch, logs={}):
#         self.acc.append(logs.get('acc'))
#
# history = AccuracyHistory()

# model.fit(x_train, y_train,
#           batch_size=batch_size,
#           epochs=epochs,
#           verbose=2,
#           validation_split=0.1,
#           callbacks=[history])

gen = ImageDataGenerator(rotation_range=90, width_shift_range=0.1, shear_range=0.3,
                         height_shift_range=0.1, zoom_range=0.1)
test_gen = ImageDataGenerator()

train_generator = gen.flow(x_train, y_train, batch_size=batch_size)
test_generator = test_gen.flow(x_test, y_test, batch_size=batch_size)

model.fit_generator(train_generator, steps_per_epoch=len(x_train)//batch_size, epochs=epochs,
                    validation_data=test_generator, validation_steps=len(x_test)//batch_size)

# model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
#                     steps_per_epoch=len(x_train) / batch_size, epochs=epochs)

# score = model.evaluate(x_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])
# plt.plot(range(1, 11), history.acc)
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.show()

#######################################  Prediction   ###########################
preds = model.predict_classes(x_pred, verbose=0)

with open('predictions_cnn.csv', 'w', newline='') as f:
    fieldnames = ["Id", "Label"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(len(preds)):
        writer.writerow({'Id': i+1, 'Label': np.uint8(preds[i])})
