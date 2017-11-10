from __future__ import print_function

import pickle
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import csv
import h5py
from keras.models import model_from_json
import os

np.random.seed(12345)

"""
    Mapping labels to [0,39]
     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
     20, 21, 24, 25, 27, 28, 30, 32, 35, 36, 40, 42, 45, 48, 49, 54, 56, 63, 64, 72, 81]
"""

def label_map(in_labels, num_instance):
    out_labels = np.zeros((in_labels.shape[0], ))
    for i in range(num_instance):
        if in_labels[i] < 19:
            out_labels[i] = in_labels[i]
        if in_labels[i] == 20:
            out_labels[i] = 19
        if in_labels[i] == 21:
            out_labels[i] = 20
        if in_labels[i] == 24:
            out_labels[i] = 21
        if in_labels[i] == 25:
            out_labels[i] = 22
        if in_labels[i] == 27:
            out_labels[i] = 23
        if in_labels[i] == 28:
            out_labels[i] = 24
        if in_labels[i] == 30:
            out_labels[i] = 25
        if in_labels[i] == 32:
            out_labels[i] = 26
        if in_labels[i] == 35:
            out_labels[i] = 27
        if in_labels[i] == 36:
            out_labels[i] = 28
        if in_labels[i] == 40:
            out_labels[i] = 29
        if in_labels[i] == 42:
            out_labels[i] = 30
        if in_labels[i] == 45:
            out_labels[i] = 31
        if in_labels[i] == 48:
            out_labels[i] = 32
        if in_labels[i] == 49:
            out_labels[i] = 33
        if in_labels[i] == 54:
            out_labels[i] = 34
        if in_labels[i] == 56:
            out_labels[i] = 35
        if in_labels[i] == 63:
            out_labels[i] = 36
        if in_labels[i] == 64:
            out_labels[i] = 37
        if in_labels[i] == 72:
            out_labels[i] = 38
        if in_labels[i] == 81:
            out_labels[i] = 39
    return out_labels

def label_rev_map(in_labels, num_instance):
    out_labels = np.zeros((in_labels.shape[0], ))
    for i in range(num_instance):
        if in_labels[i] < 19:
            out_labels[i] = in_labels[i]
        if in_labels[i] == 19:
            out_labels[i] = 20
        if in_labels[i] == 20:
            out_labels[i] = 21
        if in_labels[i] == 21:
            out_labels[i] = 24
        if in_labels[i] == 22:
            out_labels[i] = 25
        if in_labels[i] == 23:
            out_labels[i] = 27
        if in_labels[i] == 24:
            out_labels[i] = 28
        if in_labels[i] == 25:
            out_labels[i] = 30
        if in_labels[i] == 26:
            out_labels[i] = 32
        if in_labels[i] == 27:
            out_labels[i] = 35
        if in_labels[i] == 28:
            out_labels[i] = 36
        if in_labels[i] == 29:
            out_labels[i] = 40
        if in_labels[i] == 30:
            out_labels[i] = 42
        if in_labels[i] == 31:
            out_labels[i] = 45
        if in_labels[i] == 32:
            out_labels[i] = 48
        if in_labels[i] == 33:
            out_labels[i] = 49
        if in_labels[i] == 34:
            out_labels[i] = 54
        if in_labels[i] == 35:
            out_labels[i] = 56
        if in_labels[i] == 36:
            out_labels[i] = 63
        if in_labels[i] == 37:
            out_labels[i] = 64
        if in_labels[i] == 38:
            out_labels[i] = 72
        if in_labels[i] == 39:
            out_labels[i] = 81
    return out_labels

def main():

    x_train = np.loadtxt("train_x.csv", delimiter=",")  # load from text
    print("train_x is loaded")
    y_train = np.loadtxt("train_y.csv", delimiter=",")
    print("train_y is loaded")
    x_predict = np.loadtxt("test_x.csv", delimiter=",")
    print("test_x is loaded")

    # hyper parameter
    batch_size = 64
    num_classes = 40
    epochs1 = 30
    epochs2 = 50
    aug_epochs1 = 50
    aug_epochs2 = 400
    test_size = 0.05
    # input image dimensions
    img_rows, img_cols = 64, 64

    input_shape = (img_rows, img_cols, 1)
    x_train = x_train.reshape((-1, img_rows, img_cols, 1))
    x_predict = x_predict.reshape((-1, img_rows, img_cols, 1))
    print('y_train shape:', y_train.shape)
    y_train = label_map(y_train, y_train.shape[0])
    print('y_train shape:', y_train.shape)
    y_train = y_train.reshape((y_train.shape[0],))

    x_train = x_train.astype('float32')
    x_predict = x_predict.astype('float32')
    x_train /= 255
    x_predict /= 255
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=test_size, random_state=12345)

    print('x_train shape:', x_train.shape)  # (-1,28,28,1)
    print(x_train.shape[0], 'train samples')
    print('y_train shape:', y_train.shape)  # (-1,)
    class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    print(class_weights)
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

##########################################    CNN Model   ################################################
    mean_px = x_train.mean().astype(np.float32)
    std_px = x_train.std().astype(np.float32)

    def norm_input(x):
        return (x - mean_px) / std_px

    model = Sequential([
        Lambda(norm_input, input_shape=input_shape),
        Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=input_shape),
        Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Flatten(),

        Dense(1024, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1024, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1024, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(40, activation='softmax')
    ])

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=0.001),
                  metrics=['accuracy'])

################################################    Training    ################################################

    earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=30, verbose=0, mode='min')

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs1,
              verbose=1,
              validation_data=(x_test, y_test),
              class_weight=class_weights,
              callbacks=[earlyStopping])

    model.optimizer.lr = 0.0001
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs2,
              verbose=1,
              validation_data=(x_test, y_test),
              class_weight=class_weights,
              callbacks=[earlyStopping])

    # score = model.evaluate(x_test, y_test, verbose=0)
    # print('Test loss:', score[0])
    # print('Test accuracy:', score[1])
    gen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1, shear_range=0.3,
                             height_shift_range=0.1, zoom_range=0.1)
    batches = gen.flow(x_train, y_train, batch_size=batch_size)
    val_batches = gen.flow(x_test, y_test, batch_size=batch_size)
    model.optimizer.lr = 0.001
    model.fit_generator(batches, steps_per_epoch=x_train.shape[0] // batch_size, epochs=aug_epochs1,
                        validation_data=val_batches, validation_steps=x_test.shape[0] // batch_size,
                        use_multiprocessing=False, class_weight=class_weights, callbacks=[earlyStopping])

    model.optimizer.lr = 0.0001
    model.fit_generator(batches, steps_per_epoch=x_train.shape[0] // batch_size, epochs=aug_epochs2,
                        validation_data=val_batches, validation_steps=x_test.shape[0] // batch_size,
                        use_multiprocessing=False, class_weight=class_weights)

#############################################   Predicting  ####################################################
    y_predict = model.predict_classes(x_predict, batch_size=batch_size, verbose=0)
    y_predict = label_rev_map(y_predict, y_predict.shape[0])
    with open('predictions_cnn.csv', 'w', newline='') as f:
        fieldnames = ["Id", "Label"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(y_predict)):
            writer.writerow({'Id': i + 1, 'Label': np.uint8(y_predict[i])})
    print("File Written")

    # model_json = model.to_json()
    # json_name = "model_" + str(version) + ".json"
    # h5_name = "model_" + str(version) + ".h5"
    # with open(json_name, "w") as json_file:
    #     json_file.write(model_json)
    # model.save_weights(h5_name)
    # print("Saved model to disk")


if __name__ == '__main__':
    main()
