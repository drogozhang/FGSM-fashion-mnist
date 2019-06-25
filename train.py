# -*- coding:utf-8 -*-
#
# Created by Drogo Zhang
#
# On 2019-06-22


from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import Adam
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

data = input_data.read_data_sets('data/fashion')

X_train, y_train = data.train.images, data.train.labels
X_test, y_test = data.test.images, data.test.labels

# reshape data to add colour channel
img_rows, img_cols = 28, 28
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols).astype('float32')
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols).astype('float32')

# confirm load
print('Training data has {} rows'.format(X_train.shape[0]))
print('Test data has {} rows'.format(X_test.shape[0]))

class_label = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1).astype('float32')
# confirm data reshape
print('Training data shape: {}'.format(X_train.shape))
print('Test data shape: {}'.format(X_test.shape))

# one-hot encode outputs
n_classes = 10
y_train = np_utils.to_categorical(y_train, n_classes)
y_test = np_utils.to_categorical(y_test, n_classes)

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# initialize parameters
input_shape = (img_rows, img_cols, 1)
batch_size = 32
lr = 1e-03
epochs = 50
opt = Adam(lr=lr, decay=lr / epochs)

# data augmentation
aug = ImageDataGenerator(rotation_range=10,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         shear_range=0.1,
                         zoom_range=0.1,
                         horizontal_flip=True,
                         fill_mode="nearest")

# LeNet-5 architecture with updated activation (ReLU) and optimizer (Adam)
model = Sequential()

# first layer CONV => ACTIVATION => POOL
model.add(Conv2D(20, kernel_size=(5, 5), padding='same', activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# second layer CONV => ACTIVATION => POOL
model.add(Conv2D(50, kernel_size=(5, 5), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# first FC layer
model.add(Flatten())
model.add(Dense(500, activation='relu'))

# second FC layer with softmax classifier
model.add(Dense(n_classes, activation='softmax'))

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# fit model using data augmentation
H = model.fit_generator(aug.flow(X_train, y_train, batch_size=batch_size),
                        validation_data=(X_test, y_test),
                        steps_per_epoch=len(X_train) // batch_size,
                        epochs=epochs,
                        verbose=1)

model.save('best_model.h5')
