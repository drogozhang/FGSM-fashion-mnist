# -*- coding:utf-8 -*-
#
# Created by Drogo Zhang
#
# On 2019-06-22

"""Using original fashion mnist to test the """

import numpy as np
import keras.backend as K
from keras.utils import np_utils
from keras.models import load_model
from tensorflow.examples.tutorials.mnist import input_data
from main import aiTest

model = load_model('best_model.h5')

data = input_data.read_data_sets('data/fashion')

X_test, y_test = data.test.images, data.test.labels

# reshape data to add colour channel
img_rows, img_cols = 28, 28
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols).astype('float32')

class_label = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1).astype('float32')

# one-hot encode outputs
n_classes = 10
y_test_one_hot = np_utils.to_categorical(y_test, n_classes)

predict_result = np.argmax(model.predict(X_test), axis=1)

print("Accuracy:{:.2f}".format(np.sum(predict_result == y_test) / y_test.size * 100))

# # todo using main aiTest
# generate_images = aiTest(X_test, 1)
# predict_result = np.argmax(model.predict(generate_images), axis=1)
# print("Accuracy:{:.2f}".format(np.sum(predict_result == y_test) / y_test.size * 100))



# todo self test

# Set variables
epochs = 80
epsilon = 0.1
prev_probs = []

x_adv = X_test
x_noise = np.zeros_like(X_test)
sess = K.get_session()

initial_class = np.argmax(model.predict(X_test), axis=1)
for i in range(epochs):
    # One hot encode the initial class
    target = K.one_hot(initial_class, n_classes)

    # Get the loss and gradient of the loss wrt the inputs
    loss = K.categorical_crossentropy(target, model.output)
    grads = K.gradients(loss, model.input)

    # Get the sign of the gradient
    delta = K.sign(grads[0])
    x_noise = x_noise + delta

    # Perturb the image
    x_adv = x_adv + epsilon * delta

    # Get the new image and predictions
    x_adv = sess.run(x_adv, feed_dict={model.input: X_test})
    preds = model.predict(x_adv)
    # Store the probability of the target class
    prev_probs.append(preds[0][initial_class])

    preds = np.argmax(preds, axis=1)
    if i % 20 == 0:
        print("After epoch " + str(i + 1) + " attack: ")
        print("Accuracy:{:.2f}\n".format(np.sum(preds == y_test) / y_test.size * 100))
