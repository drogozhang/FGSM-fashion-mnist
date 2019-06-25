# -*- coding:utf-8 -*-
#
# Created by Drogo Zhang
#
# On 2019-06-22

from keras.models import load_model
import keras.backend as K
import numpy as np


def aiTest(images, shape):
    # default setting
    # class_label = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    #                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    n_classes = 10

    model = load_model('best_model.h5')
    # Set variables
    epochs = 40
    epsilon = 0.05
    x_adv = images
    x_noise = np.zeros_like(images)
    sess = K.get_session()

    initial_class = np.argmax(model.predict(images), axis=1)
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
        x_adv = sess.run(x_adv, feed_dict={model.input: images})
        # get new gradients
        model.predict(x_adv)
    generate_images = x_adv
    return generate_images
