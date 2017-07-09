# -*- coding: utf-8 -*-

""" Convolutional network applied to CIFAR-10 dataset classification task.
References:
    Learning Multiple Layers of Features from Tiny Images, A. Krizhevsky, 2009.
Links:
    [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

    CSAL4243 Introduction to Machine Learning's assignment 3.
"""
from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation

# Data loading and preprocessing
from tflearn.data_utils import image_preloader
import numpy as np

dataset_file = 'train_data.txt'
X_train, Y_train = image_preloader(dataset_file, image_shape=(50, 50), mode='file',categorical_labels=True, normalize=True,files_extension=['.png','jpg','jpeg'], filter_channel=True)

#print(X_train.shape)
# Real-time data preprocessing
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Real-time data augmentation
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)
img_aug.add_random_blur(sigma_max=5.0)

# Convolutional network building
## To Do.
    ## Define your network here
network = input_data(shape=[None, 50, 50, 3],data_preprocessing=img_prep,data_augmentation=img_aug)
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)
network = fully_connected(network, 256, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 256, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 5, activation='softmax')
network = regression(network, optimizer='adam',loss='categorical_crossentropy',learning_rate=0.001)

# Train using classifier
## To Do
    ## Define model and assign network
    ## Call the fit function for training
model = tflearn.DNN(network, tensorboard_verbose=0)
model.fit(X_train, Y_train, n_epoch=5, shuffle=True, validation_set=0.1,show_metric=True, batch_size=4, run_id='diabetic_cnn')

# Manually save model
## To Do
    ## Save model
model.save('model/modelnew.tflearn')
