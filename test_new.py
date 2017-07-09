from __future__ import absolute_import, division, print_function

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation



# Real-time data preprocessing
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Real-time data augmentation
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)

# Convolutional network building
## To Do. same as before.
    ## Define your network here
network = input_data(shape=[None, 50, 50, 3], 
                        data_preprocessing=img_prep, 
                        data_augmentation=img_aug) 

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
network = regression(network, optimizer='adam', 
                        loss='categorical_crossentropy', 
                        learning_rate=0.001) 
# Define model
## To Do
    ## Define model and assign network. Same as training.
model = tflearn.DNN(network, tensorboard_verbose=0)
# Load Model into model object
## To Do.
    ## Use the model.load() function
model.load('model/modelnew.tflearn')

# load test images
import numpy as np
# Load path/class_id image file:
dataset_folder = 'testdata/'
from tflearn.data_utils import image_preloader

X_test, Y_test = image_preloader(dataset_folder, image_shape=(50, 50), mode='folder',
                       categorical_labels=True, normalize=True,
                       files_extension=['.jpeg', '.png','.jpg'], filter_channel=True)
X_test = np.array(X_test)
Y_test = np.array(Y_test)

# predict test images label
y_pred = model.predict(X_test)

# Compute accuracy of trained model on test images
print ("Accuracy: ",np.sum(np.argmax(y_pred, axis=1) == np.argmax(Y_test, axis=1))*100/Y_test.shape[0],"%")

#import pandas as pd
#pd.dataframe({'id': variable_name,'saleprice':variable2_name}).to_csv('test.csv',index=False)
