# ML-project
# Identify signs of diabetic retinopathy in eye images
Author: Iqra Imran 
### About DataSet
  - we take dataset of images from kaggle competition platform which was comprised of 64Gb sized dataset we got dataset with the help of IT Deparment of University of Central Punjab. We take 16526 images out of avaible datset on kaggle.
  - We got a large set of high-resolution retina images taken under a variety of imaging conditions. A left and right field is provided for every subject. Images are labeled with a subject id as well as either left or right (e.g. 1_left.jpeg is the left eye of patient id 1).
A learned machine is rating the presence of diabetic retinopathy in each image on a scale of 0 to 4, according to the following scale:

    0 - No DR
1 - Mild
2 - Moderate
3 - Severe
4 - Proliferative DR we take 16526 images out of total images available. 

#### Zipped file contain the following file 
- iml-proj- it is containig python code 
- test_data- containig 16526 images dataset with value 
- train_data- containing 40 images dataset with values
- model-containing saved model  

### How to make Test-data file 
we placed a folder named Images (containing 16526 images) in the same folder where we places 'iml-proj.py' file and then we take their paths in an excel file by concatinating 3 columns,1st column contain the name of the folder which is "Images" in our case ,2nd column contain the names of image that we get simply by copy and pasting of names of images which is already available on kaggle, 3rd column contain the format of the images which is "Jpeg"in our case. After concatination we go One column with is containg the path of the images.
After that we placed values of images another column. We got 2 column now. 

### How to make Train-data file
we take sample of 40 dataset from test-data

### Tool and language 

  - We are using Anaconda tool 
  - we use Python Language with following pacakges
  0 - tensorflow
1 - tflearn
2 - h5py
3 - hdf5
4 - SciPy

### How to Download Packages 

Tensorflow

If you have anaconda installed, then installing tensorflow and tflearn is straightforward. To install tensorflow, activate your conda environment and run the following command.

    conda install tensorflow
    or
    conda install -c conda-forge tensorflow

Press 'y' when asked for permission. This will install tensorflow and all required packages.
tflearn

To install tflearn run the following command in Ubuntu. 'pip' should already be installed on your system.

    pip install tflearn

Other packages

You can directly install other packages using anaconda.

    conda install package_name

### For Windows

In order to install these packages in anaconda on Windows systems, we first create a new environment with python 3.5 using anaconda navigator. Now we launched our environment in console or we simply launched anaconda console and activate your environment using command 'activate env_name'. To install tensorflow and tflearn we can used commands given below.

conda create -n my_env python=3.5  # run this if you did not create environment using anaconda gui.
    conda install -c conda-forge tensorflow
    pip install tflearn

### Algorithm 
We are using convolutional neural networks with the following layers 

network = input_data(shape=[None, 100, 100, 3],data_preprocessing=img_prep,data_augmentation=img_aug)network = conv_2d(network,63, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)
network = fully_connected(network, 256, activation='relu')
network = dropout(network, 0.75)
network = fully_connected(network, 256, activation='relu')
network = dropout(network, 0.75)
network = fully_connected(network, 5, activation='softmax')
network = regression(network, optimizer='adam',loss='categorical_crossentropy',learning_rate=0.01)
 
### Results 
Accuracy of our model is 75%.
