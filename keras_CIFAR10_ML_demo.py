# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 14:27:54 2018

This script is an example for training a CNN model using keras, the script accepts
epochs, learning_rate and batch as argument.
Default values:
    epochs: 5
    learning_rate: 0.001
    batch: 32
How use this script:

python keras_CIFAR10_ML_demo.py --epochs <ep> --learning_rate <lr> --batch <batch_size> --path <path_to_dataset>

@author: jaydeep.deka
"""
import argparse
import numpy as np
import os
import sys
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from six.moves import cPickle
#from matplotlib import pyplot


def get_arguments():
    """
    Description: This function can be used to get arguments from the user that
    are planning in the text box.
    Args: None
    Returns: Argument dictionary
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch', type=int, default=32, help='No of images in each batch.')
    parser.add_argument('--path', type=str, default=r'/home/jaydeepdeka/project_hitachi/cifar-10-batches-py', help='Location of the dataset')
    parser.add_argument('--outputpath', type=str, default=r'/home/jaydeepdeka/project_hitachi/', help='Location to store the output')
    return parser.parse_args()

def loadData(path):
    """
    This function is the copy of cifar10.load_data() available.
    will take the path of the dataset set locally avaiable and returns the training and testing datasets
    splitted.
    
    """
    num_train_samples = 50000

    x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.empty((num_train_samples,), dtype='uint8')

    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i))
        (x_train[(i - 1) * 10000: i * 10000, :, :, :],
         y_train[(i - 1) * 10000: i * 10000]) = load_batch(fpath)

    fpath = os.path.join(path, 'test_batch')
    x_test, y_test = load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    if K.image_data_format() == 'channels_last':
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

    return (x_train, y_train), (x_test, y_test)

def preProcessData(X_train, y_train, X_test, y_test):
    """
    Description: This function will load data for us using the available API for our example
    Preprocess the data normalisation
    Args: None
    Returns: Processed datasets
    """
    
    # Step1: Normalise the data
    # normalize inputs from 0-255 to 0.0-1.0
    X_train = X_train.astype('float32')/255.0
    X_test = X_test.astype('float32')/ 255.0

    # Step2: One hot encode outputs
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    
    return X_train, X_test, y_train, y_test

def createModel(num_classes):
    """
    Description: This function is used to create a CNN model.
    Args: num_classes
    Returns: model
    """
    # Create the model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def train(X_train, y_train, X_test, y_test, model, batch_size, epochs, lr):
    """
    Description: This function can be used to train the model
    Args: X_train, y_train, X_test, y_test, model, batch_size, epochs, lr
    Returns: model
    """
    decay = lr/epochs

    # Define the optimiser
    sgd = SGD(lr=lr, momentum=0.9, decay=decay, nesterov=False)

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    print(model.summary())

    # Fit the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size)
    return model

def saveTheCheckpoint(outputpath, model):
    """
    Description: This function can be used to train the model
    Args: model
    Returns: None
    """
    model_json = model.to_json()
    jsonpath = os.path.join(outputpath, "model.json")
    with open(jsonpath, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    h5path = os.path.join(outputpath, "model.h5")
    model.save_weights(h5path)
    print("Saved model to disk")

def main():
    args = get_arguments()
    epochs = args.epochs
    lr = args.learning_rate
    batch_size = args.batch
    path = args.path
    outputpath = args.outputpath

    print("Received data from the user learning_rate={}, epochs={}, batch_size={}".format(lr,epochs,batch_size))
    # Step1: Loading the data
    (X_train, y_train), (X_test, y_test) = loadData(path)
    # Load and preprocess the data
    X_train, X_test, y_train, y_test = preProcessData(X_train, y_train, X_test, y_test)
    
    # Optional: Show the images in 3x3 grid
    # create a grid of 3x3 images
#    for i in range(0, 9):
#    	pyplot.subplot(330 + 1 + i)
#    	pyplot.imshow(toimage(X_train[i]))
#    # show the plot
#    pyplot.show()
    
    # Create the model
    model = createModel(num_classes=y_test.shape[1])
    
    # Train the model
    model = train(X_train, y_train, X_test, y_test, model, batch_size, epochs, lr)
    
    # Save the model
    saveTheCheckpoint(outputpath, model)

def load_batch(fpath, label_key='labels'):
    """Internal utility for parsing CIFAR data.
    # Arguments
        fpath: path the file to parse.
        label_key: key for label data in the retrieve
            dictionary.
    # Returns
        A tuple `(data, labels)`.
    """
    with open(fpath, 'rb') as f:
        if sys.version_info < (3,):
            d = cPickle.load(f)
        else:
            d = cPickle.load(f, encoding='bytes')
            # decode utf8
            d_decoded = {}
            for k, v in d.items():
                d_decoded[k.decode('utf8')] = v
            d = d_decoded
    data = d['data']
    labels = d[label_key]

    data = data.reshape(data.shape[0], 3, 32, 32)
    return data, labels
    
if __name__=='__main__':
    main()
