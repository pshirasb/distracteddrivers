#!/usr/bin/python2

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.constraints import maxnorm
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD

import cv2, numpy as np
from myutils import benchmark
import os
import h5py

def VGG_16_BN_PRELU(weights_path=None, full = True, trainable = 0):

    # trainable:    0 - freeze all layers
    #               1 - freeze conv layers only                    
    #               2 - all layers are trainable

    if trainable == 0:
        trainable_1 = False
        trainable_2 = False
    elif trainable == 1:
        trainable_1 = False
        trainable_2 = True
    else:
        trainable_1 = True
        trainable_2 = True


    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Convolution2D(64, 3, 3, trainable=trainable_1))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, trainable=trainable_1))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', trainable=trainable_1))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', trainable=trainable_1))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', trainable=trainable_1))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', trainable=trainable_1))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', trainable=trainable_1))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', trainable=trainable_1))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', trainable=trainable_1))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', trainable=trainable_1))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', trainable=trainable_1))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', trainable=trainable_1))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', trainable=trainable_1))


    # Build all the way through or only up to the bottleneck layer?
    if full:
        model.add(MaxPooling2D((2,2), strides=(2,2)))
        model.add(Flatten())
        model.add(Dense(4096, activation='relu', 
                  W_constraint = maxnorm(2), trainable=trainable_2))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu', 
                  W_constraint = maxnorm(2), trainable=trainable_2))
        model.add(Dropout(0.5))
        #model.add(Dense(1000, activation='softmax'))

    #model.load_weights(weights_path)

    # Load the weights but only up to layers built
    if weights_path:
        assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
        f = h5py.File(weights_path)
        for k in range(f.attrs['nb_layers']):
            if k >= len(model.layers):
                # we don't look at the last (fully-connected) layers 
                # in the savefile
                break
            g = f['layer_{}'.format(k)]
            weights = [g['param_{}'.format(p)] for p in 
                      range(g.attrs['nb_params'])]
            model.layers[k].set_weights(weights)
        f.close()
        #print('Model loaded.')

    return model

def VGG_16(weights_path=None, full = True, trainable = 0):

    # trainable:    0 - freeze all layers
    #               1 - freeze conv layers only                    
    #               2 - all layers are trainable

    if trainable == 0:
        trainable_1 = False
        trainable_2 = False
    elif trainable == 1:
        trainable_1 = False
        trainable_2 = True
    else:
        trainable_1 = True
        trainable_2 = True


    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Convolution2D(64, 3, 3, activation='relu', trainable=trainable_1))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', trainable=trainable_1))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', trainable=trainable_1))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', trainable=trainable_1))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', trainable=trainable_1))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', trainable=trainable_1))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', trainable=trainable_1))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', trainable=trainable_1))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', trainable=trainable_1))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', trainable=trainable_1))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', trainable=trainable_1))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', trainable=trainable_1))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', trainable=trainable_1))


    # Build all the way through or only up to the bottleneck layer?
    if full:
        model.add(MaxPooling2D((2,2), strides=(2,2)))
        model.add(Flatten())
        model.add(Dense(4096, activation='relu', 
                  W_constraint = maxnorm(2), trainable=trainable_2))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu', 
                  W_constraint = maxnorm(2), trainable=trainable_2))
        model.add(Dropout(0.5))
        #model.add(Dense(1000, activation='softmax'))

    #model.load_weights(weights_path)

    # Load the weights but only up to layers built
    if weights_path:
        assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
        f = h5py.File(weights_path)
        for k in range(f.attrs['nb_layers']):
            if k >= len(model.layers):
                # we don't look at the last (fully-connected) layers 
                # in the savefile
                break
            g = f['layer_{}'.format(k)]
            weights = [g['param_{}'.format(p)] for p in 
                      range(g.attrs['nb_params'])]
            model.layers[k].set_weights(weights)
        f.close()
        print('Model loaded.')

    return model


