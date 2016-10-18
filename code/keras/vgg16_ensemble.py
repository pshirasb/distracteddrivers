#!/usr/bin/python2.7
import matplotlib.pyplot as plt
import keras.backend as K
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, MaxoutDense
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.constraints import maxnorm
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.visualize_util import model_to_dot
from keras.utils.visualize_util import plot
import theano.tensor.nnet.abstract_conv as absconv
import pandas as pd
import numpy as np
import time
import os
import cv2
from VGG16 import VGG_16
from myutils import benchmark, ProgressBar, table, confusion_matrix
from dataset import Dataset, KFoldGenerator
HEIGHT = 224 
WIDTH  = 224
BATCH_SIZE = 32 


@benchmark
def data_generators(train_dir, valid_dir, test_dir):
  

    # Train data augmentation
    train_datagen = ImageDataGenerator(
                rotation_range=10,
                width_shift_range=0.1,
                height_shift_range=0.1,
                #shear_range=0.1,
                zoom_range=0.1,
                channel_shift_range=10.0,
                #rescale=1./255,
                fill_mode='reflect')


    # Test data augmentation
    valid_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(WIDTH, HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        shuffle = True)

    valid_generator = valid_datagen.flow_from_directory(
        valid_dir,
        target_size=(WIDTH,HEIGHT),
        batch_size=BATCH_SIZE,
        shuffle = True,
        class_mode='sparse')

    test_datagen = ImageDataGenerator()

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(WIDTH,HEIGHT),
        batch_size=BATCH_SIZE,
        shuffle=False,
        class_mode=None)

     

    return train_generator, valid_generator, test_generator

def custom_generators():
 
    train_dir = "../../data/train"
    test_dir  = "../../data/test"
    driver_csv = "../../data/driver_imgs_list.csv"
    ds = Dataset(train_dir, test_dir, driver_csv)

    train_gen = ds.train_gen(
           batch_size = BATCH_SIZE,
           width  = 224,
           height = 224,
           #scale = 1./255
           )

    valid_gen = ds.valid_gen(
           batch_size = BATCH_SIZE,
           width  = 224,
           height = 224,
           #scale = 1./255
           )

    test_gen = ds.test_gen(
           batch_size = BATCH_SIZE,
           width  = 224,
           height = 224,
           #scale = 1./255
           )

    return train_gen, valid_gen, test_gen
 
   
@benchmark
def create_submission():

    print "Creating submission..." 
    nfold       = 13
    max_count   = 79726
    batch_size  = 32
    upper_count = ((max_count / batch_size) + 1) * batch_size
    iters       = upper_count / batch_size
    assert (upper_count % batch_size == 0)

    model_name = "VGG16_FULL"

    train_dir = "../../data/train"
    test_dir  = "../../data/test"
    driver_csv = "../../data/driver_imgs_list.csv"
    weights_dir = "./kfold_weights"
    ds = Dataset(train_dir, test_dir, driver_csv)

    preds  = np.zeros([nfold, upper_count, 10])
    #names  = np.zeros([nfold, upper_count,  1], dtype = "S30")
    names  = ds.get_test_list()

    for k in xrange(nfold):
        print (">" * 36) + (" [Fold %i] " % (k)) + ("<" * 37)

        # Setup test data generator
        # Note: the reason this is inside the loop is that
        # each run might mess up the internal indices so
        # just to be sure we don't run into problems, we create
        # a new generator for each fold. Although in theory, there 
        # should be no problem reusing the same generator for all folds.
        # BUT! you should keep in mind the labels/image name ordering as well.
        ds = Dataset(train_dir, test_dir, driver_csv)
        test_gen = ds.test_gen(
                  batch_size = batch_size,
                  width  = 224,
                  height = 224
        )
        

        # Build and compile model
        weights_path = os.path.join(
            weights_dir,
            "%s_fold%i_weights.hdf5" % (model_name, k)
        )
        model = build_vgg16_full(0.0, weights_path = weights_path)
        
        pgbar = ProgressBar()
        pgbar.start(iters)
        for i in xrange(iters):
            x = test_gen.next()
            start = (i    ) * batch_size
            end   = (i + 1) * batch_size
            preds[k,start:end] = model.predict_on_batch(x)
            #names[k,start:end,0] = y
            pgbar.step()
        pgbar.stop()
 
    # Discard the extra results from the bottom
    preds = preds[:,:max_count,:]

    # Average all the predictions
    preds = preds.mean(axis = 0)
    assert(preds.shape == (max_count, 10))

    # This might be an issue since we're assuming all the folds
    # predictions are made on exactly the same order of images 
    # so we only take the names from the first fold data
    labels      = names 
    class_names = ["c" + str(x) for x in range(10)]
   
    labels_df = pd.DataFrame({"img" : labels})
    preds_df  = pd.DataFrame(preds, columns=class_names)
    res_df    = pd.concat((labels_df, preds_df), axis = 1)
    
    res_df.to_csv("submission.csv",
                  index = False)

def build_homebrew(learning_rate, trainable = 2, weights_path = None, resume = False):
   
    HEIGHT = 224
    WIDTH  = 224

    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=(3, HEIGHT, WIDTH)))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, 3, 3))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(128, 3, 3))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())  

    model.add(Dense(256, W_constraint = maxnorm(2)))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Dropout(0.5))
    
    model.add(Dense(256, W_constraint = maxnorm(2)))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(Dropout(0.5))
    
    model.add(Dense(10, init="normal", activation='softmax'))
    model.name = "HOMEBREW_CNN"

    if resume:
        model.load_weights(os.path.join(weights_path, "weights_" + model.name + ".hdf5"))
   
    decay = 1.0 / (500)

    # Learning rate is changed to 0.001, decay = 1e-6, momentum = .9, nesterov=True
    sgd = SGD(lr=learning_rate, decay=decay, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy')
    
    print "Model: %s | Resume: %s | Learning Rate: %0.6f" % (model.name, resume, learning_rate)
    return model   
   
def build_vgg16_full(learning_rate, trainable = 2, weights_path = None):
    
    model = VGG_16("./vgg16_weights.h5", full = True, trainable = trainable)
    
    #model.layers.pop()
    #model.layers.pop()
    #model.outputs = [model.layers[-1].output]
    #model.layers[-1].outbound_nodes = []
    model.add(Dense(10, init="normal", activation='softmax'))
    model.name = "VGG16_FULL"

    decay = 1.0 / (500)

    if weights_path is not None:
        model.load_weights(weights_path)

    # Learning rate is changed to 0.001, decay = 1e-6, momentum = .9, nesterov=True
    sgd = SGD(lr=learning_rate, decay=decay, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy')
    
    print "Model: %s | Learning Rate: %0.6f" % (model.name, learning_rate)
    return model   
 
def build_vgg16_custom(learning_rate, trainable = 2, weights_path = None, resume = False):
    
    model = VGG_16("./vgg16_weights.h5", full = False, trainable = trainable)
    
    model.add(MaxPooling2D((2,2), strides=(2,2), trainable = False))
    model.add(Flatten())
    model.add(MaxoutDense(32))
    #model.add(Dense(2048, init="normal", activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxoutDense(32))
    #model.add(Dense(2048, init="normal", activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, init="normal", activation='softmax'))

    model.name = "VGG16_CUSTOM"

    if resume:
        model.load_weights(os.path.join(weights_path, "weights_" + model.name + ".hdf5"))

    # Learning rate is changed to 0.001, decay = 1e-6
    sgd = SGD(lr=learning_rate, decay=0, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy')
    
    print "Model: %s | Resume: %s | Learning Rate: %0.6f" % (model.name, resume, learning_rate)
    return model   
 
def build_vgg16_cam(learning_rate = 0.01, num_input_channels = 1024, trainable = 2, weights_path = None, resume = False):
   
    model = VGG_16("./vgg16_weights.h5", full = False, trainable = trainable)

    # Add another conv layer with ReLU + GAP
    model.add(Convolution2D(2048, 3, 3, activation='relu', border_mode="same"))
    model.add(AveragePooling2D((14, 14), strides = (14,14)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    
    # Add the W layer
    model.add(Dense(10, activation='softmax'))

    model.name = "VGG16_CAM"

    if resume:
        model.load_weights(os.path.join(weights_path, "weights_" + model.name + ".hdf5"))

    decay = 1 / 40.0
    decay = 0.0001
    # Or
    #decay = learning_rate / (50.0 * 100.0)
    # Or
    #decay = 1 / (50.0 * 100.0)
    
    sgd = SGD(lr=learning_rate, decay=decay, momentum=0.9, nesterov=True)
    adam = Adam(lr=learning_rate)
    model.compile(optimizer=adam, loss='sparse_categorical_crossentropy')

    print "Model: %s | Resume: %s | Learning Rate: %0.6f" % (model.name, resume, learning_rate)
    return model


#@benchmark
def train(model, weights_path = None, resume = False):

    samples_per_epoch = 100
    nb_epoch=50
    nb_valid = (5820 / BATCH_SIZE) / 10

    train_gen, valid_gen, test_gen = data_generators(
        "../../data/train_train",
        "../../data/train_valid",
        "../../data/test_keras")

    mean_pixel = np.array([103.939, 116.779, 123.68]) 

    losses = []
    val_lossess = []
    best_val_loss = np.Inf

    pgbar = ProgressBar()
    for epoch in range(nb_epoch):
        pgbar.start(samples_per_epoch)
        for i in range(samples_per_epoch):
            x, y  = train_gen.next()
            #center around mean for each channel
            for c in range(3):
                x[:, c, :, :] = x[:, c, :, :] - mean_pixel[c]

            # VGG is trained in BGR, so we need to convert from RGB to BGR
            x = x[:,[2,1,0],:,:]

            loss = model.train_on_batch(x, y)
            pgbar.step(pretxt = ("[%2d/%2d] " % (epoch + 1, nb_epoch)),
                       txt = ("loss: %.4f" % loss))
        pgbar.stop()
    
        # Get current learning rate
        opt = model.optimizer
        exact_lr = opt.lr.get_value() * (1.0 / (1.0 + opt.decay.get_value() * opt.iterations.get_value()))
        print "[%i] Learning Rate %0.6f" % (opt.iterations.get_value(), exact_lr)

        # Epoch Validation
        val_loss = 0.0
        for j in range(nb_valid):
            x, y  = valid_gen.next()
            #center around mean for each channel
            for c in range(3):
                x[:, c, :, :] = x[:, c, :, :] - mean_pixel[c]

            # VGG is trained in BGR, so we need to convert from RGB to BGR
            x = x[:,[2,1,0],:,:]
            val_loss   += model.test_on_batch(x, y)

        val_loss = val_loss / nb_valid
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_weights(os.path.join(weights_path, "weights_" + model.name + ".hdf5"), overwrite=True)
            print "[  +  ] val_loss: %.4f - " % (val_loss),
            print "Model weights saved: weights_" + model.name + ".hdf5"
        else:
            print "[     ] val_loss: %.4f - " % (val_loss),
            print "val_loss did not improve."

        losses.append(loss)
        val_lossess.append(val_loss)
   
    _, valid_gen, test_gen = custom_generators()
    run_validation(model, valid_gen)
    model.save_weights(os.path.join(weights_path, "weights_" + model.name + "_final.hdf5"), overwrite=True)
    #create_submission(model, test_gen)

    return losses, val_lossess


    
def train_kfold():

    # Params
    lr    = 0.0002
    nfold = 13
    nb_epoch = 15
    batch_size = 32
    samples_per_epoch = batch_size * 600
    nb_val = 1600
    width  = 224
    height = 224
    train_dir    = "/home/boss/kaggle/dashcam/data/train"
    test_dir     = "/home/boss/kaggle/dashcam/data/test"
    driver_csv   = "/home/boss/kaggle/dashcam/data/driver_imgs_list.csv"
    weights_path = './kfold_weights'
    resume    = False
    FREEZE_ALL  = 0
    FREEZE_CONV = 1
    FREEZE_NONE = 2
    trainable = FREEZE_NONE


    # Vars
    val_losses = 0.0
    fg = KFoldGenerator(train_dir = train_dir,
                        test_dir  = test_dir,
                        driver_csv = driver_csv,
                        nfold = nfold,
                        batch_size = batch_size,
                        width = width,
                        height = height)
    # Training loop
    for k in range(nfold):

        print (">" * 36) + (" [Fold %i] " % (k)) + ("<" * 37)

        # Build and compile model
        model = build_vgg16_full(lr, trainable = trainable)

        # Data generators for this fold
        train_gen = fg.train_gen(k)
        valid_gen = fg.valid_gen(k)

        # Train
        model.fit_generator(
            train_gen,
            samples_per_epoch=samples_per_epoch,
            nb_epoch=nb_epoch,
            validation_data = valid_gen,
            nb_val_samples = nb_val,
            max_q_size = 10)
        
        # Save Weights for this model
        model.save_weights(
            os.path.join(
                weights_path, 
                "%s_fold%i_weights.hdf5" % (model.name, k)
            ),
            overwrite=True
        )
        
        # Validation
        val_loss = model.evaluate_generator(
                                valid_gen,
                                val_samples = 3840)
        print "Fold [%i] loss: %0.6f" % (k, val_loss)
        val_losses += val_loss
    
        #print (">" * 35)  +  "[ End ]"  + ("<" * 35)
    
    val_losses = val_losses / float(nfold)
    print ">>> Training Complete. Avg Loss: %0.6f" % (val_losses)

def run_validation(model, valid_gen):

    max_count = 5820 / BATCH_SIZE

    preds  = np.zeros([0, 10]) 
    labels = np.zeros([0, 1])

    total_loss = 0

    pgbar = ProgressBar()
    pgbar.start(max_count)
    for i in xrange(max_count):
        x, y = valid_gen.next()
        pred   = model.predict_on_batch(x)
        loss   = model.test_on_batch(x, y)
        total_loss += loss
        preds  = np.vstack((preds, pred))
        labels = np.vstack((labels, y))
        
        pgbar.step(txt = ("loss: %.4f" % loss))
    pgbar.stop()

    print(confusion_matrix(np.argmax(preds, axis = 1),
            np.squeeze(labels).astype(np.int)))

    print "Average Loss: %.4f" % (total_loss / float(max_count))

    #print model.metrics_names
    #print model.evaluate_generator(
        #valid_gen,
        #val_samples = 5820)

def plotLossHist(loss, label, color = "blue"):
    x   = range(len(loss))
    y   = loss
    fit = np.polyfit(x, y, 1)
    fit_fn = np.poly1d(fit) 
    
    l,   = plt.plot(x, y, "o-", label = label, color = color)
    l_2, = plt.plot(x, fit_fn(x), "--", color = color)
    return l


def main():

    '''
    lr notes:
    0.001 only top:   around 2+ loss
    0.0001 only top:  3+ val_loss
    0.00001 only top: below val_loss 1.0, seems to be the sweet spot
    p.s. only valid with decay+momentum
    '''

    train_kfold()
    create_submission()

if __name__ == '__main__':
    main()

