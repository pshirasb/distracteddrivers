#!/usr/bin/python2

import os
import csv
import cv2
import random
import numpy as np
from myutils import ProgressBar, benchmark

class Dataset:

    def __init__(self, 
                 train_dir, 
                 test_dir, 
                 driver_csv, 
                 train_drv_list = None, 
                 valid_drv_list = None):
        """
        Args:
            train_dir: train directory with images insides folders named after
                       their labels
            test_dir:  test directory with images and no subdirectories

            driver_csv: the path to "driver_imgs_list.csv"
        """
                
        self._train_dir  = train_dir
        self._test_dir   = test_dir
        self._driver_csv = driver_csv
        self._train_size = 0
        self._valid_size = 0
        self._test_size  = 0
        self._train_list = []
        self._valid_list = []
        self._test_list  = []
        self._train_index = 0
        self._valid_index = 0
        self._test_index  = 0
        self._train_drv_list = train_drv_list
        self._valid_drv_list = valid_drv_list
        self._create_train_and_valid_list()
        self._create_test_list()


    def train_gen(self, 
                  batch_size,
                  width, 
                  height, 
                  scale = 1):
        """ training data batch generator
        Args:
            batch_size:     int
                number of samples to return
            
            width, height:  int, int
                image dimensions
            
            scale:          float32
                scaling factor, each image is multiplied by this value.

                
        Returns:
            x:  image data, numpy array of shape 
                    [batch_size, channels (BGR), height, width]
            y:  sparse labels, numpy array of shape
                    [batch_size, 1]
        """
        while(1):
            x = np.zeros([batch_size, 3, height, width], dtype = np.float32)
            y = np.zeros([batch_size, 1], dtype = np.int8)

            for i in xrange(batch_size):
                #retn = "%s  %s" % (self._train_list[self._train_index][0],
                                  #self._train_list[self._train_index][1])
                
                # Read the image in BGR and resize it
                im = cv2.imread(self._train_list[self._train_index][0])
                im = cv2.resize(im, (width, height)).astype(np.float32)
                
                # Subtract mean per channel
                im[:,:,0] -= 103.939
                im[:,:,1] -= 116.779
                im[:,:,2] -= 123.68

                # Scale
                im = im * scale

                # Reorder the dimensions
                im = im.transpose((2,0,1))

                # append to return array
                x[i] = im
                y[i] = np.int(self._train_list[self._train_index][1])

                self._train_index += 1
                if self._train_index >= len(self._train_list):
                    self._train_index = 0
                    random.shuffle(self._train_list)
                    #self._epoch += 1
                    
            #y = np_utils.to_categorical(y, 10) 
            yield (x, y)

    def valid_gen(self, 
                  batch_size,
                  width, 
                  height, 
                  scale = 1):
        """ validation data batch generator
        Args:
            batch_size:     int
                number of samples to return
            
            width, height:  int, int
                image dimensions
            
            scale:          float32
                scaling factor, each image is multiplied by this value.

                
        Returns:
            x:  image data, numpy array of shape 
                    [batch_size, channels (BGR), height, width]
            y:  sparse labels, numpy array of shape
                    [batch_size, 1]
        """
        while(1):
            x = np.zeros([batch_size, 3, height, width], dtype = np.float32)
            y = np.zeros([batch_size, 1], dtype = np.int8)

            for i in xrange(batch_size):
                #retn = "%s  %s" % (self._valid_list[self._valid_index][0],
                                  #self._valid_list[self._valid_index][1])
                
                # Read the image in BGR and resize it
                im = cv2.imread(self._valid_list[self._valid_index][0])
                im = cv2.resize(im, (width, height)).astype(np.float32)
                
                # Subtract mean per channel
                im[:,:,0] -= 103.939
                im[:,:,1] -= 116.779
                im[:,:,2] -= 123.68

                # Scale
                im = im * scale

                # Reorder the dimensions
                im = im.transpose((2,0,1))

                # append to return array
                x[i] = im
                y[i] = np.int(self._valid_list[self._valid_index][1])

                self._valid_index += 1
                if self._valid_index >= len(self._valid_list):
                    self._valid_index = 0
                    random.shuffle(self._valid_list)
                    #self._epoch += 1
                    
            
            #y = np_utils.to_categorical(y, 10) 
            yield (x, y)
        
    def test_gen(self, 
              batch_size,
              width, 
              height, 
              scale = 1):
        """ test data batch generator
        Args:
            batch_size:     int
                number of samples to return
            
            width, height:  int, int
                image dimensions
            
            scale:          float32
                scaling factor, each image is multiplied by this value.

                
        Returns:
            x:  image data, numpy array of shape 
                    [batch_size, channels (BGR), height, width]
        """
        while(1):
            x = np.zeros([batch_size, 3, height, width], dtype = np.float32)
            y = []

            for i in xrange(batch_size):
                
                img_name = self._test_list[self._test_index]
                img_path = os.path.join(self._test_dir, img_name)

                # Read the image in BGR and resize it
                im = cv2.imread(img_path)
                im = cv2.resize(im, (width, height)).astype(np.float32)
                
                # Subtract mean per channel
                im[:,:,0] -= 103.939
                im[:,:,1] -= 116.779
                im[:,:,2] -= 123.68

                # Scale
                im = im * scale

                # Reorder the dimensions
                im = im.transpose((2,0,1))

                # append to return array
                x[i] = im
                y.append(img_name)

                self._test_index += 1
                if self._test_index >= len(self._test_list):
                    self._test_index = 0
                    #self._epoch += 1
                    
            
            #y = np_utils.to_categorical(y, 10) 
            #y = np.array(y).transpose()

            yield x
    def get_test_list(self):
        return self._test_list

    def _create_train_and_valid_list(self):
        # list of drivers for validation
        
        if self._valid_drv_list is None:
            valid_drv_list = ['p021', 'p022', 'p024', 'p026', 'p042', 'p072']
        else: 
            valid_drv_list = self._valid_drv_list

        if self._train_drv_list is None:
            train_drv_list = [
                "p002","p012","p014","p015","p016","p035","p039","p041","p045","p047",
                "p049","p050","p051","p052","p056","p061","p064","p066","p075","p081"
            ]
        else:
            train_drv_list = self._train_drv_list
       
        train_dir  = self._train_dir 
        driver_csv = self._driver_csv

        if not os.path.exists(train_dir):
            raise "train_dir does not exist!"
        if not os.path.exists(driver_csv):
            raise "driver_csv does not exist!"

        train_list = []
        valid_list = []

        with open(driver_csv, 'rb') as f:
            reader = csv.reader(f, delimiter=",")
            # Skip header
            next(reader)
            for row in reader:
                # csv format: (driverID, label, img_name)
                img_path  = os.path.join(train_dir, row[1], row[2])

                if row[0] in train_drv_list:
                    # goes to training set
                    train_list.append([img_path, row[1][1]])
                elif row[0] in valid_drv_list:
                    # goes to validation set
                    valid_list.append([img_path, row[1][1]])

        random.shuffle(train_list)
        random.shuffle(valid_list)

        print "%i training images, %i validation images." % (
                    len(train_list), len(valid_list))
        
        self._train_size = len(train_list)
        self._valid_size = len(valid_list)
        self._train_list = train_list
        self._valid_list = valid_list

    def _create_test_list(self):
               
        test_dir  = self._test_dir 

        if not os.path.exists(test_dir):
            raise "train_dir does not exist!"

        test_list = []

        for f in os.listdir(test_dir):
            test_list.append(f)

        self._test_size = len(test_list)
        self._test_list = test_list

        print "%i testing images." % self._test_size


def labelToText(label):
    
    lmap = [
            "normal driving",               # 0
            "texting - right",              # 1
            "talking on the phone - right", # 2
            "texting - left",               # 3
            "talking on the phone - left",  # 4
            "operating the radio",          # 5
            "drinking",                     # 6
            "reaching behind",              # 7
            "hair and makeup",              # 8
            "talking to passenger"          # 9
    ]
    return lmap[label]

class KFoldGenerator:

    def __init__ (self,
                  train_dir,
                  test_dir,
                  driver_csv,
                  nfold,
                  batch_size,
                  width,
                  height):
        
        self._train_dir  = train_dir
        self._test_dir   = test_dir
        self._driver_csv = driver_csv
        self._nfold      = nfold
        self._dataset    = []
        self._batch_size = batch_size
        self._width      = width
        self._height     = height
        self._create_folds()

    def _create_folds(self):
        
        nfold = self._nfold
        drv_list = np.array([
            "p002","p012","p014","p015","p016","p021","p022","p024",
            "p026","p035","p039","p041","p042","p045","p047","p049",
            "p050","p051","p052","p056","p061","p064","p066","p072",
            "p075","p081"
        ])

        ind = range(len(drv_list))
        random.shuffle(ind)
        if len(ind) % nfold > 0:
            ind = ind[:-(len(ind) % nfold)]
        drv_list = drv_list[ind]

        fold_size = len(ind) / nfold

        for k in range(nfold):
            
            start = (k    ) * fold_size
            end   = (k + 1) * fold_size
            valid_list = drv_list[start:end]
            train_list = np.setdiff1d(drv_list, valid_list)
            self._dataset.append(
                Dataset(self._train_dir,
                        self._test_dir,
                        self._driver_csv,
                        train_list,
                        valid_list
                )
            )            
    
    def train_gen(self, fold_num):
        if fold_num >= self._nfold:
            raise ValueError("fold_num is out of bounds!")
        return self._dataset[fold_num].train_gen(
                   batch_size = self._batch_size,
                   width  = self._width,
                   height = self._height)
 
    def valid_gen(self, fold_num):
        if fold_num >= self._nfold:
            raise ValueError("fold_num is out of bounds!")
        return self._dataset[fold_num].valid_gen(
                   batch_size = self._batch_size,
                   width  = self._width,
                   height = self._height)
                   


def test_dataset():
    train_dir  = "../../data/train"
    test_dir   = "../../data/test"
    driver_csv = "../../data/driver_imgs_list.csv"

    ds = Dataset(train_dir, test_dir, driver_csv)
    test_gen = ds.test_gen(batch_size = 5,
                       width  = 224,
                       height = 224,
                       scale = 1./255)

    count = 0
        
    for x, y in test_gen:
        count += 1
        if count > 10:
            break

        im = x[0]
        
        im = im.transpose((1,2,0))
        
        im = im * 255.0

        im[:,:,0] += 103.939
        im[:,:,1] += 116.779
        im[:,:,2] += 123.68
       
        cv2.imshow(y, im.astype(np.uint8))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def test_kfold(): 
    train_dir    = "../../data/train"
    test_dir     = "../../data/test"
    driver_csv   = "../../data/driver_imgs_list.csv"
    nfold = 5

    fg = KFoldGenerator(train_dir = train_dir,
                    test_dir  = test_dir,
                    driver_csv = driver_csv,
                    nfold = nfold,
                    batch_size = 32,
                    width = 224,
                    height = 224)

    return fg 


