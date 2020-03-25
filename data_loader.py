#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import os
import numpy as np
from random import shuffle
import pandas as pd
from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Activation, Dropout, Dense, Embedding, BatchNormalization, SpatialDropout2D, Input, Concatenate, UpSampling2D, LSTM, Bidirectional
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.losses import binary_crossentropy,categorical_crossentropy
import random
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img
from sklearn.preprocessing import Binarizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from sklearn.model_selection import KFold
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from sklearn.preprocessing import MinMaxScaler
import nibabel as nib
from tensorflow.keras.utils import Sequence


def data_prep_task1(timesteps=60):

    T=timesteps

    # load csv files:
    dataset_train = pd.read_csv('/Lab1/Lab5/train_data_stock.csv')
    dataset_val = pd.read_csv('/Lab1/Lab5/val_data_stock.csv')

    # reverse data so that they go from oldest to newest:
    dataset_train = dataset_train.iloc[::-1] #taking all the dataset and reversing it starting from last row
    dataset_val = dataset_val.iloc[::-1] #taking all the dataset and reversing it starting from last row

    # concatenate training and test datasets:
    dataset_total = pd.concat((dataset_train['Open'], dataset_val['Open']), axis=0)

    # select the values from the “Open” column as the variables to be predicted:
    training_set = dataset_train.iloc[:, 1:2].values
    val_set = dataset_val.iloc[:, 1:2].values

    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(training_set)

    # split training data into T time steps:
    X_train = []
    y_train = []
    for i in range(T, len(training_set)):
        X_train.append(training_set_scaled[i-T:i, 0])
        y_train.append(training_set_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    # normalize the validation set according to the normalization applied to the training set:
    inputs = dataset_total[len(dataset_total) - len(dataset_val) - 60:].values
    inputs = inputs.reshape(-1, 1)
    inputs = sc.transform(inputs)
    # split validation data into T time steps:
    X_val = []

    for i in range(T, T + len(val_set)):
        X_val.append(inputs[i-T:i, 0])
    X_val = np.array(X_val)
    y_val = sc.transform(val_set)
    # reshape to 3D array (format needed by LSTMs -> number of samples, timesteps, input dimension)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))
    return X_train, y_train, X_val, y_val


def load_streamlines(dataPath, subject_ids, bundles, n_tracts_per_bundle):
    X = []
    y = []
    for i in range(len(subject_ids)):
        for c in range((len(bundles))):
            filename = dataPath + subject_ids[i] + '/' + bundles[c] + '.trk'
            tfile = nib.streamlines.load(filename)
            streamlines = tfile.streamlines
            n_tracts_total = len(streamlines)
            ix_tracts = np.random.choice(range(n_tracts_total),
            n_tracts_per_bundle, replace=False)
            streamlines_data = streamlines.data
            streamlines_offsets = streamlines._offsets
            for j in range(n_tracts_per_bundle):
                ix_j = ix_tracts[j]
                offset_start = streamlines_offsets[ix_j]
                if ix_j < (n_tracts_total - 1):
                    offset_end = streamlines_offsets[ix_j + 1]
                    streamline_j = streamlines_data[offset_start:offset_end]
                else:
                    streamline_j = streamlines_data[offset_start:]
                X.append(np.asarray(streamline_j))
                y.append(c)
    return X, y


class MyBatchGenerator(Sequence):
    def __init__(self, X, y, batch_size=1, shuffle=True):
    #'Initialization'
        self.X = X

        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
    def __len__(self):
    #'Denotes the number of batches per epoch'
        return int(np.floor(len(self.y)/self.batch_size))
    def __getitem__(self, index):
        return self.__data_generation(index)
    def on_epoch_end(self):
    #'Shuffles indexes after each epoch'
        self.indexes = np.arange(len(self.y))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    def __data_generation(self, index):
        Xb = np.empty((self.batch_size, *self.X[index].shape))
        yb = np.empty((self.batch_size, 1))
    # naively use the same sample over and over again
        for s in range(0, self.batch_size):
            Xb[s] = self.X[index]
            yb[s] = self.y[index]
        return Xb, yb
    
    
def test_train_data(img_path,mask_path,img_w=256,img_h=256):
    data_img = [] 
    data_mask = []
    
    
    combined = list(zip(os.listdir(img_path),os.listdir(mask_path)))
    random.shuffle(combined)
    
    image_files=os.listdir(img_path)
    mask_files=os.listdir(mask_path)
    for i in range(500):
            image_name=image_files[i]
            mask_ind=mask_files.index(image_name[0:-4]+'_Tumor.png')
            mask_name=mask_files[mask_ind]

            img = imread(os.path.join(img_path, image_name), as_grey=True)
            img = resize(img, (img_h, img_w), anti_aliasing = True).astype('float32')
            data_img.append(img)

            mask = imread(os.path.join(mask_path, mask_name), as_grey=True)
            mask = resize(mask, (img_h, img_w), anti_aliasing = True).astype('float32')
            data_mask.append(mask)

            if i % 200 == 0:
                 print('Reading: {0}/{1}  of images/masks'.format(i, len(combined)))
            
            data_mask_array=np.array(data_mask)
            data_mask_array[data_mask_array==0]=0
            data_mask_array[data_mask_array>0]=1
    # Convert into array
    data_img=np.array(data_img)
    
    # Expand dimensions
    data_img=np.expand_dims(data_img,axis=-1)
    data_mask_array=np.expand_dims(data_mask_array,axis=-1)

    return data_img,data_mask_array



    
    
    
    
    
    
    
    
    
    


