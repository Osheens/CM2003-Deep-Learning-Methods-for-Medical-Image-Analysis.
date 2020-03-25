#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
import keras as keras
import time
import sys
import csv
import cv2
import matplotlib.pyplot as plt
import scipy as sp
from keras.models import Model
from keras import backend as K
import scipy
    


from collections import defaultdict
from skimage.io import imread
from skimage.transform import resize
from data_loader import *
from models import *
from plot_results import *

args = sys.argv[1]
print(args)

# Paths to the files with training, and validation sets.
# Each file contains pairs (path to image, output vector)
train_path = '../../dl_data/CheXpert-v1.0-small/train.csv'
val_path = '../../dl_data/CheXpert-v1.0-small/valid.csv'

train_dir = '../../dl_data/CheXpert-v1.0-small/train/'
val_dir = '../../dl_data/CheXpert-v1.0-small/valid/'


# Class names
our_targets = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']


# Load data
train_data = pd.read_csv(train_path)
val_data = pd.read_csv(val_path)


# Preprocessing csv data
labels_list, label_2_idx, idx_2_label = preproc_data(train_data, val_data)


# Getting images (array lists: train_img, val_img and different arrays: train_images, val_images)
img_h, img_w= 240,240

train_img, train_images = get_train_data(train_data, img_h, img_w)
val_img, val_images = get_val_data(val_data, img_h, img_w)


# Getting labels (array lists and different arrays)
Train_labels, Val_labels, train_labels, val_labels = get_labels(train_data, val_data, train_img, val_img, label_2_idx)


#Hyperparameters
n_classes = len(label_2_idx)
batch_size = 8
n_epoch = 3


# model calling
if args == "cnn":

    # call model
    model = cnn5_model(Base = 8, filter_size = 3, input_shape=(240,240,1), batch_norm=True, spatial_drop=True, spatial_drop_r = 0.1)
    # compiling
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()

    # setting an early stop
    early_stop = EarlyStopping(monitor='loss', patience=5, verbose=1)

    #fitting
    history_cnn = model.fit(train_img, Train_labels,
                            batch_size = batch_size,
                            epochs = n_epoch,
                            verbose = 1,
                            validation_split = 0.2,
                            callbacks = [early_stop])

    # accuracy and loss plots
    plot_acc(history_cnn, n_epoch)
    plot_loss(history_cnn, n_epoch)
    

if args == "densenet":
    # call model
    model = dense_net(input_shape = (240,240,1), Base = 8, dense_blocks = 4, 
                            blocks_per_layer = None, growth_rate = 8, classes = n_classes,
                            depth = 4, bottleneck = False, 
                            dropout_rate = 0.5)
    model.summary()
    
    # compiling and fitting
    model.compile(optimizer=Adam(lr=0.00001), loss='binary_crossentropy', metrics=['accuracy'])
    History_dense = model.fit(train_img, Train_labels, batch_size=batch_size,
                                    epochs=n_epoch, shuffle=True, validation_split = 0.2,verbose=1) 
    
    # accuracy and loss plots
    plot_acc(History_dense, n_epoch)
    plot_loss(History_dense, n_epoch)
    
    
    
# ---------------------------- EVALUATION WITH VALIDATION DATA ----------------------------

# evaluation of the model
scores = model.evaluate(train_img, Train_labels, verbose=1)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# prediction of the validation set
y_prob = model.predict(val_img) 

# taking weights of last layer and create new model
gap_weights = model.layers[-1].get_weights()[0]

new_model = Model(inputs=model.input, 
                    outputs=(model.layers[-3].output, model.layers[-1].output)) 

# predicting
features, results = new_model.predict(val_img)

# evaluate the first 10 images
for idx in range(10):
    plt.figure(facecolor='white')

    ax = plt.subplot(1, 2, 1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.imshow(val_images[idx], cmap = 'gray')

    plt.xlabel("Real Labels: " + str(val_labels[idx].astype(int)))
    #plt.title(Test_labels[idx])

    plt.subplot(1, 2, 2)
    plt.bar(range(5), results[idx])
    plt.xticks(np.arange(0, 5.1, 1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlabel("Predicted Labels")
    plt.grid()

    plt.show()
    print(" ******** predicted probabilities ******** ")
    print(" >>> ", results[idx])


# ---------------------------- ACTIVATION MAPS ----------------------------

    

    












