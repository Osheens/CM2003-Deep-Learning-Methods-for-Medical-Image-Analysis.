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

from collections import defaultdict
from skimage.io import imread
from skimage.transform import resize



def feature_string(row):
    
    # Applying policies depending on pathology
    u_one_policy = ['Atelectasis', 'Edema']
    u_zero_policy = ['Cardiomegaly', 'Consolidation', 'Pleural Effusion']

    feature_list = []
    for feature in u_one_policy:
        if row[feature] in [-1,1]: # takes positives (1) and uncertain ones (uncertains = -1 and policy maps to 1)
            feature_list.append(feature)

    for feature in u_zero_policy:
        if row[feature] == 1: # takes positives (1) (uncertains will go to negative)
            feature_list.append(feature)

    return ';'.join(feature_list)


def preproc_data(train_data, val_data):

    # Add new columns to training and validation set.
    train_data['feature_string'] = train_data.apply(feature_string,axis = 1)
    val_data['feature_string'] = val_data.apply(feature_string,axis = 1)

    labels_freq = defaultdict(int)

    for r in train_data['feature_string']:
        labels = r.split(";")
        for l in labels:
            labels_freq[l] += 1

    labels_list = list(labels_freq.keys())
    labels_list = labels_list[1:6]
    # print(labels_list)

    label_2_idx = {}
    idx_2_label = {}
    for i,v in enumerate(labels_list):
        label_2_idx[v] = i
        idx_2_label[i] = v

    return labels_list, label_2_idx, idx_2_label



# ---------------------------- LOADING OF IMAGES ----------------------------

def get_train_data(train_data, img_h, img_w):
    train_images = []
    #for i in range (len(train_data)):
    for i in range (0, 15):
        direc = '../../dl_data/'
        path = train_data['Path'].iloc[i]
        img = imread(os.path.join(direc,path), as_grey = True)
        img = resize(img, (img_h, img_w), anti_aliasing = True).astype('float32')
        train_images.append(img)
        
        if i % 200 == 0:
            print('Reading: {0}/{1}  of images'.format(i, 15000))

    #Saving all the training lists into a single array
    Train_Img = np.zeros((len(train_images), img_h, img_w), dtype = np.float32)
    for i in range(len(train_images)):
        Train_Img[i] = train_images[i]
    Train_Img = np.expand_dims(Train_Img, axis = 3)

    return Train_Img, train_images

def get_val_data(val_data, img_h, img_w):
    val_images = []
    for i in range (10):
        direc = '../../dl_data/'
        path = val_data['Path'].iloc[i]
        img = imread(os.path.join(direc,path), as_grey = True)
        img = resize(img, (img_h, img_w), anti_aliasing = True).astype('float32')
        val_images.append(img)

    #Saving all the validation lists into a single array
    Val_Img = np.zeros((len(val_images), img_h, img_w), dtype = np.float32)
    for i in range(len(val_images)):
        Val_Img[i] = val_images[i]
    Val_Img = np.expand_dims(Val_Img, axis = 3)

    return Val_Img, val_images


# ---------------------------- LOADING OF LABELS ----------------------------

#Features to one hot vector

def feature_2_one_hot(feature, n_classes=1, lookup_dict=None):
    y = np.zeros((n_classes))
    f = feature.split(";")
    if f == ['']:
        y = y
    else:
        for i in f:
            idx = lookup_dict[i]
            y[idx] = 1

    return y


def get_labels(train_data, val_data, Train_Img, Val_Img, lookup_dict):
    
    train_labels=[]
    #for i in range (len(train_data)):
    for i in range (len(Train_Img)):
        y = feature_2_one_hot(train_data['feature_string'].iloc[i+1], n_classes=5,  lookup_dict = lookup_dict)
        train_labels.append(y)


    Train_labels = np.zeros((len(train_labels),5), dtype = np.float32)
    for i in range(len(train_labels)):
        Train_labels[i] = train_labels[i]


    #Labels for validation data by calling the definition
    val_labels=[]
    #for i in range (len(val_data)):
    for i in range (len(Val_Img)):
        y = feature_2_one_hot(val_data['feature_string'].iloc[i], n_classes=5,  lookup_dict = lookup_dict)
        val_labels.append(y)

    Val_labels = np.zeros((len(val_labels),5), dtype = np.float32)
    for i in range(len(val_labels)):
        Val_labels[i] = val_labels[i]



    return Train_labels, Val_labels, train_labels, val_labels








