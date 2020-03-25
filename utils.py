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



#**********Dice Coefficient Function**********

def dice_coef(y_true, y_pred, smooth=1):
    
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)


#**********Dice loss function**********

def dice_coef_loss(y_true, y_pred, smooth=1):
    return 1-dice_coef(y_true, y_pred, smooth=1)

