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

def lstm_model(hidden_size, dropout=True, dr=0.2, n_batches = 16, input_size = None , input_dimension = 3, bidirectional=False):
    lstm_model = Sequential()
    #lstm_model.add(Embedding(training_set, hidden_size, input_length=num_steps))
    if bidirectional==True:
        lstm_model.add(Bidirectional(LSTM(hidden_size, return_sequences=True), input_shape = (input_size, input_dimension) ))
    else:
        lstm_model.add(LSTM(hidden_size, return_sequences=True, stateful = True, batch_input_shape=(n_batches, input_size, input_dimension)))
    if dropout==True:
        lstm_model.add(Dropout(dr))
    lstm_model.add(LSTM(hidden_size, return_sequences=True))
    if dropout==True:
        lstm_model.add(Dropout(dr))
    lstm_model.add(LSTM(hidden_size, return_sequences=True))
    if dropout==True:
        lstm_model.add(Dropout(dr))
    lstm_model.add(LSTM(hidden_size))
    if dropout==True:
        lstm_model.add(Dropout(dr))
    lstm_model.add(Dense(1, activation='sigmoid'))
    
    lstm_model.summary()
    return lstm_model   


def U_Net(size= (240,240,1),batch_norm=True,dropout=True,drop_r=0.5,spatial_drop=False,spatial_drop_r=0.1,multi_class=False,classes=3):
    
    # U-Net model.
    
    # Optionally, introduce:
    # Batch normalization
    # Dropout and dropout rate
    # Spatial dropout and spatial dropout rate
    # Multi-class segmentation and number of classes
    
    #******************************************* C O N T R A C T I O N ******************************************#
    U_Net=Sequential()
    
    image = Input(shape=size)
    #weight= Input(shape=size)
    #inputs= concatenate([image,weight],axis=-1)
    c1_1 = Conv2D(Base, 3, padding = 'same', kernel_initializer = 'he_normal')(image)
    if batch_norm==True:
        b1_1=BatchNormalization()(c1_1)
    else:
        b1_1=c1_1
    a1_1 = Activation('relu')(b1_1)
    c1_2 = Conv2D(Base, 3, padding = 'same', kernel_initializer = 'he_normal')(a1_1)
    if batch_norm==True:
        b1_2=BatchNormalization()(c1_2)
    else:
        b1_2=c1_2
    a1_2 = Activation('relu')(b1_2)
    if spatial_drop==True:
        sd1=SpatialDropout2D(spatial_drop_r)(a1_2)
    else:
        sd1= a1_2
    p1 = MaxPooling2D(pool_size=(2, 2))(s1)

    c2_1= Conv2D(Base*2, 3, padding = 'same', kernel_initializer = 'he_normal')(p1)
    if batch_norm==True:
        b2_1=BatchNormalization()(c2_1)
    else:
        b2_1=c2_1
    a2_1 = Activation('relu')(b2_1)
    c2_2 = Conv2D(Base*2, 3, padding = 'same', kernel_initializer = 'he_normal')(a2_1)
    if batch_norm==True:
        b2_2=BatchNormalization()(c2_1)
    else:
        b2_2=c2_2
    a2_2 = Activation('relu')(b2_2)
    if spatial_drop==True:
        sd2=SpatialDropout2D(spatial_drop_r)(a2_2)
    else:
        sd2=a2_2
    p2 = MaxPooling2D(pool_size=(2, 2))(sd2)
        
    c3_1 = Conv2D(Base*4, 3, padding = 'same', kernel_initializer = 'he_normal')(p2)
    if batch_norm==True:
        b3_1=BatchNormalization()(c3_1)
    else:
        b3_1=c3_1
    a3_1 = Activation('relu')(b3_1)
    c3_2 = Conv2D(Base*4, 3, padding = 'same', kernel_initializer = 'he_normal')(a3_1)
    if batch_norm==True:
        b3_2=BatchNormalization()(c3_2)
    else:
        b3_2=c3_2
    a3_2 = Activation('relu')(b3_2)
    if spatial_drop==True:
        sd3=SpatialDropout2D(spatial_drop_r)(a3_2)
    else:
        sd3=a3_2
    p3 = MaxPooling2D(pool_size=(2, 2))(sd3)
        
    c4_1 = Conv2D(Base*8, 3, padding = 'same', kernel_initializer = 'he_normal')(p3)
    if batch_norm==True:
        b4_1=BatchNormalization()(c4_1)
    else:
        b4_1=c4_1
    a4_1 = Activation('relu')(b4_1)
    c4_2 = Conv2D(Base*8, 3, padding = 'same', kernel_initializer = 'he_normal')(a4_1)
    if batch_norm==True:
        b4_2=BatchNormalization()(c4_2)
    else:
        b4_2=c4_2
    a4_2 = Activation('relu')(b4_2)
    if spatial_drop==True:
        sd4=SpatialDropout2D(spatial_drop_r)(a4_2)
    else:
        sd4=a4_2
    if dropout==True:
        d4 = Dropout(drop_r)(sd4)
    else:
        d4=sd4
    p4 = MaxPooling2D(pool_size=(2, 2))(d4)
        
    c5_1 = Conv2D(Base*16, 3, padding = 'same', kernel_initializer = 'he_normal')(p4)
    if batch_norm==True:
        b5_1=BatchNormalization()(c5_1)
    else:
        b5_1=c5_1
    a5_1= Activation('relu')(b5_1)
    c5_2 = Conv2D(Base*16, 3, padding = 'same', kernel_initializer = 'he_normal')(a5_1)
    if batch_norm==True:
        b5_2=BatchNormalization()(c5_2)
    else:
        b5_2=c5_2
    a5_2= Activation('relu')(b5_2)

    if dropout==True:
        d5 = Dropout(drop_r)(a5_2)
    else:
        d5=a5_2

    #******************************************* E X P A N S I O N ******************************************#

    c_up6 = Conv2D(Base*8, 2, padding = 'same', kernel_initializer = 'he_normal')(d5)

    a6_1 = Activation('relu')(c_up6)
    upsample6=UpSampling2D(size = (2,2))(a6_1)
    merge6 = concatenate([d4,upsample6])
    c6_1 = Conv2D(Base*8, 3, padding = 'same', kernel_initializer = 'he_normal')(merge6)
    if batch_norm==True:
        b6_1=BatchNormalization()(c6_1)
    else:
        b6_1=c6_1
    a6_2 = Activation('relu')(b6_1)
    c6_2 = Conv2D(Base*8, 3, padding = 'same', kernel_initializer = 'he_normal')(a6_2)
    if batch_norm==True:
        b6_2=BatchNormalization()(c6_2)
    else:
        b6_2=c6_2
    a6_3 = Activation('relu')(b6_2)

    c_up7 = Conv2D(Base*4, 2, padding = 'same', kernel_initializer = 'he_normal')(a6_3)
    a7_1 = Activation('relu')(c_up7)
    upsample7=UpSampling2D(size = (2,2))(a7_1)
    merge7 = concatenate([c3_2,upsample7])
    c7_1 = Conv2D(Base*4, 3, padding = 'same', kernel_initializer = 'he_normal')(merge7)
    if batch_norm==True:
        b7_1=BatchNormalization()(c7_1)
    else:
        b7_1=c7_1
    a7_2 = Activation('relu')(b7_1)
    c7_2 = Conv2D(Base*4, 3, padding = 'same', kernel_initializer = 'he_normal')(a7_2)
    if batch_norm==True:
        b7_2=BatchNormalization()(c7_2)
    else:
        b7_2=c7_2
    a7_3 = Activation('relu')(b7_2)

    c_up8 = Conv2D(Base*2, 2, padding = 'same', kernel_initializer = 'he_normal')(a7_3)
    a8_1 = Activation('relu')(c_up8)
    upsample8=UpSampling2D(size = (2,2))(a8_1)
    merge8 = concatenate([c2_2,upsample8])
    c8_1 = Conv2D(Base*2, 3, padding = 'same', kernel_initializer = 'he_normal')(merge8)
    if batch_norm==True:
        b8_1=BatchNormalization()(c8_1)
    else:
        b8_1=c8_1
    a8_2 = Activation('relu')(b8_1)
    c8_2 = Conv2D(Base*2, 3, padding = 'same', kernel_initializer = 'he_normal')(a8_2)
    if batch_norm==True:
        b8_2=BatchNormalization()(c8_2)
    else:
        b8_2=c8_2
    a8_3 = Activation('relu')(b8_2)

    c_up9 = Conv2D(Base, 2, padding = 'same', kernel_initializer = 'he_normal')(a8_3)
    a9_1 = Activation('relu')(c_up9)
    upsample9=UpSampling2D(size = (2,2))(a9_1)
    merge9 = concatenate([c1_2,upsample9])
    c9_1 = Conv2D(Base, 3, padding = 'same', kernel_initializer = 'he_normal')(merge9)
    if batch_norm==True:
        b9_1=BatchNormalization()(c9_1)
    else:
        b9_1=c9_1
    a9_2 = Activation('relu')(b9_1)
    c9_2 = Conv2D(Base, 3, padding = 'same', kernel_initializer = 'he_normal')(a9_2)
    if batch_norm==True:
        b9_2=BatchNormalization()(c9_2)
    else:
        b9_2=c9_2
    a10_1 = Activation('relu')(b9_2)
    c10_1 = Conv2D(Base, 3, padding = 'same', kernel_initializer = 'he_normal')(a10_1)
    if batch_norm==True:
        b10_1=BatchNormalization()(c10_1)
    else:
        b10_1=c10_1
    a10_2 = Activation('relu')(b10_1)
    if multi_class==False:
        last_layer = Conv2D(1, 1, activation = 'sigmoid')(a10_2)
    else:
        c10_2 = Conv2D(classes, 1)(a10_2)
        b10_3=BatchNormalization()(c10_2)
        last_layer=Activation('softmax')(b10_3)

    unet_weights=Model(inputs=[image,weight],outputs=last_layer)
    return unet_weights

def unet(Base= 16, size= (240,240,1),batch_norm=True,dropout=True,drop_r=0.5,spatial_drop=False,spatial_drop_r=0.1,multi_class=False,classes=3):
    
    # U-Net model.
    
    # Optionally, introduce:
    # Batch normalization
    # Dropout and dropout rate
    # Spatial dropout and spatial dropout rate
    # Multi-class segmentation and number of classes
    
    #******************************************* C O N T R A C T I O N ******************************************#
    unet=Sequential()
    
    image = Input(shape=size)
    #weight= Input(shape=size)
    #inputs= concatenate([image,weight],axis=-1)
    
    #1
    c1_1 = Conv2D(Base, 3, padding = 'same', kernel_initializer = 'he_normal')(image)
    if batch_norm==True:
        b1_1=BatchNormalization()(c1_1)
    else:
        b1_1=c1_1
    a1_1 = Activation('relu')(b1_1)
    c1_2 = Conv2D(Base, 3, padding = 'same', kernel_initializer = 'he_normal')(a1_1)
    if batch_norm==True:
        b1_2=BatchNormalization()(c1_2)
    else:
        b1_2=c1_2
    a1_2 = Activation('relu')(b1_2)
    if spatial_drop==True:
        sd1=SpatialDropout2D(spatial_drop_r)(a1_2)
    else:
        sd1= a1_2
    p1 = MaxPooling2D(pool_size=(2, 2))(sd1)

    #2
    c2_1= Conv2D(Base*2, 3, padding = 'same', kernel_initializer = 'he_normal')(p1)
    if batch_norm==True:
        b2_1=BatchNormalization()(c2_1)
    else:
        b2_1=c2_1
    a2_1 = Activation('relu')(b2_1)
    c2_2 = Conv2D(Base*2, 3, padding = 'same', kernel_initializer = 'he_normal')(a2_1)
    if batch_norm==True:
        b2_2=BatchNormalization()(c2_1)
    else:
        b2_2=c2_2
    a2_2 = Activation('relu')(b2_2)
    if spatial_drop==True:
        sd2=SpatialDropout2D(spatial_drop_r)(a2_2)
    else:
        sd2=a2_2
    p2 = MaxPooling2D(pool_size=(2, 2))(sd2)
        
    #3    
    c3_1 = Conv2D(Base*4, 3, padding = 'same', kernel_initializer = 'he_normal')(p2)
    if batch_norm==True:
        b3_1=BatchNormalization()(c3_1)
    else:
        b3_1=c3_1
    a3_1 = Activation('relu')(b3_1)
    c3_2 = Conv2D(Base*4, 3, padding = 'same', kernel_initializer = 'he_normal')(a3_1)
    if batch_norm==True:
        b3_2=BatchNormalization()(c3_2)
    else:
        b3_2=c3_2
    a3_2 = Activation('relu')(b3_2)
    if spatial_drop==True:
        sd3=SpatialDropout2D(spatial_drop_r)(a3_2)
    else:
        sd3=a3_2
    p3 = MaxPooling2D(pool_size=(2, 2))(sd3)
     
    #4    
    c4_1 = Conv2D(Base*8, 3, padding = 'same', kernel_initializer = 'he_normal')(p3)
    if batch_norm==True:
        b4_1=BatchNormalization()(c4_1)
    else:
        b4_1=c4_1
    a4_1 = Activation('relu')(b4_1)
    c4_2 = Conv2D(Base*8, 3, padding = 'same', kernel_initializer = 'he_normal')(a4_1)
    if batch_norm==True:
        b4_2=BatchNormalization()(c4_2)
    else:
        b4_2=c4_2
    a4_2 = Activation('relu')(b4_2)
    if spatial_drop==True:
        sd4=SpatialDropout2D(spatial_drop_r)(a4_2)
    else:
        sd4=a4_2
    if dropout==True:
        d4 = Dropout(drop_r)(sd4)
    else:
        d4=sd4
    p4 = MaxPooling2D(pool_size=(2, 2))(d4)
        
    #5
    c5_1 = Conv2D(Base*16, 3, padding = 'same', kernel_initializer = 'he_normal')(p4)
    if batch_norm==True:
        b5_1=BatchNormalization()(c5_1)
    else:
        b5_1=c5_1
    a5_1= Activation('relu')(b5_1)
    c5_2 = Conv2D(Base*16, 3, padding = 'same', kernel_initializer = 'he_normal')(a5_1)
    if batch_norm==True:
        b5_2=BatchNormalization()(c5_2)
    else:
        b5_2=c5_2
    a5_2= Activation('relu')(b5_2)

    if dropout==True:
        d5 = Dropout(drop_r)(a5_2)
    else:
        d5=a5_2

    #******************************************* E X P A N S I O N ******************************************#
    #6
    c_up6 = Conv2D(Base*8, 2, padding = 'same', kernel_initializer = 'he_normal')(d5)

    a6_1 = Activation('relu')(c_up6)
    upsample6=UpSampling2D(size = (2,2))(a6_1)
    x1 = Reshape(target_shape=(1, np.int32(size[0]/8), np.int32(size[1]/8), Base*8))(d4)
    x2 = Reshape(target_shape=(1, np.int32(size[0]/8), np.int32(size[1]/8), Base*8))(upsample6)  
    merge6 = concatenate([x1,x2])
    merge6 = ConvLSTM2D(Base*4, 3, padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge6)
    c6_1 = Conv2D(Base*8, 3, padding = 'same', kernel_initializer = 'he_normal')(merge6)
    if batch_norm==True:
        b6_1=BatchNormalization()(c6_1)
    else:
        b6_1=c6_1
    a6_2 = Activation('relu')(b6_1)
    c6_2 = Conv2D(Base*8, 3, padding = 'same', kernel_initializer = 'he_normal')(a6_2)
    if batch_norm==True:
        b6_2=BatchNormalization()(c6_2)
    else:
        b6_2=c6_2
    a6_3 = Activation('relu')(b6_2)
    
    #7
    c_up7 = Conv2D(Base*4, 2, padding = 'same', kernel_initializer = 'he_normal')(a6_3)
    a7_1 = Activation('relu')(c_up7)
    upsample7=UpSampling2D(size = (2,2))(a7_1)
    x1 = Reshape(target_shape=(1, np.int32(size[0]/4), np.int32(size[1]/4), Base*4))(c3_2)
    x2 = Reshape(target_shape=(1, np.int32(size[0]/4), np.int32(size[1]/4), Base*4))(upsample7) 
    merge7 = concatenate([x1,x2])
    merge7 = ConvLSTM2D(Base*2, 3, padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge7)
    c7_1 = Conv2D(Base*4, 3, padding = 'same', kernel_initializer = 'he_normal')(merge7)
    if batch_norm==True:
        b7_1=BatchNormalization()(c7_1)
    else:
        b7_1=c7_1
    a7_2 = Activation('relu')(b7_1)
    c7_2 = Conv2D(Base*4, 3, padding = 'same', kernel_initializer = 'he_normal')(a7_2)
    if batch_norm==True:
        b7_2=BatchNormalization()(c7_2)
    else:
        b7_2=c7_2
    a7_3 = Activation('relu')(b7_2)
    
    #8
    c_up8 = Conv2D(Base*2, 2, padding = 'same', kernel_initializer = 'he_normal')(a7_3)
    a8_1 = Activation('relu')(c_up8)
    upsample8=UpSampling2D(size = (2,2))(a8_1)
    x1 = Reshape(target_shape=(1, np.int32(size[0]/2), np.int32(size[1]/2), Base*2))(c2_2)
    x2 = Reshape(target_shape=(1, np.int32(size[0]/2), np.int32(size[1]/2), Base*2))(upsample8) 
    merge8 = concatenate([x1,x2])
    merge8 = ConvLSTM2D(Base, 3, padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge8)
    c8_1 = Conv2D(Base*2, 3, padding = 'same', kernel_initializer = 'he_normal')(merge8)
    if batch_norm==True:
        b8_1=BatchNormalization()(c8_1)
    else:
        b8_1=c8_1
    a8_2 = Activation('relu')(b8_1)
    c8_2 = Conv2D(Base*2, 3, padding = 'same', kernel_initializer = 'he_normal')(a8_2)
    if batch_norm==True:
        b8_2=BatchNormalization()(c8_2)
    else:
        b8_2=c8_2
    a8_3 = Activation('relu')(b8_2)

    #9
    c_up9 = Conv2D(Base, 2, padding = 'same', kernel_initializer = 'he_normal')(a8_3)
    a9_1 = Activation('relu')(c_up9)
    upsample9=UpSampling2D(size = (2,2))(a9_1)
    x1 = Reshape(target_shape=(1, np.int32(size[0]), np.int32(size[1]), Base))(c1_2)
    x2 = Reshape(target_shape=(1, np.int32(size[0]), np.int32(size[1]), Base))(upsample9)
    merge9 = concatenate([x1,x2])
    merge9 = ConvLSTM2D(int(Base/2), 3, padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge9)
    c9_1 = Conv2D(Base, 3, padding = 'same', kernel_initializer = 'he_normal')(merge9)
    if batch_norm==True:
        b9_1=BatchNormalization()(c9_1)
    else:
        b9_1=c9_1
    a9_2 = Activation('relu')(b9_1)
    c9_2 = Conv2D(Base, 3, padding = 'same', kernel_initializer = 'he_normal')(a9_2)
    if batch_norm==True:
        b9_2=BatchNormalization()(c9_2)
    else:
        b9_2=c9_2
    a10_1 = Activation('relu')(b9_2)
    
    #10
    c10_1 = Conv2D(Base, 3, padding = 'same', kernel_initializer = 'he_normal')(a10_1)
    if batch_norm==True:
        b10_1=BatchNormalization()(c10_1)
    else:
        b10_1=c10_1
    a10_2 = Activation('relu')(b10_1)
    if multi_class==False:
        last_layer = Conv2D(1, 1, activation = 'sigmoid')(a10_2)
    else:
        c10_2 = Conv2D(classes, 1)(a10_2)
        b10_3=BatchNormalization()(c10_2)
        last_layer=Activation('softmax')(b10_3)

    unet=Model(inputs=image,outputs=last_layer)
    unet.summary()
    return unet


def plot_model(History):
    plt.figure(figsize=(4, 4))
    plt.title("Learning curve")
    plt.plot(History.history["loss"], label="loss")
    plt.plot(History.history["val_loss"], label="val_loss")
    plt.plot( np.argmin(History.history["val_loss"]),
             np.min(History.history["val_loss"]),
             marker="x", color="r", label="best model")

    plt.xlabel("Epochs")
    plt.ylabel("Loss Value")
    plt.legend()









