#!/usr/bin/env python
# coding: utf-8

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, BatchNormalization, SpatialDropout2D, Flatten
from keras.callbacks import EarlyStopping
from keras import regularizers

from keras.layers import Input, Conv2D, Dense, BatchNormalization
from keras.layers import Dropout, Activation, concatenate, AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.models import Model
from keras import regularizers
from keras.optimizers import Adam, SGD
from sklearn.metrics import f1_score
import numpy as np


# -------------------------------------- MODELS -----------------------------------

# CNN-------------------------------------------------------------------------

def cnn5_model(Base = 8, filter_size = 3, input_shape=(240,240,1), batch_norm=False, spatial_drop=False, spatial_drop_r = 0.1):
    
    cnn5_model = Sequential()

    #first layer
    cnn5_model.add(Conv2D(Base, filter_size, activation='relu', padding='same', input_shape=input_shape, kernel_regularizer=regularizers.l2(0.01)))
    if batch_norm==True:
        cnn5_model.add(BatchNormalization())
    if spatial_drop==True:
        cnn5_model.add(SpatialDropout2D(spatial_drop_r))
    cnn5_model.add(MaxPooling2D(pool_size=(2, 2)))
    
    #second layer
    cnn5_model.add(Conv2D(Base, filter_size, activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01)))
    if batch_norm==True:
        cnn5_model.add(BatchNormalization())
    if spatial_drop==True:
        cnn5_model.add(SpatialDropout2D(spatial_drop_r))
    cnn5_model.add(MaxPooling2D(pool_size=(2, 2)))

    #third layer
    cnn5_model.add(Conv2D(Base, filter_size, activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01)))
    if batch_norm==True:
        cnn5_model.add(BatchNormalization())
    if spatial_drop==True:
        cnn5_model.add(SpatialDropout2D(spatial_drop_r))
    cnn5_model.add(MaxPooling2D(pool_size=(2, 2)))

    #forth layer
    cnn5_model.add(Conv2D(Base, filter_size, activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01)))
    if batch_norm==True:
        cnn5_model.add(BatchNormalization())
    if spatial_drop==True:
        cnn5_model.add(SpatialDropout2D(spatial_drop_r))
    cnn5_model.add(MaxPooling2D(pool_size=(2, 2)))

    #fifth layer
    cnn5_model.add(Conv2D(Base, filter_size, activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01)))
    if batch_norm==True:
        cnn5_model.add(BatchNormalization())
    if spatial_drop==True:
        cnn5_model.add(SpatialDropout2D(spatial_drop_r))
    cnn5_model.add(MaxPooling2D(pool_size=(2, 2)))

    #output layer
    cnn5_model.add(GlobalAveragePooling2D())
    cnn5_model.add(Dense(5, activation='sigmoid'))

    return cnn5_model



# DENSENET1-------------------------------------------------------------------------

def denseNet121_model(n_classes=1, input_shape=(320,320,1)):
    base_model = DenseNet121(weights=None, include_top=False, input_shape=input_shape)
    
    x = AveragePooling2D(pool_size=(3,3), name='avg_pool')(base_model.output)
    x = Flatten()(x)
    
    x = Dense(1024, activation='relu', name='dense_post_pool')(x)
    x = Dropout(0.5)(x)
    
    output = Dense(n_classes, activation='sigmoid', name='predictions')(x)
    model = Model(inputs=base_model.input, output=output)

    return model, output



# DENSENET2-------------------------------------------------------------------------

def dense_net(input_shape = (320,320,1), Base = 16, dense_blocks = 4, blocks_per_layer = 4,
              growth_rate = 8, classes=1, depth=5, bottleneck=False, drp_rate=None):

    if not blocks_per_layer:
        blocks_per_layer = [depth]*dense_blocks
    else:
        if not len(blocks_per_layer) == dense_blocks:
            raise Exception("Error: blocks_per_layer == dense_blocks")

    inputs = Input(input_shape)
    n_ch = Base

    # first layer
    X = Conv2D(Base, (5,5), padding='same', strides=(3,3), use_bias=False)(inputs)

    for i in range(dense_blocks):
        X = dense_block(X, blocks_per_layer[i], growth_rate, drp_rate, bottleneck)

        if (dense_blocks - 1) > i:
            n_ch = n_ch + growth_rate*blocks_per_layer[i]
            X = trans_layer(X, n_ch, drp_rate = drp_rate)

    # Last layer: output
    X = out_layer(X)
    X_out = Dense(classes, activation='sigmoid', use_bias=False)(X)

    N = np.sum((1+bottleneck) * np.asarray(blocks_per_layer)) + len(blocks_per_layer) + 1
    
    model = Model(inputs = inputs, outputs =  X_out, name = 'DenseNet{}'.format(N))
    
    return model

# blocks 
def dense_block(X, depth = 5, Base = 12, drp_rate = None, bottleneck = False):

    # concatenations
    for i in range(depth):
        X_n = comp_layer(X, Base, drp_rate = drp_rate)
        X = concatenate([X, X_n], axis=-1)

    return X

# bottle neck layer
def bottleneck_layer(X, Base, drp_rate=None):

    X = BatchNormalization(axis=-1)(X)
    X = Activation('relu')(X)
    X = Conv2D(Base * 4, kernel_size=(1, 1), strides=(1, 1), use_bias=False)(X)

    if drp_rate:
            X = Dropout(drp_rate)(X)
    return X

# composition layer
def comp_layer(X, Base, drp_rate = None, bottleneck = False):

    if bottleneck:
        bottleneck_layer(X, Base, drp_rate = drp_rate)

    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(Base, kernel_size=(3, 3), padding ='same', use_bias=False)(X)
    
    return X

# transition layer
def trans_layer(X, Base, drp_rate = None):

    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(Base, kernel_size=(1,1), padding='same', use_bias=False, kernel_regularizer=regularizers.l2(0.01))(X)

    if drp_rate:
        X = Dropout(drp_rate)(X)
        
    X = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(X)
    
    return X

# final output layer
def out_layer(X):

    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    
    X = GlobalAveragePooling2D()(X)
    
    return X




