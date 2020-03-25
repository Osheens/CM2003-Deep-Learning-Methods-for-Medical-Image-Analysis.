#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import argparse
import sys
import os
import keras
#tf.compat.v1.disable_eager_execution()
#tf.config.gpu.set_per_process_memory_fraction(0.6)
#tf.config.gpu.set_per_process_memory_growth(True)
from keras.backend.tensorflow_backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

from data_loader import data_prep_task1, load_streamlines, test_train_data, MyBatchGenerator
from sklearn.preprocessing import MinMaxScaler
from models import plot_model, lstm_model, unet
from utils import dice_coef, dice_coef_loss

#parser = argparse.ArgumentParser()
#parser.add_argument("-t", "--task", type=int,default="1", help="Number of the task")

#args = parser.parse_args()
args = sys.argv[1]
print(args)

#### task1 ####

if args == "1":
    print('hola2')
    # Preprocessing of data
    X_train, y_train, X_val, y_val = data_prep_task1(timesteps=60)

    # Hyperparameters
    lr = 0.001
    epochs = 100
    n_batches = 16

    # call model, compile and fit
    Model  = lstm_model(40, dropout=True, dr=0.2, n_batches = 16, input_size = X_train.shape[1], input_dimension = 1, bidirectional=False)
    Model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    History = Model.fit(X_train, y_train, batch_size=n_batches, epochs=epochs, validation_data=(X_val, y_val), verbose = 1, shuffle = False)

    # plot evaluation of results
    #plot_model(History)

    #predicted_stock_price = Model.predict(X_val)
    #predicted_stock_price = sc.inverse_transform(predicted_stock_price)
    #predicted_stock_price

    
#### task2 ####

elif args == "2":
    # load data
    n_tracts_per_bundle = 20
    dataPath = '/Lab1/Lab5/HCP_lab/'
    train_subjects_list = ['599469', '599671', '601127']   #your choice of 3 training subjects
    val_subjects_list = ['613538'] # your choice of 1 validation subjects
    bundles_list = ['CST_left', 'CST_right']
    X_train, y_train = load_streamlines(dataPath, train_subjects_list,bundles_list, n_tracts_per_bundle)
    X_val, y_val = load_streamlines(dataPath, val_subjects_list, bundles_list,
                                    n_tracts_per_bundle)
    
    # Hyperparameters
    lr = 0.001
    epochs = 100
    n_batches = 16

    # call model, compile and fit
    Model_fiber = lstm_model(10, dropout=True, dr=0.2, n_batches = 1, input_size = None, input_dimension = 3, bidirectional=True)
    Model_fiber.compile(loss='mse', optimizer='adam', metrics=['mae'])
    History = Model_fiber.fit_generator(MyBatchGenerator(X_train, y_train, batch_size=1),
                                        epochs=epochs, validation_data=MyBatchGenerator(X_val, y_val, batch_size=1),
                                        validation_steps=len(X_val))

    # plot evaluation of results
    plot_model(History)

    
    
#### task3 #### 

elif args == "3a":
    # load data
    MRI_Path = '/Lab1/Lab3/MRI'
    image_path= MRI_Path+'/Image'
    mask_path= MRI_Path+'/Mask'
    img_w,img_h=240,240
    images,masks=test_train_data(image_path,mask_path,img_w,img_h)
    img_train, img_val, mask_train, mask_val = train_test_split(images, masks, test_size=0.2, random_state=42)

    # hyperparameters
    batch_size = 8
    seed = 1
    lr = 0.0001
    n_epochs = 100
    size =(240,240,1)

    # call model, compile and fit
    Unet_Model = unet(Base = 16,size =(240,240,1), batch_norm=True,dropout=True,drop_r=0.5,spatial_drop=False,
                      spatial_drop_r=0.1,multi_class=False, classes = 1)
    Unet_Model.compile(optimizer = Adam(lr = lr), loss = dice_coef_loss, 
                       metrics = [dice_coef, tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    History = Unet_Model.fit(img_train, mask_train, batch_size = batch_size, epochs= n_epochs,  
                             verbose=1, validation_data=(img_val,mask_val))
    
    # plot evaluation of results
    plot_model(History)
    
elif args == "3b":
    # load data
    MRI_Path = '/Lab1/Lab3/MRI'
    image_path= MRI_Path+'/Image'
    mask_path= MRI_Path+'/Mask'
    img_w,img_h=240,240
    images,masks=test_train_data(image_path,mask_path,img_w,img_h)
    img_train, img_val, mask_train, mask_val = train_test_split(images, masks, test_size=0.2, random_state=42)

    # hyperparameters
    batch_size = 8
    seed = 1
    lr = 0.0001
    n_epochs = 80
    size =(240,240,1)

    Unet_Model = unet(Base = 16,size =(240,240,1), batch_norm=True,dropout=True,drop_r=0.5,spatial_drop=False,spatial_drop_r=0.1,multi_class=False, classes = 1)
    Unet_Model.compile(optimizer = Adam(lr = lr), loss = 'binary_crossentropy', metrics = [dice_coef])

    History = Unet_Model.fit_generator(MyBatchGenerator(img_train, mask_train, batch_size=1),
                                    epochs=n_epochs, validation_data=MyBatchGenerator(img_val, mask_val, batch_size=1))
    # plot evaluation of results
    plot_model(History)


















