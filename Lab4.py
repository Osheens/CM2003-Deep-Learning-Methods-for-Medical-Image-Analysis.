#!/usr/bin/env python
# coding: utf-8

# In[7]:


import tensorflow as tf
tf.config.gpu.set_per_process_memory_fraction(0.3)
tf.config.gpu.set_per_process_memory_growth(True)


# In[8]:


####********************Importing libraries********************####


import os
import numpy as np
from random import shuffle
from skimage.io import imread
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, Lambda, MaxPooling2D, Dropout, Conv2DTranspose, concatenate, BatchNormalization, Activation
from tensorflow.keras.activations  import relu, sigmoid, softmax
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.metrics import Precision, Recall
from sklearn.model_selection import StratifiedKFold, KFold
import cv2 


# In[ ]:


####********************Function for Loading Data********************####

def test_train_data(data_path,img_w,img_h):                             
    Image_path = os.path.join(data_path, 'Image')   
    Mask_path = os.path.join(data_path, 'Mask')
    Image_list = os.listdir(Image_path)
    Mask_list = os.listdir(Mask_path)
    
    
    im_train = []
    for i in range(len(Image_list)):
        image_name = Image_list[i]
        img = imread(os.path.join(Image_path, image_name), as_grey=True)
        img = resize(img, (img_h, img_w), anti_aliasing = True).astype('float32')
        im_train.append(img) 
        
        if i % 200 == 0:
             print('Reading: {0}/{1}  of train images'.format(i, len(Image_list)))
                
    msk_train = []
    for i in range(len(Mask_list)):
        mask_name = Mask_list[i]
        m = imread(os.path.join(Mask_path, mask_name), as_grey=True)
        m = resize(m, (img_h, img_w), anti_aliasing = True).astype('float32')
        msk_train.append(m) 
        
        if i % 200 == 0:
             print('Reading: {0}/{1}  of train masks'.format(i, len(Mask_list)))      
        
    img_train = np.expand_dims(im_train, axis = -1)
    img_train = np.array(img_train)
    
    mask_train = np.expand_dims(msk_train, axis = -1)
    mask_train = np.array(mask_train)
    

      
    return img_train, mask_train
    
    
####********************Calling the function for Loading Data********************####

Path = '/Lab1/Lab3/MRI/'
im_train, mask_train = test_train_data(Path,240,240)


# In[ ]:


def dice_coefficient(output, target, loss_type='jaccard', axis=(1, 2, 3), smooth=1e-5):
    inse = tf.reduce_sum(output * target, axis=axis)
    if loss_type == 'jaccard':
        l = tf.reduce_sum(output * output, axis=axis)
        r = tf.reduce_sum(target * target, axis=axis)
    elif loss_type == 'sorensen':
        l = tf.reduce_sum(output, axis=axis)
        r = tf.reduce_sum(target, axis=axis)
    else:
        raise Exception("Unknow loss_type")
    
    dice = (2. * inse + smooth) / (l + r + smooth)
    
    dice = tf.reduce_mean(dice, name='dice_coe')
    return dice


# In[ ]:


####********************Building U-Net Model with K-Fold as 3********************####

def model_UNet(Base, img_ch, img_height, img_width, batch_norm, dr):
    
    inputs = Input((img_height, img_width, img_ch))

    #######################*********CONTRACTION*********#######################

    c1 = Conv2D(Base, (3, 3), kernel_initializer='he_normal',padding='same')(inputs)
    c1 = BatchNormalization()(c1) if batch_norm else c1
    c1 = Activation('relu')(c1)
    c1 = Conv2D(Base, (3, 3), kernel_initializer='he_normal',padding='same')(c1)
    c1 = BatchNormalization()(c1) if batch_norm else c1
    c1 = Activation('relu')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(drp)(p1) if dr else p1

    c2 = Conv2D(Base*2, (3, 3), kernel_initializer='he_normal',padding='same')(p1)
    c2 = BatchNormalization()(c2) if batch_norm else c2
    c2 = Activation('relu')(c2)
    c2 = Conv2D(Base*2, (3, 3), kernel_initializer='he_normal',padding='same')(c2)
    c2 = BatchNormalization()(c2) if batch_norm else c2
    c2 = Activation('relu')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(drp)(p2) if dr else p2

    c3 = Conv2D(Base*4, (3, 3), kernel_initializer='he_normal',padding='same')(p2)
    c3 = BatchNormalization()(c3) if batch_norm else c3
    c3 = Activation('relu')(c3)
    c3 = Conv2D(Base*4, (3, 3), kernel_initializer='he_normal',padding='same')(c3)
    c3 = BatchNormalization()(c3) if batch_norm else c3
    c3 = Activation('relu')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(drp)(p3) if dr else p3

    c4 = Conv2D(Base*8, (3, 3), kernel_initializer='he_normal',padding='same')(p3)
    c4 = BatchNormalization()(c4) if batch_norm else c4
    c4 = Activation('relu')(c4)
    c4 = Conv2D(Base*8, (3, 3), kernel_initializer='he_normal',padding='same')(c4)
    c4 = BatchNormalization()(c4) if batch_norm else c4
    c4 = Activation('relu')(c4)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(drp)(p4) if dr else p4


    #######################*********BOTTLENECK*********#######################

    c5 = Conv2D(Base*16, (3, 3), kernel_initializer='he_normal',padding='same')(p4) 
    c5 = BatchNormalization()(c5) if batch_norm else c5
    c5 = Activation('relu')(c5)
    c5 = Conv2D(Base*16, (3, 3), kernel_initializer='he_normal',padding='same')(c5) 
    c5 = BatchNormalization()(c5) if batch_norm else c5
    c5 = Activation('relu')(c5)

    #######################*********EXPANSION*********#######################

    c6 = Conv2DTranspose(Base*8, (2, 2), strides=(2, 2), kernel_initializer='he_normal',padding='same')(c5)
    c6 = concatenate([c6, c4])
    c6 = Dropout(drp)(c6) if dr else c6
    c6 = Conv2D(Base*8, (3, 3), kernel_initializer='he_normal',padding='same')(c6)
    c6 = BatchNormalization()(c6) if batch_norm else c6
    c6 = Activation('relu')(c6)
    c6 = Conv2D(Base*8, (3, 3), kernel_initializer='he_normal',padding='same')(c6)
    c6 = BatchNormalization()(c6) if batch_norm else c6
    c6 = Activation('relu')(c6)

    c7 = Conv2DTranspose(Base*4, (2, 2), strides=(2, 2), kernel_initializer='he_normal',padding='same')(c6)
    c7 = concatenate([c7, c3])
    c7 = Dropout(drp)(c7) if dr else c7
    c7 = Conv2D(Base*4, (3, 3), kernel_initializer='he_normal',padding='same')(c7)
    c7 = BatchNormalization()(c7) if batch_norm else c7
    c7 = Activation('relu')(c7)   
    c7 = Conv2D(Base*4, (3, 3), kernel_initializer='he_normal',padding='same')(c7)
    c7 = BatchNormalization()(c7) if batch_norm else c7
    c7 = Activation('relu')(c7) 
    

    c8 = Conv2DTranspose(Base*2, (2, 2), strides=(2, 2), kernel_initializer='he_normal',padding='same')(c7)
    c8 = concatenate([c8, c2])
    c8 = Dropout(drp)(c8) if dr else c8
    c8 = Conv2D(Base*2, (3, 3), kernel_initializer='he_normal',padding='same')(c8)
    c8 = BatchNormalization()(c8) if batch_norm else c8
    c8 = Activation('relu')(c8)
    c8 = Conv2D(Base*2, (3, 3), kernel_initializer='he_normal',padding='same')(c8)
    c8 = BatchNormalization()(c8) if batch_norm else c8
    c8 = Activation('relu')(c8)

    c9 = Conv2DTranspose(Base, (2, 2), strides=(2, 2), kernel_initializer='he_normal',padding='same')(c8)
    c9 = concatenate([c9, c1])
    c9 = Dropout(drp)(c9) if dr else c9
    c9 = Conv2D(Base, (3, 3), kernel_initializer='he_normal',padding='same')(c9)
    c9 = BatchNormalization()(c9) if batch_norm else c9
    c9 = Activation('relu')(c9)
    c9 = Conv2D(Base, (3, 3), kernel_initializer='he_normal',padding='same')(c9)
    c9 = BatchNormalization()(c9) if batch_norm else c9
    c9 = Activation('relu')(c9)

    outputs = Conv2D(1, (1, 1))(c9)
    outputs = BatchNormalization()(outputs) if batch_norm else outputs
    outputs = Activation('sigmoid')(outputs)
    
    Unet_Model = Model(inputs, outputs)
    Unet_Model.summary()
    return Unet_Model


# In[ ]:


## ********************Fold Dataset and Train the Model********************####

k_fold = 3
batch_size = 2
lr = 0.00001
drp = 0.5
n_epochs = 150
 
for train_index,test_index in KFold(k_fold).split(im_train):
    
    x_train,x_test=im_train[train_index],im_train[test_index]
    y_train,y_test=mask_train[train_index],mask_train[test_index]

    Unet_Model = model_UNet(16,1,240,240, 'batch_norm','dr')
    Unet_Model.compile(optimizer = Adam(lr = lr), loss = 'binary_crossentropy', metrics = [dice_coefficient,
                                                                                           tf.keras.metrics.Precision(), 
                                                                                           tf.keras.metrics.Recall()])
    
    History = Unet_Model.fit(x_train, y_train, batch_size = batch_size, epochs= n_epochs,validation_data = (x_test, y_test), 
                             verbose=1)
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


# In[1]:


get_ipython().system("ls '/Lab1/Lab3/MRI/Mask'")


# In[9]:


img = cv2.imread('Brats17_2013_10_1_t1ce_32_Tumor.png') 
cv2.imshow('image',img)


# In[1]:


import cv2 
import numpy as np 
  
# Reading the input image 
img = cv2.imread('Brats17_2013_10_1_t1ce_32_Tumor.png') 
  
# Taking a matrix of size 5 as the kernel 
kernel = np.ones((2,2), np.uint8) 
  
# The first parameter is the original image, 
# kernel is the matrix with which image is  
# convolved and third parameter is the number  
# of iterations, which will determine how much  
# you want to erode/dilate a given image.  
img_erosion = cv2.erode(img, kernel, iterations=1) 
img_dilation = cv2.dilate(img, kernel, iterations=1) 
diff = img_erosion - img_dilation
  
cv2.imshow('Mask', img) 
cv2.imshow('Erosion', img_erosion) 
cv2.imshow('Dilation', img_dilation) 
cv2.imshow('DIfference', diff) 
  
cv2.waitKey(0) 


# # TASK 2

# In[ ]:


def binary_masks(mask):
   
    # Taking a matrix of size 2 as the kernel 
    kernel = np.ones((2,2), np.uint8) 

    # The first parameter is the original image, 
    # kernel is the matrix with which image is  
    # convolved and third parameter is the number  
    # of iterations, which will determine how much  
    # you want to erode/dilate a given image.  

    mask_dilation = cv2.dilate(mask, kernel, iterations=1) 
    mask_erosion = cv2.erode(mask_dilation, kernel, iterations=1) 
    binary_mask = mask_dilation-mask_erosion

    cv2.imshow('Input', mask) 
    cv2.imshow('Dilation', mask_dilation) 
    cv2.imshow('Erosion', mask_erosion) 
    
    cv2.waitKey(0) 
    
    return binary_mask


# In[ ]:


####********************Calling the binary mask function********************####
 
mask_binary = []
for i in range(len(mask_train)):
    mask_name = mask_train[i]
    mask_binary.append(mask_name)
cv2.imshow(mask_binary[1])
cv2.imshow(mask_binary[2])

        


# In[ ]:


def weighted_loss(weight_map, weight_strength):
    def weighted_dice_loss(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    weight_f = K.flatten(weight_map)
    weight_f = weight_f * weight_strength
    weight_f = ## complete!
    weighted_intersection = K.sum(weight_f * (y_true_f * y_pred_f))
    return -(2. * weighted_intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())
return weighted_dice_loss


# In[ ]:


train_generator = generator_with_weights(x_train, y_train, weight_train,
Batch_size)
model_history = model.fit_generator(
 train_generator, steps_per_epoch=steps_per_epoch, epochs=n_epoch,
 verbose=1, max_queue_size=1, validation_steps=len(x_val),
 validation_data=([x_val, weight_val], y_val),
 shuffle=True, class_weight='auto')
def combine_generator(gen1, gen2, gen3):
    while True:
    x = gen1.next()
    y = gen2.next()
    w = gen3.next()
    yield([x, w], y)
def generator_with_weights(x_train, y_train, weights_train, batch_size):
    background_value = x_train.min()
    data_gen_args = dict(rotation_range=10.,
    width_shift_range=0.1,
    height_shift_range=0.1,
    cval=background_value,
    zoom_range=0.2,
    horizontal_flip=True)
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    weights_datagen = ImageDataGenerator(**data_gen_args)
    image_generator = image_datagen.flow(x_train, shuffle=False,
    batch_size=batch_size,
    seed=1)
    mask_generator = mask_datagen.flow(y_train, shuffle=False,
    batch_size=batch_size,
    seed=1)
    weight_generator = weights_datagen.flow(weights_train, shuffle=False,
    batch_size=batch_size,
    seed=1)
    train_generator = combine_generator(image_generator, mask_generator,
    weight_generator)
    return train_generator

