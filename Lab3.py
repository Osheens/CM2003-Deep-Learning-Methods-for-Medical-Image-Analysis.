#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
tf.config.gpu.set_per_process_memory_fraction(0.3)
tf.config.gpu.set_per_process_memory_growth(True)


# In[3]:


# Importing libraries
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


# In[4]:


#Function for Image Loader, Mask Loader, Similarity matrices (Dice Coefficient)

#Function for Loading data
def test_train_data(data_path,img_w,img_h):                             
    Image_path = os.path.join(data_path, 'Image')   
    Mask_path = os.path.join(data_path, 'Mask')
    Image_list = os.listdir(Image_path)
    Mask_list = os.listdir(Mask_path)
    dataset= list(zip(Image_list, Mask_list))
    shuffle(dataset)
    Image_list, Mask_list = zip(*dataset)
    
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
             print('Reading: {0}/{1}  of train images'.format(i, len(Mask_list)))         
    
    img_train, img_val, mask_train, mask_val = train_test_split(im_train, msk_train, test_size = 0.2)
        
    img_train = np.expand_dims(img_train, axis = -1)
    img_train = np.array(img_train)
    img_val = np.expand_dims(img_val, axis = -1)    
    img_val = np.array(img_val)
    mask_train = np.expand_dims(mask_train, axis = -1)
    mask_train = np.array(mask_train)
    mask_val = np.expand_dims(mask_val, axis = -1)
    mask_val = np.array(mask_val)

    return img_train, img_val, mask_train, mask_val


# In[5]:


#**********************Implementation of Data Augmentation for Task 4**********************

def augmentation_generator(x,y,seed,batch_size):
    genX1 = im_datagen.flow(x, y, batch_size=batch_size, seed=seed)
    genX2 = im_datagen.flow(y, x, batch_size=batch_size, seed=seed)
    while True:
        X1i = genX1.next()
        X2i = genX2.next()

        yield X1i[0], X2i[0]      
        
        
        
#**********************Functions for Dice Coefficient**********************        


        
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



#**********************Functions for Dice Loss**********************       

def dice_loss(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1,2,3))
    denominator = tf.reduce_sum(y_true + y_pred, axis=(1,2,3))

    return (1 - numerator / denominator)


# In[ ]:


#**********************Task 1 to 4 Calling the function for loading data (For x-ray Lung Data)**********************

Path = '/Lab1/Lab3/X_ray/'
img_train, img_val, mask_train, mask_val = test_train_data(Path,256,256)


# In[6]:


#**********************U-net Model**********************

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, Lambda, MaxPooling2D, Dropout, Conv2DTranspose, concatenate, BatchNormalization, Activation
from tensorflow.keras.activations  import relu, sigmoid, softmax
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.metrics import Precision, Recall


batch_size = 8
lr = 0.0001
drp = 0.5
n_epochs = 80

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


# In[6]:


Unet_Model = model_UNet(16,1,256,256, 'batch_norm','dr')
Unet_Model.compile(optimizer = Adam(lr = lr), loss = 'binary_crossentropy', metrics = [dice_coefficient])


# In[ ]:


#Task 4: Compiling and Fitting the U-Net Model with data augmentation Xray 
batch_size = 8
seed = 1
im_datagen = ImageDataGenerator(rotation_range = 10, width_shift_range = 0.1, height_shift_range = 0.1, zoom_range = 0.2, horizontal_flip = True)



History = Unet_Model.fit_generator(augmentation_generator(img_train, mask_train, seed, batch_size), steps_per_epoch=np.ceil(float(len(img_train)) / float(batch_size)),
                validation_data = augmentation_generator(img_val, mask_val,seed, batch_size), 
                validation_steps = np.ceil(float(len(img_val)) / float(batch_size)), shuffle=True, epochs=n_epochs)
# Learning curve plots
plt.figure(figsize=(4, 4))
plt.title("Learning curve")
plt.plot(History.history["loss"], label="loss")
plt.plot(History.history["val_loss"], label="val_loss")
plt.plot( np.argmin(History.history["val_loss"]),
         np.min(History.history["val_loss"]),
         marker="x", color="r", label="best model")

plt.xlabel("Epochs")
plt.ylabel("Loss Value")
plt.legend(); 


# In[7]:


#Task 5 to Calling the function test_train_data for loading data(For CT lung Data)
Path = '/Lab1/Lab3/CT/'
img_train, img_val, mask_train, mask_val = test_train_data(Path,256,256)


# In[8]:


#Task 3 and 5a: Compiling and Fitting the U-Net Model without data augmentation
History = Unet_Model.fit(img_train, mask_train, batch_size = batch_size, epochs= n_epochs,  verbose=1, validation_data=(img_val,mask_val))
plt.figure(figsize=(4, 4))
plt.title("Learning curve")
plt.plot(History.history["loss"], label="loss")
plt.plot(History.history["val_loss"], label="val_loss")
plt.plot( np.argmin(History.history["val_loss"]),
         np.min(History.history["val_loss"]),
         marker="x", color="r", label="best model")

plt.xlabel("Epochs")
plt.ylabel("Loss Value")
plt.legend(); 


# In[9]:


#Task 5b With Data augmentation for CT images
Unet_Model = model_UNet(16,1,256,256, 'batch_norm','dr')
Unet_Model.compile(optimizer = Adam(lr = lr), loss = 'binary_crossentropy', metrics = [dice_coefficient,tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
batch_size = 8
seed = 1
im_datagen = ImageDataGenerator(rotation_range = 10, width_shift_range = 0.1, height_shift_range = 0.1, zoom_range = 0.2, horizontal_flip = True)



History = Unet_Model.fit_generator(augmentation_generator(img_train, mask_train, seed, batch_size), steps_per_epoch=np.ceil(float(len(img_train)) / float(batch_size)),
                validation_data = augmentation_generator(img_val, mask_val,seed, batch_size), 
                validation_steps = np.ceil(float(len(img_val)) / float(batch_size)), shuffle=True, epochs=n_epochs)
# Learning curve plots
plt.figure(figsize=(4, 4))
plt.title("Learning curve")
plt.plot(History.history["loss"], label="loss")
plt.plot(History.history["val_loss"], label="val_loss")
plt.plot( np.argmin(History.history["val_loss"]),
         np.min(History.history["val_loss"]),
         marker="x", color="r", label="best model")

plt.xlabel("Epochs")
plt.ylabel("Loss Value")
plt.legend(); 


# In[ ]:


#**********************Task 6 Loading of Data**********************

Path = '/Lab1/Lab3/CT/'
img_train, img_val, mask_train, mask_val = test_train_data(Path,256,256)


# In[10]:


#One hot encoding for masks with 3 classes i.e. left lung, right lung and the background
tf.keras.utils.to_categorical(mask_train,num_classes=3,dtype='float32')
tf.keras.utils.to_categorical(mask_val,num_classes=3,dtype='float32')


# In[11]:


#**********************U-net Model for Task 6**********************

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, Lambda, MaxPooling2D, Dropout, Conv2DTranspose, concatenate, BatchNormalization, Activation
from tensorflow.keras.activations  import relu, sigmoid, softmax
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.metrics import Precision, Recall


batch_size = 8
lr = 0.0001
drp = 0.5
n_epochs = 70

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

    outputs = Conv2D(3, (1,1))(c9)
    outputs = BatchNormalization()(outputs) if batch_norm else outputs
    outputs = Activation('softmax')(outputs)
    
    Unet_Model = Model(inputs, outputs)
    Unet_Model.summary()
    return Unet_Model


# In[12]:


#Task 6 With Data augmentation for CT images
Unet_Model = model_UNet(16,1,256,256, 0,'dr')
Unet_Model.compile(optimizer = Adam(lr = lr), loss = 'categorical_crossentropy', metrics = [dice_coefficient])
batch_size = 8
seed = 1
im_datagen = ImageDataGenerator(rotation_range = 10, width_shift_range = 0.1, height_shift_range = 0.1, zoom_range = 0.2, horizontal_flip = True)

History = Unet_Model.fit_generator(augmentation_generator(img_train, mask_train, seed, batch_size), steps_per_epoch=np.ceil(float(len(img_train)) / float(batch_size)),
                validation_data = augmentation_generator(img_val, mask_val,seed, batch_size), 
                validation_steps = np.ceil(float(len(img_val)) / float(batch_size)), shuffle=True, epochs=n_epochs)

# Learning curve plots
plt.figure(figsize=(4, 4))
plt.title("Learning curve")
plt.plot(History.history["loss"], label="loss")
plt.plot(History.history["val_loss"], label="val_loss")
plt.plot( np.argmin(History.history["val_loss"]),
         np.min(History.history["val_loss"]),
         marker="x", color="r", label="best model")

plt.xlabel("Epochs")
plt.ylabel("Loss Value")
plt.legend(); 


# In[7]:


#Task 7: Brain tumor segmentation in MRI images
#Call the function to load the MRI Brain Data
Path = '/Lab1/Lab3/MRI/'
img_train, img_val, mask_train, mask_val = test_train_data(Path,240,240)


# In[8]:


# Task 7 Compiling and running the model without data augmentation technique
# (Use model from task 5, not 6)
Unet_Model = model_UNet(16,1,240,240, 'batch_norm','dr')
Unet_Model.compile(optimizer = Adam(lr = lr), loss = dice_loss, metrics = [dice_coefficient])
History = Unet_Model.fit(img_train, mask_train, batch_size = batch_size, epochs= n_epochs,  verbose=1, validation_data=(img_val,mask_val))
plt.figure(figsize=(4, 4))
plt.title("Learning curve")
plt.plot(History.history["loss"], label="loss")
plt.plot(History.history["val_loss"], label="val_loss")
plt.plot( np.argmin(History.history["val_loss"]),
         np.min(History.history["val_loss"]),
         marker="x", color="r", label="best model")

plt.xlabel("Epochs")
plt.ylabel("Loss Value")
plt.legend(); 


# In[ ]:


# Task 7 Compiling and running the model with data augmentation technique
Unet_Model = model_UNet(16,1,240,240, 'batch_norm','dr')
Unet_Model.compile(optimizer = Adam(lr = lr), loss = dice_loss, metrics = [dice_coefficient])
batch_size = 8
seed = 1
im_datagen = ImageDataGenerator(rotation_range = 10, width_shift_range = 0.1, height_shift_range = 0.1, zoom_range = 0.2, horizontal_flip = True)



History = Unet_Model.fit_generator(augmentation_generator(img_train, mask_train, seed, batch_size), steps_per_epoch=np.ceil(float(len(img_train)) / float(batch_size)),
                validation_data = augmentation_generator(img_val, mask_val,seed, batch_size), 
                validation_steps = np.ceil(float(len(img_val)) / float(batch_size)), shuffle=True, epochs=n_epochs)

plt.figure(figsize=(4, 4))
plt.title("Learning curve")
plt.plot(History.history["loss"], label="loss")
plt.plot(History.history["val_loss"], label="val_loss")
plt.plot( np.argmin(History.history["val_loss"]),
         np.min(History.history["val_loss"]),
         marker="x", color="r", label="best model")

plt.xlabel("Epochs")
plt.ylabel("Loss Value")
plt.legend(); 


# In[ ]:




