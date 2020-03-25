#!/usr/bin/env python
# coding: utf-8

# # GPU AND LIBRARIES

# In[ ]:


import tensorflow as tf
tf.compat.v1.disable_eager_execution()
tf.config.gpu.set_per_process_memory_fraction(0.6)
tf.config.gpu.set_per_process_memory_growth(True)


# In[ ]:


import os
import numpy as np
from random import shuffle
from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Activation, Dropout, Dense, BatchNormalization, SpatialDropout2D, Input, Concatenate, UpSampling2D
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


# # FUNCTIONS

# In[ ]:


def dice_coef(y_true, y_pred, smooth=1):
    
    
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)


def dice_coef_loss(y_true, y_pred, smooth=1):
    return 1-dice_coef(y_true, y_pred, smooth=1)


# In[ ]:


def get_data(img_path,mask_path,img_w=256,img_h=256):
    data_img = [] 
    data_mask = []
    
    
    combined = list(zip(os.listdir(img_path),os.listdir(mask_path)))
    random.shuffle(combined)
    
    image_files=os.listdir(img_path)
    mask_files=os.listdir(mask_path)
    
    
    
    for i in range(len(image_files)):
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


# In[ ]:


# Path accessing
MRI_Path = '/Lab1/Lab3/MRI'
image_path= MRI_Path+'/Image'
mask_path= MRI_Path+'/Mask'
img_w, img_h = 240,240

# Data loading
images,masks=get_data(image_path,mask_path,img_w,img_h)


# # U-NET MODEL WITHOUT WEIGHT INPUT

# In[ ]:


def unet(input_size = (256,256,1),batch_norm=True,dropout=True,drop_r=0.5,spatial_drop=False,spatial_drop_r=0.1,multi_class=False,classes=3):
    
    # U-Net model. 
    
    # Optionally, introduce:
    # Batch normalization
    # Dropout and dropout rate
    # Spatial dropout and spatial dropout rate
    # Multi-class segmentation and number of classes
    
    
    unet=Sequential()
    inputs = unet.add(Input(input_size))
    conv1 = unet.add(Conv2D(Base, 3, padding = 'same', kernel_initializer = 'he_normal'))
    if batch_norm==True:
        unet.add(BatchNormalization())
    acti1 = unet.add(Activation('relu'))
    conv1 = unet.add(Conv2D(Base, 3, padding = 'same', kernel_initializer = 'he_normal'))
    if batch_norm==True:
        unet.add(BatchNormalization())
    acti1 = unet.add(Activation('relu'))
    if spatial_drop==True:
        unet.add(SpatialDropout2D(spatial_drop_r))
    pool1 = unet.add(MaxPooling2D(pool_size=(2, 2)))
    
    conv2 = unet.add(Conv2D(Base*2, 3, padding = 'same', kernel_initializer = 'he_normal'))
    if batch_norm==True:
        unet.add(BatchNormalization())
    acti2 = unet.add(Activation('relu'))
    conv2 = unet.add(Conv2D(Base*2, 3, padding = 'same', kernel_initializer = 'he_normal'))
    if batch_norm==True:
        unet.add(BatchNormalization())
    acti2 = unet.add(Activation('relu'))
    if spatial_drop==True:
        unet.add(SpatialDropout2D(spatial_drop_r))
    pool2 = unet.add(MaxPooling2D(pool_size=(2, 2)))
    
    conv3 = unet.add(Conv2D(Base*4, 3, padding = 'same', kernel_initializer = 'he_normal'))
    if batch_norm==True:
        unet.add(BatchNormalization())
    acti3 = unet.add(Activation('relu'))
    conv3 = unet.add(Conv2D(Base*4, 3, padding = 'same', kernel_initializer = 'he_normal'))
    if batch_norm==True:
        unet.add(BatchNormalization())
    acti3 = unet.add(Activation('relu'))
    if spatial_drop==True:
        unet.add(SpatialDropout2D(spatial_drop_r))
    pool3 = unet.add(MaxPooling2D(pool_size=(2, 2)))
    
    conv4 = unet.add(Conv2D(Base*8, 3, padding = 'same', kernel_initializer = 'he_normal'))
    if batch_norm==True:
        unet.add(BatchNormalization())
    acti4 = unet.add(Activation('relu'))
    conv4 = unet.add(Conv2D(Base*8, 3, padding = 'same', kernel_initializer = 'he_normal'))
    if batch_norm==True:
        unet.add(BatchNormalization())
    acti4 = unet.add(Activation('relu'))
    if dropout==True:
        drop4 = unet.add(Dropout(drop_r))
    if spatial_drop==True:
        unet.add(SpatialDropout2D(spatial_drop_r))
    pool4 = unet.add(MaxPooling2D(pool_size=(2, 2)))

    conv5 = unet.add(Conv2D(Base*16, 3, padding = 'same', kernel_initializer = 'he_normal'))
    if batch_norm==True:
        unet.add(BatchNormalization())
    acti5= unet.add(Activation('relu'))
    conv5 = unet.add(Conv2D(Base*16, 3, padding = 'same', kernel_initializer = 'he_normal'))
    if batch_norm==True:
        unet.add(BatchNormalization())
    acti5= unet.add(Activation('relu'))
    if dropout==True:
        drop5 = unet.add(Dropout(drop_r))

    up6 = unet.add(Conv2D(Base*8, 2, padding = 'same', kernel_initializer = 'he_normal'))
    if batch_norm==True:
        unet.add(BatchNormalization())
    acti6 = unet.add(Activation('relu'))
    unet.add(UpSampling2D(size = (2,2)))
    merge6 = Concatenate([drop4,up6])
    conv6 = unet.add(Conv2D(Base*8, 3, padding = 'same', kernel_initializer = 'he_normal'))
    if batch_norm==True:
        unet.add(BatchNormalization())
    acti6 = unet.add(Activation('relu'))
    conv6 = unet.add(Conv2D(Base*8, 3, padding = 'same', kernel_initializer = 'he_normal'))
    if batch_norm==True:
        unet.add(BatchNormalization())
    acti6 = unet.add(Activation('relu'))

    up7 = unet.add(Conv2D(Base*4, 2, padding = 'same', kernel_initializer = 'he_normal'))
    if batch_norm==True:
        unet.add(BatchNormalization())
    acti7 = unet.add(Activation('relu'))
    unet.add(UpSampling2D(size = (2,2)))
    merge7 = Concatenate([conv3,up7])
    conv7 = unet.add(Conv2D(Base*4, 3, padding = 'same', kernel_initializer = 'he_normal'))
    if batch_norm==True:
        unet.add(BatchNormalization())
    acti7 = unet.add(Activation('relu'))
    conv7 = unet.add(Conv2D(Base*4, 3, padding = 'same', kernel_initializer = 'he_normal'))
    if batch_norm==True:
        unet.add(BatchNormalization())
    acti7 = unet.add(Activation('relu'))

    up8 = unet.add(Conv2D(Base*2, 2, padding = 'same', kernel_initializer = 'he_normal'))
    if batch_norm==True:
        unet.add(BatchNormalization())
    acti8 = unet.add(Activation('relu'))
    unet.add(UpSampling2D(size = (2,2)))
    merge8 = Concatenate([conv2,up8])
    conv8 = unet.add(Conv2D(Base*2, 3, padding = 'same', kernel_initializer = 'he_normal'))
    if batch_norm==True:
        unet.add(BatchNormalization())
    acti8 = unet.add(Activation('relu'))
    conv8 = unet.add(Conv2D(Base*2, 3, padding = 'same', kernel_initializer = 'he_normal'))
    if batch_norm==True:
        unet.add(BatchNormalization())
    acti8 = unet.add(Activation('relu'))

    up9 = unet.add(Conv2D(Base, 2, padding = 'same', kernel_initializer = 'he_normal'))
    if batch_norm==True:
        unet.add(BatchNormalization())
    acti9 = unet.add(Activation('relu'))
    unet.add(UpSampling2D(size = (2,2)))
    merge9 = Concatenate([conv1,up9])
    conv9 = unet.add(Conv2D(Base, 3, padding = 'same', kernel_initializer = 'he_normal'))
    if batch_norm==True:
        unet.add(BatchNormalization())
    acti9 = unet.add(Activation('relu'))
    conv9 = unet.add(Conv2D(Base, 3, padding = 'same', kernel_initializer = 'he_normal'))
    if batch_norm==True:
        unet.add(BatchNormalization())
    acti9 = unet.add(Activation('relu'))
    conv9 = unet.add(Conv2D(Base, 3, padding = 'same', kernel_initializer = 'he_normal'))
    if batch_norm==True:
        unet.add(BatchNormalization())
    acti9 = unet.add(Activation('relu'))
    if multi_class==False:
        conv10 = unet.add(Conv2D(1, 1, activation = 'sigmoid'))
    else:
        conv10 = unet.add(Conv2D(classes, 1))
        unet.add(BatchNormalization())
        unet.add(Activation('softmax'))
    
    unet.summary()
    return unet


# # Task 1: IMPLEMENT K-FOLD & COMPILE THE MODEL, DISPLAY RESULTS

# In[ ]:


## ********************Fold Dataset and Train the Model********************####


Base=16
k_fold = 3
batch_size = 2
lr = 0.00001
drp = 0.5
n_epochs = 100
 
for train_index,test_index in KFold(k_fold).split(images):
    
    x_train,x_test=images[train_index],images[test_index]
    y_train,y_test=masks[train_index],masks[test_index]


    Unet_Model=unet(input_size = (x_train.shape[1],x_train.shape[2],1),batch_norm=True,dropout=True,drop_r=0.5,spatial_drop=False,spatial_drop_r=0.1,multi_class=False)
    
    Unet_Model.compile(optimizer = Adam(lr = lr), loss = dice_coef_loss, metrics = [dice_coef,
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


# # Task 2: IMPLEMENT WEIGHT MAPS & TRAIN THE MODEL WITH & WITHOUT DATA AUGMENTATION

# In[ ]:


# Dilation and Erosion Checking
mask = imread('/Lab1/Lab3/MRI/Mask/Brats17_TCIA_118_1_t1ce_106_Tumor.png') 
mask_dilation = binary_dilation(mask)
mask_erosion = binary_erosion(mask)
difference =  mask_dilation.astype(int)-mask_erosion.astype(int)
plt.figure()
plt.imshow(mask, cmap='gray')
plt.figure()
plt.imshow(mask_dilation, cmap='gray')
plt.figure()
plt.imshow(mask_erosion, cmap='gray')
plt.figure()
plt.imshow(difference, cmap='gray')


# In[ ]:


#Binary masks

mask_dilation = np.zeros(masks.shape)
mask_erosion = np.zeros(masks.shape)
for i in range(masks.shape[0]):
    mask_dilation[i,:,:,0] = binary_dilation(masks[i,:,:,0])
    mask_erosion[i,:,:,0] = binary_erosion(masks[i,:,:,0])
wt_maps =  mask_dilation.astype(int)-mask_erosion.astype(int)

plt.imshow(difference[10,:,:,0], cmap = 'gray')


# In[ ]:


def weighted_loss(weight_map,weight_strength):
    def weighted_dice_loss(y_true,y_pred):
        y_true_f =K.flatten(y_true)
        y_pred_f =K.flatten(y_pred)
        weight_f =K.flatten(weight_map)
        weight_f =weight_f *weight_strength
        weight_f =1/(weight_f+1)
        weighted_intersection =K.sum(weight_f *(y_true_f *y_pred_f))
        return -(2.*weighted_intersection +K.epsilon())/(K.sum(y_true_f)+K.sum(y_pred_f)+K.epsilon())
    return weighted_dice_loss


# In[ ]:


def U_Net(size = (240,240,1),batch_norm=True,dropout=True,drop_r=0.5,spatial_drop=False,spatial_drop_r=0.1,multi_class=False,classes=3):
    
    # U-Net model. 
    
    # Optionally, introduce:
    # Batch normalization
    # Dropout and dropout rate
    # Spatial dropout and spatial dropout rate
    # Multi-class segmentation and number of classes
    
    #*******************************************C O N T R A C T I O N******************************************#
    U_Net=Sequential()
    
    image = Input(shape=size)
    weight= Input(shape=size)
    inputs= concatenate([image,weight],axis=-1)
    c1_1 = Conv2D(Base, 3, padding = 'same', kernel_initializer = 'he_normal')(inputs)
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

    #*******************************************E X P A N S I O N******************************************#
    
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


# In[ ]:


## ********************Fold Dataset and Train the Model with Weight Maps********************####


Base=16
k_fold = 3
batch_size = 2
lr = 0.00001
drp = 0.5
n_epochs = 150
weight_strength = 1
 
for train_index,test_index in KFold(k_fold).split(images):
    
    x_train,x_test=images[train_index],images[test_index]
    y_train,y_test=masks[train_index],masks[test_index]
    wt_train,wt_test=wt_maps[train_index], wt_maps[test_index]
   


    Unet_Weights_Model=unet_weights(input_size (x_train.shape[1],x_train.shape[2],1),batch_norm=True,dropout=True,drop_r=0.5,spatial_drop=False,spatial_drop_r=0.1,multi_class=False)
    
    Unet_Weights_Model.compile(optimizer = Adam(lr = lr), loss = weighted_loss, metrics = [dice_coef,
                                                                                           tf.keras.metrics.Precision(), 
                                                                                           tf.keras.metrics.Recall()])
    
    History = Unet_Weights_Model.fit([x_train, wt_train], y_train, batch_size = batch_size, epochs= n_epochs,validation_data = (x_test, y_test), 
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
    


# In[ ]:


#Function for Data Augmentation Includion Weights

def combine_generator(gen1, gen2, gen3):
    while True:
        x = gen1.next()
        y = gen2.next()
        w = gen3.next()
        yield([x, w], y)

def generator_with_weights(x_train, y_train, weights_train, batch_size):
    background_value = x_train.min()
    data_gen_args = dict(rotation_range=10.,width_shift_range=0.1,height_shift_range=0.1,cval=background_value,zoom_range=0.2,horizontal_flip=True)
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    weights_datagen = ImageDataGenerator(**data_gen_args)
    seed=1
    image_datagen.fit(x_train, augment=True, seed=seed)
    mask_datagen.fit(y_train, augment=True, seed=seed)
    weights_datagen.fit(weights_train, augment=True, seed=seed)
    image_generator = image_datagen.flow(x_train, shuffle=False,batch_size=batch_size,seed=1)
    mask_generator = mask_datagen.flow(y_train, shuffle=False,batch_size=batch_size,seed=1)
    weight_generator = weights_datagen.flow(weights_train, shuffle=False,batch_size=batch_size,seed=1)
    train_generator = combine_generator(image_generator, mask_generator,weight_generator)
    return train_generator


# In[ ]:


## ********************Fold Dataset and Train the Model With Data Augmentation********************####

Base=16
k_fold = 3
batch_size = 2
lr = 0.00001
drp = 0.5
n_epochs = 150
weight_strength = 1
 
for train_index,test_index in KFold(k_fold).split(images):
    
    x_train,x_test=images[train_index],images[test_index]
    y_train,y_test=masks[train_index],masks[test_index]
    wt_train,wt_test=wt_maps[train_index], wt_maps[test_index]
    train_generator =generator_with_weights(x_train,y_train,wt_train,batch_size) 
    
        
    Unet_Model=unet(input_size (x_train.shape[1],x_train.shape[2],1),batch_norm=True,dropout=True,drop_r=0.5,spatial_drop= False,spatial_drop_r=0.1,multi_class=False)
    
    Unet_Model.compile(optimizer = Adam(lr = lr), loss = weighted_loss(loss_weights,weight_strength), metrics = [dice_coef,
                                                                                           tf.keras.metrics.Precision(), 
                                                                                           tf.keras.metrics.Recall()])
                                                                                           
    model_history = model.fit_generator(train_generator,epochs=n_epochs,validation_data=([x_test, wt_test], y_test),verbose=1)
    
    
    plt.figure(figsize=(4, 4))
    plt.title("Learning curve")
    plt.plot(History.history["loss"], label="loss")
    plt.plot(History.history["val_loss"], label="val_loss")
    plt.plot( np.argmin(History.history["val_loss"]),
             np.min(History.history["val_loss"]),
             marker="x", color="r", label="best model")

    plt.xlabel("Epochs")
    plt.ylabel("Loss Value")
  


# # Task 3: AUTOCONTEXT

# In[ ]:


cv_split=3 # Split for CV
max_cycles=3 # Autocontext cycles
Base=16
k_fold = 3
batch_size = 2
lr = 0.00001
drp = 0.5
n_epochs = 150
weight_strength = 1


for i in range(max_cycles):
    # Define info for extra channel
    if i==0:
        # Initialization for Autocontext
        prior_info=np.zeros(images.shape)+0.5
        segment_results=np.zeros(masks.shape)
    else:
        prior_info=segment_results
    
    for train_index,test_index in KFold(k_fold).split(images):
        x_train, x_test = images[train_index], images[test_index]
        y_train, y_test = masks[train_index], masks[test_index]
        prior_train,prior_test=prior_info[train_index], prior_info[test_index]
        
        # Add extra channel to original inputs
        train_input=extra_channel(x_train,prior_train)
        test_input=extra_channel(x_test,prior_test)
        
        
        

        model=unet(input_size = (x_train.shape[1],x_train.shape[2],2),batch_norm=True,dropout=True,drop_r=0.5,spatial_drop=False,spatial_drop_r=0.1,multi_class=False)


        model.compile(optimizer = Adam(lr = lr), loss = dice_coef_loss, metrics = [dice_coef,
                                                                                           tf.keras.metrics.Precision(), 
                                                                                           tf.keras.metrics.Recall()])
                                                                                           
        model_history = model.fit(train_input,y_train,epochs=n_epochs,validation_data=(test_input,y_test),verbose=1))
        
        
        # Result saving from current validation set
        
        segment_results[test_index]=mri_model.predict(test_input, batch_size=batch_size)
        
        plt.figure(figsize=(4, 4))
        plt.title("Learning curve")
        plt.plot(History.history["loss"], label="loss")
        plt.plot(History.history["val_loss"], label="val_loss")
        plt.plot( np.argmin(History.history["val_loss"]),
                 np.min(History.history["val_loss"]),
                 marker="x", color="r", label="best model")

        plt.xlabel("Epochs")
        plt.ylabel("Loss Value")    

