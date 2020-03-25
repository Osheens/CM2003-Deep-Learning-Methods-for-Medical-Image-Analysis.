#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
tf.config.gpu.set_per_process_memory_fraction(0.3)
tf.config.gpu.set_per_process_memory_growth(True)


# In[ ]:


# Data Loader
import os
import numpy as np
from random import shuffle
from skimage.io import imread
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


img_w, img_h = 128, 128                                 # Setting the width and heights of the images
data_path = '/Lab1/Bone/'           # Path to data root. Inside this path,
                                                        #two subfolder are placed one for train data and one for test data.


train_data_path = os.path.join(data_path, 'train')   
test_data_path = os.path.join(data_path, 'test')

train_list = os.listdir(train_data_path)
test_list = os.listdir(test_data_path)

# Assigning labels two images; those images contains pattern1 in their filenames
# will be labeled as class 0 and those with pattern2 will be labeled as class 1.
def gen_labels(im_name, pat1, pat2):
        if pat1 in im_name:
            Label = np.array([0])
        elif pat2 in im_name:
            Label = np.array([1])
        return Label

# reading and resizing the training images with their corresponding labels
def train_data(train_data_path, train_list):
    train_img = []       
    for i in range(len(train_list)):
        image_name = train_list[i]
        img = imread(os.path.join(train_data_path, image_name), as_grey=True)
        img = resize(img, (img_h, img_w), anti_aliasing = True).astype('float32')
        train_img.append([np.array(img), gen_labels(image_name, 'AFF', 'NFF')]) 
        
        if i % 200 == 0:
             print('Reading: {0}/{1}  of train images'.format(i, len(train_list)))
             
    shuffle(train_img)
    return train_img

# reading and resizing the testing images with their corresponding labels
def test_data(test_data_path, test_list):
    test_img = []       
    for i in range(len(test_list)):
        image_name = test_list[i]
        img = imread(os.path.join(test_data_path, image_name), as_grey=True)
        img = resize(img, (img_h, img_w), anti_aliasing = True).astype('float32')
        test_img.append([np.array(img), gen_labels(image_name, 'AFF', 'NFF')]) 
        
        if i % 100 == 0:
            print('Reading: {0}/{1} of test images'.format(i, len(test_list)))
             
    shuffle(test_img)   
    return test_img

# Instantiating images and labels for the model.
def get_train_test_data(train_data_path, test_data_path, train_list, test_list):
    
    Train_data = train_data(train_data_path, train_list)
    Test_data = test_data(test_data_path, test_list)
       
    Train_Img = np.zeros((len(train_list), img_h, img_w), dtype = np.float32)
    Test_Img = np.zeros((len(test_list), img_h, img_w), dtype = np.float32)
    
    Train_Label = np.zeros((len(train_list)), dtype = np.int32)
    Test_Label = np.zeros((len(test_list)), dtype = np.int32)
    
    for i in range(len(train_list)):
        Train_Img[i] = Train_data[i][0]
        Train_Label[i] = Train_data[i][1]
        
    Train_Img = np.expand_dims(Train_Img, axis = 3)   
    
    for j in range(len(test_list)):
        Test_Img[j] = Test_data[j][0]
        Test_Label[j] = Test_data[j][1]
        
    Test_Img = np.expand_dims(Test_Img, axis = 3)
        
    return Train_Img, Test_Img, Train_Label, Test_Label

x_train, x_test, y_train, y_test = get_train_test_data(
        train_data_path, test_data_path,
        train_list, test_list)


# In[ ]:


#Task 1 & 2 & 3
# AlexNet Model for skin images
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD, Adam
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation, Dropout, BatchNormalization, SpatialDropout2D

def model(Base, img_ch, img_width, img_height):
    
    model = Sequential()
    
    model.add(Conv2D(filters=Base, input_shape=(img_width, img_height, img_ch),
                     kernel_size=(3,3), strides=(1,1), padding='same'))
    #model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(SpatialDropout2D(0.1))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(filters=Base*2, kernel_size=(3,3), strides=(1,1), padding='same'))
    #model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(SpatialDropout2D(0.1))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(filters=Base*4, kernel_size=(3,3), strides=(1,1), padding='same'))
    #model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(filters=Base*4, kernel_size=(3,3), strides=(1,1), padding='same'))
    #model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(filters=Base*2, kernel_size=(3,3), strides=(1,1), padding='same'))
    #model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(SpatialDropout2D(0.1))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    #model.add(Dropout(0.4))
  

    model.add(Dense(64))
    model.add(Activation('relu'))
    #model.add(Dropout(0.4))


    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.summary()   
    return model
model_AlexNet=model(64,1,128,128)


# In[ ]:



n_epochs = 150
Batch_Size = 8

model_AlexNet.compile(loss='binary_crossentropy',optimizer = Adam(lr=0.00001),metrics=['accuracy'])


History = model_AlexNet.fit(x_train, y_train, batch_size = Batch_Size, epochs= n_epochs, verbose=1, validation_data=(x_test,y_test))
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


#Task 4
#VGG
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation, Dropout, BatchNormalization, SpatialDropout2D


input_shape = (128, 128, 1)
def model_vgg(Base, img_ch, img_width, img_height):
    model_vgg = Sequential()
    
    model_vgg.add(Conv2D(Base, (3, 3), input_shape=(img_width, img_height, img_ch), padding='same'))
    model_vgg.add(BatchNormalization())
    model_vgg.add(Activation('relu'))
    
    model_vgg.add(Conv2D(Base, (3, 3), padding='same'))
    model_vgg.add(BatchNormalization())
    model_vgg.add(Activation('relu'))
    model_vgg.add(SpatialDropout2D(0.1))
    model_vgg.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    model_vgg.add(Conv2D(Base*2, (3, 3),  padding='same'))
    model_vgg.add(BatchNormalization())
    model_vgg.add(Activation('relu'))
    
    model_vgg.add(Conv2D(Base*2, (3, 3), padding='same'))
    model_vgg.add(BatchNormalization())
    model_vgg.add(Activation('relu'))
    model_vgg.add(SpatialDropout2D(0.1))
    model_vgg.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    model_vgg.add(Conv2D(Base*4, (3, 3),padding='same'))
    model_vgg.add(BatchNormalization())
    model_vgg.add(Activation('relu'))
    
    model_vgg.add(Conv2D(Base*4, (3, 3),  padding='same'))
    model_vgg.add(BatchNormalization())
    model_vgg.add(Activation('relu'))
    
    model_vgg.add(Conv2D(Base*4, (3, 3),  padding='same'))
    model_vgg.add(BatchNormalization())
    model_vgg.add(Activation('relu'))
    model_vgg.add(SpatialDropout2D(0.1))
    model_vgg.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    model_vgg.add(Conv2D(Base*8, (3, 3),  padding='same'))
    model_vgg.add(BatchNormalization())
    model_vgg.add(Activation('relu'))
    
    model_vgg.add(Conv2D(Base*8, (3, 3),  padding='same'))
    model_vgg.add(BatchNormalization())
    model_vgg.add(Activation('relu'))
    
    model_vgg.add(Conv2D(Base*8, (3, 3), padding='same'))
    model_vgg.add(BatchNormalization())
    model_vgg.add(Activation('relu'))
    model_vgg.add(SpatialDropout2D(0.1))
    model_vgg.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    model_vgg.add(Conv2D(Base*8, (3, 3),  padding='same'))
    model_vgg.add(BatchNormalization())
    model_vgg.add(Activation('relu'))
    
    model_vgg.add(Conv2D(Base*8, (3, 3), padding='same'))
    model_vgg.add(BatchNormalization())
    model_vgg.add(Activation('relu'))
    
    model_vgg.add(Conv2D(Base*8, (3, 3),  padding='same'))
    model_vgg.add(BatchNormalization())
    model_vgg.add(Activation('relu'))
    model_vgg.add(SpatialDropout2D(0.1))
    model_vgg.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    
    model_vgg.add(Flatten())
    model_vgg.add(Dense(4096))
    model_vgg.add(Activation('relu'))
    model_vgg.add(Dropout(0.4))
    model_vgg.add(Dense(4096))
    model_vgg.add(Activation('relu'))
    model_vgg.add(Dropout(0.4))
    model_vgg.add(Dense(2))
    model_vgg.add(Activation('softmax'))
    

    model_vgg.summary()
    return model_vgg
model_VGG=model_vgg(32,1,128,128)


# In[ ]:


# Compile the model
n_epochs = 150
Batch_Size = 8
LR = 0.001
model_VGG.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=LR), metrics=['accuracy'])
History = model_VGG.fit(x_train, y_train, batch_size = Batch_Size, epochs= n_epochs, verbose=1, validation_data=(x_test,y_test))
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


# Task5a
import numpy as np
from skimage.io import imread
from skimage.transform import rescale
from skimage.transform import rotate
from skimage import exposure
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")

Sample = '/Lab1/X_ray/train/C4_4662.jpg'
Img = imread(Sample)
row, col = Img.shape

def show_paired(Original, Transform, Operation):
    fig, axes = plt.subplots(nrows=1, ncols=2)
    ax = axes.ravel()
    ax[0].imshow(Original, cmap='gray')
    ax[0].set_title("Original image")

    ax[1].imshow(Transform, cmap='gray')
    ax[1].set_title(Operation + " image")
    if Operation == "Rescaled":
        ax[0].set_xlim(0, col)
        ax[0].set_ylim(row, 0)
    else:        
        ax[0].axis('off')
        ax[1].axis('off')
    plt.tight_layout()

# Scaling
scale_factor = 0.8
image_rescaled = rescale(Img, scale_factor)
show_paired(Img, image_rescaled, "Rescaled")

# Roation
Angle = -70
image_rotated = rotate(Img, Angle)
show_paired(Img, image_rotated, "Rotated")

# Horizontal Flip
horizontal_flip = Img[:, ::-1]
show_paired(Img, horizontal_flip, 'Horizontal Flip')

# Vertical Flip
vertical_flip = Img[::-1, :]
show_paired(Img, vertical_flip, 'vertical Flip')


# Intensity rescaling
Min_Per, Max_Per = 5, 95
min_val, max_val = np.percentile(Img, (Min_Per, Max_Per))

better_contrast = exposure.rescale_intensity(Img, in_range=(min_val, max_val))
show_paired(Img, better_contrast, 'Intensity Rescaling')


# In[ ]:


# Task 5b
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img

Sample = '/Lab1/X_ray/train/C4_4662.jpg'
Img = imread(Sample)
Img = np.expand_dims(Img, axis = 2) 
Img = np.expand_dims(Img, axis = 0)


count = 5
MyGen = ImageDataGenerator(rotation_range = 20,
                         width_shift_range = 0.2,
                         horizontal_flip = True)


fix, ax = plt.subplots(1,count+1, figsize=(14,2))
images_flow = MyGen.flow(Img, batch_size=1)
for i, new_images in enumerate(images_flow):
    new_image = array_to_img(new_images[0], scale=True)
    ax[i].imshow(new_image,cmap="gray")
    if i >= count:
        break 


# In[ ]:


# Task 6: Data Augmentation on AlexNet Model for skin images
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD, Adam
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation, Dropout, BatchNormalization, SpatialDropout2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img

Base=64
img_ch=3

def model(img_width, img_height):
    
    model = Sequential()
    
    model.add(Conv2D(filters=Base, input_shape=(img_width, img_height, img_ch),
                     kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(SpatialDropout2D(0.1))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(filters=Base*2, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(SpatialDropout2D(0.1))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(filters=Base*4, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(filters=Base*4, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(filters=Base*2, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(SpatialDropout2D(0.1))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
  

    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))


    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.summary()   
    return model
model_AlexNet_augmentation=model(128,128)

Epochs = 80
LR = 0.00001
batch_size = 8

TRAIN_DIR = '/Lab1/Lab2/Skin/train/'
VAL_DIR = '/Lab1/Lab2/Skin/validation/'
    
train_datagen = ImageDataGenerator(rotation_range = 10, width_shift_range = 0.1, height_shift_range = 0.1, rescale = 1./255, horizontal_flip = True)
train_generator = train_datagen.flow_from_directory(TRAIN_DIR, target_size=(128, 128), batch_size=batch_size,class_mode='binary')

val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(VAL_DIR, target_size=(128, 128), batch_size=batch_size,class_mode='binary')


# In[ ]:


# MyModel.compile(…)
model_AlexNet_augmentation.compile(loss='binary_crossentropy',optimizer = Adam(lr=0.00001),metrics=['binary_accuracy'])

# History = MyModel.fit_generator(…)
History = model_AlexNet_augmentation.fit_generator(train_generator, 
                                                   steps_per_epoch=(len(train_generator) / batch_size), 
                                                   validation_data=val_generator,
                                                   validation_steps=(len(val_generator) / batch_size),
                                                   epochs=Epochs)
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


#Task 7 Apply data augmentation on VGG Model For Skin Dataset
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation, Dropout, BatchNormalization, SpatialDropout2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img

Base=64
img_ch=3

def model_vgg(img_width, img_height):
    model_vgg = Sequential()
    
    model_vgg.add(Conv2D(Base, (3, 3), input_shape=(img_width, img_height, img_ch), padding='same'))
    model_vgg.add(BatchNormalization())
    model_vgg.add(Activation('relu'))
    
    model_vgg.add(Conv2D(Base, (3, 3), padding='same'))
    model_vgg.add(BatchNormalization())
    model_vgg.add(Activation('relu'))
    model_vgg.add(SpatialDropout2D(0.1))
    model_vgg.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    model_vgg.add(Conv2D(Base*2, (3, 3),  padding='same'))
    model_vgg.add(BatchNormalization())
    model_vgg.add(Activation('relu'))
    
    model_vgg.add(Conv2D(Base*2, (3, 3), padding='same'))
    model_vgg.add(BatchNormalization())
    model_vgg.add(Activation('relu'))
    model_vgg.add(SpatialDropout2D(0.1))
    model_vgg.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    model_vgg.add(Conv2D(Base*4, (3, 3),padding='same'))
    model_vgg.add(BatchNormalization())
    model_vgg.add(Activation('relu'))
    
    model_vgg.add(Conv2D(Base*4, (3, 3),  padding='same'))
    model_vgg.add(BatchNormalization())
    model_vgg.add(Activation('relu'))
    
    model_vgg.add(Conv2D(Base*4, (3, 3),  padding='same'))
    model_vgg.add(BatchNormalization())
    model_vgg.add(Activation('relu'))
    model_vgg.add(SpatialDropout2D(0.1))
    model_vgg.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    model_vgg.add(Conv2D(Base*8, (3, 3),  padding='same'))
    model_vgg.add(BatchNormalization())
    model_vgg.add(Activation('relu'))
    
    model_vgg.add(Conv2D(Base*8, (3, 3),  padding='same'))
    model_vgg.add(BatchNormalization())
    model_vgg.add(Activation('relu'))
    
    model_vgg.add(Conv2D(Base*8, (3, 3), padding='same'))
    model_vgg.add(BatchNormalization())
    model_vgg.add(Activation('relu'))
    model_vgg.add(SpatialDropout2D(0.1))
    model_vgg.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    model_vgg.add(Conv2D(Base*8, (3, 3),  padding='same'))
    model_vgg.add(BatchNormalization())
    model_vgg.add(Activation('relu'))
    
    model_vgg.add(Conv2D(Base*8, (3, 3), padding='same'))
    model_vgg.add(BatchNormalization())
    model_vgg.add(Activation('relu'))
    
    model_vgg.add(Conv2D(Base*8, (3, 3),  padding='same'))
    model_vgg.add(BatchNormalization())
    model_vgg.add(Activation('relu'))
    model_vgg.add(SpatialDropout2D(0.1))
    model_vgg.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    
    model_vgg.add(Flatten())
    model_vgg.add(Dense(4096))
    model_vgg.add(Activation('relu'))
    model_vgg.add(Dropout(0.4))
    model_vgg.add(Dense(4096))
    model_vgg.add(Activation('relu'))
    model_vgg.add(Dropout(0.4))
    model_vgg.add(Dense(2))
    model_vgg.add(Activation('softmax'))
    

    model_vgg.summary()
    return model_vgg
model_VGG_augmentation = model_vgg(128,128)

Epochs = 80
LR = 0.00001
batch_size = 8

TRAIN_DIR = '/Lab1/Lab2/Bone/train/'
VAL_DIR = '/Lab1/Lab2/Bone/validation/'
    
train_datagen = ImageDataGenerator(rotation_range = 10, width_shift_range = 0.1, height_shift_range = 0.1, rescale = 1./255, horizontal_flip = True)
train_generator = train_datagen.flow_from_directory(TRAIN_DIR, target_size=(128, 128), batch_size=batch_size,class_mode='binary')

val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(VAL_DIR, target_size=(128, 128), batch_size=batch_size,class_mode='binary')


# In[ ]:


# MyModel.compile(…)
model_VGG_augmentation.compile(loss='sparse_categorical_crossentropy',optimizer = Adam(lr=0.00001),metrics=['accuracy'])

# History = MyModel.fit_generator(…)
History = model_VGG_augmentation.fit_generator(train_generator, 
                                                   steps_per_epoch=(len(train_generator) / batch_size), 
                                                   validation_data=val_generator,
                                                   validation_steps=(len(val_generator) / batch_size),
                                                   epochs=Epochs)
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


# Task 8 and 9
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, ZeroPadding2D
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import applications
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


def get_length(Path, Pattern):
    Length =  len(os.listdir(os.path.join(Path, Pattern)))
    return Length

def VGG_16(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))
    model.summary()
    

# parameters 
train_data_dir = '/Lab1/Lab2/Bone/train/'
validation_data_dir = '/Lab1/Lab2/Bone/validation/'
img_width, img_height = 224, 224
img_ch = 3
epochs = 150
batch_size = 8
LR = 0.00001
Base = 128
# number of data for each class
Len_C1_Train = get_length(train_data_dir,'AFF')
Len_C2_Train = get_length(train_data_dir,'NFF')
Len_C1_Val = get_length(validation_data_dir,'AFF')
Len_C2_Val = get_length(validation_data_dir,'NFF')

# loading the pre-trained model
model = applications.VGG16(include_top=False, weights='imagenet')
model.summary()


# Feature extraction from pretrained VGG (training data)
datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

features_train = model.predict_generator(
        train_generator,
        (Len_C1_Train+Len_C2_Train) // batch_size, max_queue_size=1)


# Feature extraction from pretrained VGG (validation data)
datagen = ImageDataGenerator(rescale=1. / 255)

val_generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

features_validation = model.predict_generator(
        val_generator,
        (Len_C1_Val+Len_C2_Val) // batch_size, max_queue_size=1)


# training a small MLP with extracted features from the pre-trained model
train_data = features_train
train_labels = np.array([0] * int(Len_C1_Train) + [1] * int(Len_C2_Train))

validation_data = features_validation
validation_labels = np.array([0] * int(Len_C1_Val) + [1] * int(Len_C2_Val))

#MLP Model
def model_MLP(img_ch, img_w, img_h):
    model_MLP = Sequential()
    model_MLP.add(Flatten(input_shape=(7,7,512)))
    model_MLP.add(Dense(Base, activation='relu'))
    model_MLP.add(Dropout(0.5))
    model_MLP.add(Dense(1, activation='sigmoid'))
    model_MLP.summary()
    
    return model_MLP
    


# In[ ]:


# Compile and train the model, plot learning curves
Model_MLP=model_MLP(img_ch, img_width, img_height)
Model_MLP.compile(loss='binary_crossentropy',optimizer = SGD(LR),metrics=['accuracy'])

History = Model_MLP.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size, validation_data=(validation_data, validation_labels))

#History = Model_MLP.fit_generator(train_generator,validation_data=val_generator,epochs=epochs)


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


#Task 10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation, Dropout, BatchNormalization, SpatialDropout2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


Base=64
img_ch=3

def model_vgg16(img_width, img_height):
    model_vgg16 = Sequential()
    
    model_vgg16.add(Conv2D(Base, (3, 3), input_shape=(img_width, img_height, img_ch), padding='same'))
    model_vgg16.add(Activation('relu'))
    
    model_vgg16.add(Conv2D(Base, (3, 3), padding='same'))
    model_vgg16.add(Activation('relu'))
    model_vgg16.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    model_vgg16.add(Conv2D(Base*2, (3, 3),  padding='same'))
    model_vgg16.add(Activation('relu'))
    
    model_vgg16.add(Conv2D(Base*2, (3, 3), padding='same'))
    model_vgg16.add(Activation('relu'))
    model_vgg16.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    model_vgg16.add(Conv2D(Base*4, (3, 3),padding='same'))
    model_vgg16.add(Activation('relu'))
    
    model_vgg16.add(Conv2D(Base*4, (3, 3),  padding='same'))
    model_vgg16.add(Activation('relu'))
    
    model_vgg16.add(Conv2D(Base*4, (3, 3),  padding='same'))
    model_vgg16.add(Activation('relu'))
    model_vgg16.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    model_vgg16.add(Conv2D(Base*8, (3, 3),  padding='same'))
    model_vgg16.add(Activation('relu'))
    
    model_vgg16.add(Conv2D(Base*8, (3, 3),  padding='same'))
    model_vgg16.add(Activation('relu'))
    
    model_vgg16.add(Conv2D(Base*8, (3, 3), padding='same'))
    model_vgg16.add(Activation('relu'))
    model_vgg16.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    model_vgg16.add(Conv2D(Base*8, (3, 3),  padding='same'))
    model_vgg16.add(Activation('relu'))
    
    model_vgg16.add(Conv2D(Base*8, (3, 3), padding='same'))
    model_vgg16.add(Activation('relu'))
    
    model_vgg16.add(Conv2D(Base*8, (3, 3),  padding='same', name = 'Last_ConvLayer'))
    model_vgg16.add(Activation('relu'))
    model_vgg16.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    
    model_vgg16.add(Flatten())
    model_vgg16.add(Dense(4096))
    model_vgg16.add(Activation('relu'))
    model_vgg16.add(Dropout(0.4))
    model_vgg16.add(Dense(4096))
    model_vgg16.add(Activation('relu'))
    model_vgg16.add(Dropout(0.4))
    model_vgg16.add(Dense(2))
    model_vgg16.add(Activation('softmax'))
    

    model_vgg16.summary()
    return model_vgg16
Model_VGG16 = model_vgg16(128,128)

Epochs = 5
LR = 0.00001
batch_size = 8

TRAIN_DIR = '/Lab1/Lab2/Bone/train/'
VAL_DIR = '/Lab1/Lab2/Bone/validation/'
    
train_datagen = ImageDataGenerator(rotation_range = 10, width_shift_range = 0.1, height_shift_range = 0.1, rescale = 1./255, horizontal_flip = True)
train_generator = train_datagen.flow_from_directory(TRAIN_DIR, target_size=(128, 128), batch_size=batch_size,class_mode='binary')

val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(VAL_DIR, target_size=(128, 128), batch_size=batch_size,class_mode='binary')


# In[ ]:


# Compile model
Model_VGG16.compile(loss='sparse_categorical_crossentropy',optimizer = Adam(lr=0.00001),metrics=['accuracy'])

# Fir Model
History = Model_VGG16.fit_generator(train_generator, 
                                                   steps_per_epoch=(len(train_generator) / batch_size), 
                                                   validation_data=val_generator,
                                                   validation_steps=(len(val_generator) / batch_size),
                                                   epochs=Epochs)
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


# Task 10
from tensorflow.keras import backend as K
from skimage.io import imread
from skimage.transform import resize
import numpy as np
import os
os.sys.path
import cv2
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
tf.enable_eager_execution()

img_height, img_width = 128,128
Sample = '/Lab1/Lab2/Bone/train/AFF/14.jpg'
Img = imread(Sample)
#Img = Img[:,:,0]
Img = Img/255
Img = resize(Img, (128,128), anti_aliasing = True).astype('float32')
#Img = np.expand_dims(Img, axis = 2) 
Img = np.expand_dims(Img, axis = 0)
preds = Model_VGG16.predict(Img)
class_idx = np.argmax(preds[0])
print(class_idx)
class_output = Model_VGG16.output[:, class_idx]
last_conv_layer = Model_VGG16.get_layer("Last_ConvLayer")

grads = K.gradients(class_output, last_conv_layer.output)[0]
pooled_grads = K.mean(grads, axis=(0, 1, 2))
iterate = K.function([Model_VGG16.input], [pooled_grads, last_conv_layer.output[0]])
pooled_grads_value, conv_layer_output_value = iterate([Img])
for i in range(Base*8):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

heatmap = np.mean(conv_layer_output_value, axis=-1)
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
# For visualization
img = cv2.imread(Sample)

img = cv2.resize(img, (512, 512), interpolation = cv2.INTER_AREA)
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
plt.figure()
plt.imshow(img)
plt.figure()
plt.imshow(superimposed_img)


# In[ ]:




