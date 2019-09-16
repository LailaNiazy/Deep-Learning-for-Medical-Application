#!/usr/bin/env python
# coding: utf-8

# # Task 0

# In[1]:


import tensorflow as tf
tf.config.gpu.set_per_process_memory_fraction(0.3)
tf.config.gpu.set_per_process_memory_growth(True)


# In[3]:


##image loader

import os
import numpy as np
from random import shuffle
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def path_loader(fold1, fold2, data_path):
    #Creating data path
    image_data_path = os.path.join(data_path, fold1)   
    mask_data_path = os.path.join(data_path, fold2)
    images = []
    masks = []
    #Listing all file names in the path
    for root, dirs, files in os.walk(image_data_path):
        for name in files:
            images.append(os.path.join(image_data_path,name))
    for root2, dirs2, files2 in os.walk(mask_data_path):
        for name2 in files2:
            masks.append(os.path.join(mask_data_path,name2))
    return images, masks
    

# reading and resizing the training images with their corresponding labels
def get_train_data_shuffled(images, masks, p):
    
    c = list(zip(images, masks))

    shuffle(c)

    images, masks = zip(*c)
    
    train_x, test_x, train_y, test_y = train_test_split(images,masks,test_size = p)

    return train_x, test_x, train_y, test_y 

def data_loader(fold1, fold2, data_path, p,img_h, img_w):
    
    images, masks = path_loader(fold1, fold2, data_path)
    train_x, test_x, train_y, test_y = get_train_data_shuffled(images, masks, p)
    
    train_img = []
    train_mask = []
    test_img = []
    test_mask = []
    len(train_x)
    for i in range(len(train_x)):
        image_name = train_x[i]
        img = imread(image_name, as_grey=True)
        img = resize(img, (img_h, img_w), anti_aliasing = True).astype('float32')
        train_img.append([np.array(img)]) 

        if i % 200 == 0:
             print('Reading: {0}/{1}  of train images'.format(i, len(train_x)))
    for j in range(len(train_y)):
        mask_name = train_y[j]
        mask = imread(mask_name, as_grey=True)
        mask = resize(img, (img_h, img_w), anti_aliasing = True).astype('float32')
        train_mask.append([np.array(mask)])
        
    for i in range(len(test_x)):
        image_name = test_x[i]
        img = imread(image_name, as_grey=True)
        img = resize(img, (img_h, img_w), anti_aliasing = True).astype('float32')
        test_img.append([np.array(img)]) 

        if i % 200 == 0:
             print('Reading: {0}/{1}  of test images'.format(i, len(test_x)))
                
    for j in range(len(test_y)):
        mask_name = test_y[j]
        mask = imread(mask_name, as_grey=True)
        mask = resize(img, (img_h, img_w), anti_aliasing = True).astype('float32')
        test_mask.append([np.array(mask)])        
        
        if j % 200 == 0:
             print('Reading: {0}/{1}  of test images'.format(j, len(test_y)))
 
    return train_img, train_mask, test_img, test_mask

# Instantiating images and labels for the model.
def get_train_test_data(fold1, fold2, data_path, p,img_h, img_w):
    
    train_img, train_mask, test_img, test_mask = data_loader(fold1, fold2, data_path, p,img_h, img_w)

    Train_Img = np.zeros((len(train_img), img_h, img_w), dtype = np.float32)
    Test_Img = np.zeros((len(test_img), img_h, img_w), dtype = np.float32)

    Train_Label = np.zeros((len(train_mask),img_h, img_w), dtype = np.int32)
    Test_Label = np.zeros((len(test_mask),img_h, img_w), dtype = np.int32)

    for i in range(len(train_img)):
        Train_Img[i] = train_img[i][0]
        Train_Label[i] = train_mask[i][0]

    Train_Img = np.expand_dims(Train_Img, axis = 3)  
    Train_Label = np.expand_dims(Train_Label, axis = 3) 

    for j in range(len(test_img)):
        Test_Img[j] = test_img[j][0]
        Test_Label[j] = test_mask[j][0]

    Test_Img = np.expand_dims(Test_Img, axis = 3)
    Test_Label = np.expand_dims(Test_Label, axis = 3)
    print(Train_Img.shape)
    print(Test_Img.shape)

    return Train_Img, Train_Label, Test_Img, Test_Label


# In[12]:


##Data Augmentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img

def DataAugmentation(rotation_range,width_shift,height_shift_range,rescale,horizontal_flip):
    
    #Train data
    train_datagen = ImageDataGenerator(rotation_range = rotation_range, width_shift_range = width_shift, height_shift_range=height_shift_range,
                                       horizontal_flip = horizontal_flip, rescale = rescale)
    #train_generator = train_datagen.flow_from_directory(TRAIN_DIR,target_size=(128, 128), color_mode="grayscale",class_mode='binary')
    
    #Val data
    val_datagen = ImageDataGenerator(rescale = rescale)
    #val_generator = val_datagen.flow_from_directory(VAL_DIR,target_size=(128, 128), color_mode="grayscale",class_mode='binary')
    
    
    return train_datagen, val_datagen 


# In[5]:


##similarity metrics
from sklearn.metrics import recall_score, precision_score
from scipy.spatial.distance import dice


def similarity(metric):
    
    if metric == 'dice':
        m = dice(img1,img2)
    elif metric == 'recall':
        m = recall_score(img1,img2)
    elif metric == 'precision':
        m = precision_score(img1,img2)
        
    return m


# In[6]:


from tensorflow.keras import backend as K
smooth = 1.

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


# In[7]:


##model

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, Input, concatenate, Conv2DTranspose
from tensorflow.keras.layers import Dropout, BatchNormalization, SpatialDropout2D
from tensorflow.keras.layers import MaxPooling2D

def u_net(Base,img_height, img_width, img_ch, batchNormalization, SDRate, spatial_dropout):
    print("building model")
    inputs = Input((img_height, img_width, img_ch))
    #model = Sequential(img_ch,img_width,img_height, batchNormalization, spatial_dropout, SDRate, dropout, dropoutRate)
    ## Contraction
    # Conv Block 1
    
    c1 = Conv2D(filters=Base,
                     kernel_size=(3,3), strides=(1,1), padding='same')(inputs)
    
     #Add batch Normalization
    if batchNormalization:
        c1 = BatchNormalization(axis=-1)(c1)
    
    a1 = Activation('relu')(c1)
    
    #Add spatial Dropout
    if spatial_dropout:
        a1 = SpatialDropout2D(SDRate)(a1)
        
    c2 = Conv2D(filters=Base,
                     kernel_size=(3,3), strides=(1,1), padding='same')(a1)
    
     #Add batch Normalization
    if batchNormalization:
        c2 = BatchNormalization(axis=-1)(c2)
    
    a2 = Activation('relu')(c2)
    
    #Add spatial Dropout
    if spatial_dropout:
        a2 = SpatialDropout2D(SDRate)(a2)
        
    m1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(a2)
        
    # Conv Block 2
    c3 = Conv2D(filters=Base*2,
                     kernel_size=(3,3), strides=(1,1), padding='same')(m1)
    
     #Add batch Normalization
    if batchNormalization:
        c3 = BatchNormalization(axis=-1)(c3)
    
    a3 = Activation('relu')(c3)
    
    #Add spatial Dropout
    if spatial_dropout:
        a3 = SpatialDropout2D(SDRate)(a3)
    
    c4 = Conv2D(filters=Base*2,
                     kernel_size=(3,3), strides=(1,1), padding='same')(a3)
    
     #Add batch Normalization
    if batchNormalization:
        c4 = BatchNormalization(axis=-1)(c4)
    
    a4 = Activation('relu')(c4)
    
    #Add spatial Dropout
    if spatial_dropout:
        a4 = SpatialDropout2D(SDRate)(a4)
    
    m2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(a4)
    
    # Conv Block 3
    c5 = Conv2D(filters=Base*4, 
                     kernel_size=(3,3), strides=(1,1), padding='same')(m2)
    
     #Add batch Normalization
    if batchNormalization:
        c5 = BatchNormalization(axis=-1)(c5)
    
    a5 = Activation('relu')(c5)
    
    #Add spatial Dropout
    if spatial_dropout:
        a5 = SpatialDropout2D(SDRate)(a5)
        
    c6 = Conv2D(filters=Base*4,
                     kernel_size=(3,3), strides=(1,1), padding='same')(a5)
    
     #Add batch Normalization
    if batchNormalization:
          c6 = BatchNormalization(axis=-1)(c6)
    
    a6 = Activation('relu')(c6)
    
    #Add spatial Dropout
    if spatial_dropout:
        a6 = SpatialDropout2D(SDRate)(a6)
        
    m3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(a6)
    
    # Conv Block 4
    c7 = Conv2D(filters=Base*8, 
                     kernel_size=(3,3), strides=(1,1), padding='same')(m3)
    
     #Add batch Normalization
    if batchNormalization:
        c7 = BatchNormalization(axis=-1)(c7)
    
    a7 = Activation('relu')(c7)
    
    #Add spatial Dropout
    if spatial_dropout:
        a7 = SpatialDropout2D(SDRate)(a7)
        
    c8 = Conv2D(filters=Base*8,
                     kernel_size=(3,3), strides=(1,1), padding='same')(a7)
    
     #Add batch Normalization
    if batchNormalization:
        c8 = BatchNormalization(axis=-1)(c8)
    
    a8 = Activation('relu')(c8)
    
    #Add spatial Dropout
    if spatial_dropout:
        a8 = SpatialDropout2D(SDRate)(a8)
        
    m4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(a8)
    
    ##Bottleneck
    # Conv Layer
    c9 = Conv2D(filters=Base*16, 
                     kernel_size=(3,3), strides=(1,1), padding='same')(m4)
    
     #Add batch Normalization
    if batchNormalization:
        c9 = BatchNormalization(axis=-1)(c9)
    
    a9 = Activation('relu')(c9)
    
    #Add spatial Dropout
    if spatial_dropout:
        a9 = SpatialDropout2D(SDRate)(a9)
        
    ##Expansion
    #Conv Block 1
    c10 = Conv2DTranspose(filters=Base*8,
                     kernel_size=(2,2), strides=(2,2), padding='same')(a9)
    c10 = concatenate([a8,c10])
    
    c11 = Conv2D(filters=Base*8,
                     kernel_size=(3,3), strides=(1,1), padding='same')(c10)
    
     #Add batch Normalization
    if batchNormalization:
        c11 = BatchNormalization(axis=-1)(c11)
    
    a10 = Activation('relu')(c11)
    
    #Add spatial Dropout
    if spatial_dropout:
        a10 = SpatialDropout2D(SDRate)(a10)
        
    c12 = Conv2D(filters=Base*8, 
                     kernel_size=(3,3), strides=(1,1), padding='same')(a10)
    
     #Add batch Normalization
    if batchNormalization:
        c12 = BatchNormalization(axis=-1)(c12)
    
    a11 = Activation('relu')(c12)
    
    #Add spatial Dropout
    if spatial_dropout:
        a11 = SpatialDropout2D(SDRate)(a11)
        
    
    #Conv Block 2
    c13 = Conv2DTranspose(filters=Base*4,
                     kernel_size=(2,2), strides=(2,2), padding='same')(a11)
    c13 = concatenate([a6,c13])
    
    c14 = Conv2D(filters=Base*4,
                     kernel_size=(3,3), strides=(1,1), padding='same')(c13)
    
     #Add batch Normalization
    if batchNormalization:
        c14 = BatchNormalization(axis=-1)(c14)
    
    a12 = Activation('relu')(c14)
    
    #Add spatial Dropout
    if spatial_dropout:
        a12 = SpatialDropout2D(SDRate)(a12)
        
    c15 = Conv2D(filters=Base*4, 
                     kernel_size=(3,3), strides=(1,1), padding='same')(a12)
    
     #Add batch Normalization
    if batchNormalization:
        c15 = BatchNormalization(axis=-1)(c15)
    
    a13 = Activation('relu')(c15)
    
    #Add spatial Dropout
    if spatial_dropout:
        a13 = SpatialDropout2D(SDRate)(a13)
        
    
    #Conv Block 3
    c16 = Conv2DTranspose(filters=Base*2,
                     kernel_size=(2,2), strides=(2,2), padding='same')(a13)
    c16 = concatenate([a4,c16])
    
    c17 = Conv2D(filters=Base*2, 
                     kernel_size=(3,3), strides=(1,1), padding='same')(c16)
    
     #Add batch Normalization
    if batchNormalization:
        c17 = BatchNormalization(axis=-1)(c17)
    
    a14 = Activation('relu')(c17)
    
    #Add spatial Dropout
    if spatial_dropout:
        a14 = SpatialDropout2D(SDRate)(a14)
        
    c18 = Conv2D(filters=Base*2,
                     kernel_size=(3,3), strides=(1,1), padding='same')(a14)
    
     #Add batch Normalization
    if batchNormalization:
        c18 = BatchNormalization(axis=-1)(c18)
    
    a15 = Activation('relu')(c18)
    
    #Add spatial Dropout
    if spatial_dropout:
        a15 = SpatialDropout2D(SDRate)(a15)
        
    
    #Conv Block 4
    c19 = Conv2DTranspose(filters=Base,
                     kernel_size=(2,2), strides=(2,2), padding='same')(a15)
    c19 = concatenate([a2,c19])
    
    c20 = Conv2D(filters=Base,
                     kernel_size=(3,3), strides=(1,1), padding='same')(c19)
    
     #Add batch Normalization
    if batchNormalization:
        c20 = BatchNormalization(axis=-1)(c20)
    
    a16 = Activation('relu')(c20)
    
    #Add spatial Dropout
    if spatial_dropout:
        a16 = SpatialDropout2D(SDRate)(a16)
        
    c21 = Conv2D(filters=Base,
                     kernel_size=(3,3), strides=(1,1), padding='same')(a16)
    
     #Add batch Normalization
    if batchNormalization:
        c21 = BatchNormalization(axis=-1)(c21)
    
    a17 = Activation('relu')(c21)
    
    #final layer
    c22 = Conv2D(1, kernel_size=(3,3), strides=(1,1), padding='same')(a17)
    a18 = Activation('sigmoid')(c22)
    
    model = Model(inputs,a18)
    
    model.summary()
    return model


# In[21]:


##model

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, Input, concatenate, Conv2DTranspose
from tensorflow.keras.layers import Dropout, BatchNormalization, SpatialDropout2D
from tensorflow.keras.layers import MaxPooling2D

def u_net_3labels(Base,img_height, img_width, img_ch, batchNormalization, SDRate, spatial_dropout):
    print("building model")
    inputs = Input((img_height, img_width, img_ch))
    #model = Sequential(img_ch,img_width,img_height, batchNormalization, spatial_dropout, SDRate, dropout, dropoutRate)
    ## Contraction
    # Conv Block 1
    
    c1 = Conv2D(filters=Base,
                     kernel_size=(3,3), strides=(1,1), padding='same')(inputs)
    
     #Add batch Normalization
    if batchNormalization:
        c1 = BatchNormalization(axis=-1)(c1)
    
    a1 = Activation('relu')(c1)
    
    #Add spatial Dropout
    if spatial_dropout:
        a1 = SpatialDropout2D(SDRate)(a1)
        
    c2 = Conv2D(filters=Base,
                     kernel_size=(3,3), strides=(1,1), padding='same')(a1)
    
     #Add batch Normalization
    if batchNormalization:
        c2 = BatchNormalization(axis=-1)(c2)
    
    a2 = Activation('relu')(c2)
    
    #Add spatial Dropout
    if spatial_dropout:
        a2 = SpatialDropout2D(SDRate)(a2)
        
    m1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(a2)
        
    # Conv Block 2
    c3 = Conv2D(filters=Base*2,
                     kernel_size=(3,3), strides=(1,1), padding='same')(m1)
    
     #Add batch Normalization
    if batchNormalization:
        c3 = BatchNormalization(axis=-1)(c3)
    
    a3 = Activation('relu')(c3)
    
    #Add spatial Dropout
    if spatial_dropout:
        a3 = SpatialDropout2D(SDRate)(a3)
    
    c4 = Conv2D(filters=Base*2,
                     kernel_size=(3,3), strides=(1,1), padding='same')(a3)
    
     #Add batch Normalization
    if batchNormalization:
        c4 = BatchNormalization(axis=-1)(c4)
    
    a4 = Activation('relu')(c4)
    
    #Add spatial Dropout
    if spatial_dropout:
        a4 = SpatialDropout2D(SDRate)(a4)
    
    m2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(a4)
    
    # Conv Block 3
    c5 = Conv2D(filters=Base*4, 
                     kernel_size=(3,3), strides=(1,1), padding='same')(m2)
    
     #Add batch Normalization
    if batchNormalization:
        c5 = BatchNormalization(axis=-1)(c5)
    
    a5 = Activation('relu')(c5)
    
    #Add spatial Dropout
    if spatial_dropout:
        a5 = SpatialDropout2D(SDRate)(a5)
        
    c6 = Conv2D(filters=Base*4,
                     kernel_size=(3,3), strides=(1,1), padding='same')(a5)
    
     #Add batch Normalization
    if batchNormalization:
          c6 = BatchNormalization(axis=-1)(c6)
    
    a6 = Activation('relu')(c6)
    
    #Add spatial Dropout
    if spatial_dropout:
        a6 = SpatialDropout2D(SDRate)(a6)
        
    m3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(a6)
    
    # Conv Block 4
    c7 = Conv2D(filters=Base*8, 
                     kernel_size=(3,3), strides=(1,1), padding='same')(m3)
    
     #Add batch Normalization
    if batchNormalization:
        c7 = BatchNormalization(axis=-1)(c7)
    
    a7 = Activation('relu')(c7)
    
    #Add spatial Dropout
    if spatial_dropout:
        a7 = SpatialDropout2D(SDRate)(a7)
        
    c8 = Conv2D(filters=Base*8,
                     kernel_size=(3,3), strides=(1,1), padding='same')(a7)
    
     #Add batch Normalization
    if batchNormalization:
        c8 = BatchNormalization(axis=-1)(c8)
    
    a8 = Activation('relu')(c8)
    
    #Add spatial Dropout
    if spatial_dropout:
        a8 = SpatialDropout2D(SDRate)(a8)
        
    m4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(a8)
    
    ##Bottleneck
    # Conv Layer
    c9 = Conv2D(filters=Base*16, 
                     kernel_size=(3,3), strides=(1,1), padding='same')(m4)
    
     #Add batch Normalization
    if batchNormalization:
        c9 = BatchNormalization(axis=-1)(c9)
    
    a9 = Activation('relu')(c9)
    
    #Add spatial Dropout
    if spatial_dropout:
        a9 = SpatialDropout2D(SDRate)(a9)
        
    ##Expansion
    #Conv Block 1
    c10 = Conv2DTranspose(filters=Base*8,
                     kernel_size=(2,2), strides=(2,2), padding='same')(a9)
    c10 = concatenate([a8,c10])
    
    c11 = Conv2D(filters=Base*8,
                     kernel_size=(3,3), strides=(1,1), padding='same')(c10)
    
     #Add batch Normalization
    if batchNormalization:
        c11 = BatchNormalization(axis=-1)(c11)
    
    a10 = Activation('relu')(c11)
    
    #Add spatial Dropout
    if spatial_dropout:
        a10 = SpatialDropout2D(SDRate)(a10)
        
    c12 = Conv2D(filters=Base*8, 
                     kernel_size=(3,3), strides=(1,1), padding='same')(a10)
    
     #Add batch Normalization
    if batchNormalization:
        c12 = BatchNormalization(axis=-1)(c12)
    
    a11 = Activation('relu')(c12)
    
    #Add spatial Dropout
    if spatial_dropout:
        a11 = SpatialDropout2D(SDRate)(a11)
        
    
    #Conv Block 2
    c13 = Conv2DTranspose(filters=Base*4,
                     kernel_size=(2,2), strides=(2,2), padding='same')(a11)
    c13 = concatenate([a6,c13])
    
    c14 = Conv2D(filters=Base*4,
                     kernel_size=(3,3), strides=(1,1), padding='same')(c13)
    
     #Add batch Normalization
    if batchNormalization:
        c14 = BatchNormalization(axis=-1)(c14)
    
    a12 = Activation('relu')(c14)
    
    #Add spatial Dropout
    if spatial_dropout:
        a12 = SpatialDropout2D(SDRate)(a12)
        
    c15 = Conv2D(filters=Base*4, 
                     kernel_size=(3,3), strides=(1,1), padding='same')(a12)
    
     #Add batch Normalization
    if batchNormalization:
        c15 = BatchNormalization(axis=-1)(c15)
    
    a13 = Activation('relu')(c15)
    
    #Add spatial Dropout
    if spatial_dropout:
        a13 = SpatialDropout2D(SDRate)(a13)
        
    
    #Conv Block 3
    c16 = Conv2DTranspose(filters=Base*2,
                     kernel_size=(2,2), strides=(2,2), padding='same')(a13)
    c16 = concatenate([a4,c16])
    
    c17 = Conv2D(filters=Base*2, 
                     kernel_size=(3,3), strides=(1,1), padding='same')(c16)
    
     #Add batch Normalization
    if batchNormalization:
        c17 = BatchNormalization(axis=-1)(c17)
    
    a14 = Activation('relu')(c17)
    
    #Add spatial Dropout
    if spatial_dropout:
        a14 = SpatialDropout2D(SDRate)(a14)
        
    c18 = Conv2D(filters=Base*2,
                     kernel_size=(3,3), strides=(1,1), padding='same')(a14)
    
     #Add batch Normalization
    if batchNormalization:
        c18 = BatchNormalization(axis=-1)(c18)
    
    a15 = Activation('relu')(c18)
    
    #Add spatial Dropout
    if spatial_dropout:
        a15 = SpatialDropout2D(SDRate)(a15)
        
    
    #Conv Block 4
    c19 = Conv2DTranspose(filters=Base,
                     kernel_size=(2,2), strides=(2,2), padding='same')(a15)
    c19 = concatenate([a2,c19])
    
    c20 = Conv2D(filters=Base,
                     kernel_size=(3,3), strides=(1,1), padding='same')(c19)
    
     #Add batch Normalization
    if batchNormalization:
        c20 = BatchNormalization(axis=-1)(c20)
    
    a16 = Activation('relu')(c20)
    
    #Add spatial Dropout
    if spatial_dropout:
        a16 = SpatialDropout2D(SDRate)(a16)
        
    c21 = Conv2D(filters=Base,
                     kernel_size=(3,3), strides=(1,1), padding='same')(a16)
    
     #Add batch Normalization
    if batchNormalization:
        c21 = BatchNormalization(axis=-1)(c21)
    
    a17 = Activation('relu')(c21)
    
    #final layer
    c22 = Conv2D(3, kernel_size=(3,3), strides=(1,1), padding='same')(a17)
    a18 = Activation('softmax')(c22)
    
    model = Model(inputs,a18)
    
    model.summary()
    return model


# In[16]:


##plotter
import matplotlib.pyplot as plt
def plotter(History, dice, recall,precision):
    #Training vs Validation Learning loss 
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
"""
    #Train and test accuracy plot
    plt.figure(figsize=(4,4))
    plt.title("Accuracy Learning Curve")
    plt.plot(History.history["binary_accuracy"], label="binary_accuracy")
    plt.plot(History.history["val_binary_accuracy"], label="val_binary_accuracy")
    #plt.plot(History.history["categorical_accuracy"], label="categorical_accuracy")
    #plt.plot(History.history["val_categorical_accuracy"], label="val_categorical_accuracy")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(); 
"""


# # Task 1a

# In[ ]:


from tensorflow.keras.optimizers import SGD, Adam

base = 16
image_size = 256
img_ch = 1
batch_size =8
LR = 0.0001
SDRate = 0.5
batch_normalization = True
spatial_dropout = True
metric = 'dice'
epochs = 150
p = 0.2
path = '/Lab1/Lab3/X_ray/'
fold1 = 'Image'
fold2 = 'Mask'

train_img, train_mask, test_img, test_mask = get_train_test_data(fold1, fold2, path, p,image_size, image_size)
model = u_net(base,image_size, image_size, img_ch, batch_normalization, SDRate, spatial_dropout)
print("finished")

model.compile(optimizer = Adam(lr=LR), loss = 'binary_crossentropy', metrics =[dice_coef] )
History = model.fit(train_img, train_mask, epochs = epochs, batch_size = batch_size, verbose = 1,
                    validation_data = (test_img,test_mask))
plotter(History)



# In[ ]:


get_ipython().system(" ls '/Lab1/Lab3/'")


# ### Task 5a

# In[9]:


from tensorflow.keras.optimizers import SGD, Adam

base = 16
image_size = 256
img_ch = 1
batch_size =8
LR = 0.0001
SDRate = 0.5
batch_normalization = True
spatial_dropout = True
metric = 'dice'
epochs = 150
p = 0.2
path = '/Lab1/Lab3/CT/'
fold1 = 'Image'
fold2 = 'Mask'

train_img, train_mask, test_img, test_mask = get_train_test_data(fold1, fold2, path, p,image_size, image_size)
model = u_net(base,image_size, image_size, img_ch, batch_normalization, SDRate, spatial_dropout)
print("finished")

model.compile(optimizer = Adam(lr=LR), loss = 'binary_crossentropy', metrics =[dice_coef] )
History = model.fit(train_img, train_mask, epochs = epochs, batch_size = batch_size, verbose = 1,
                    validation_data = (test_img,test_mask))
dice = True
recall = False
precision = False
plotter(History, dice, recall, precision)


# ### Task 5b

# In[17]:


from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.metrics import Recall, Precision

#Model parameters
base = 16
image_size = 256
img_ch = 1
batch_size =8
LR = 0.0001
SDRate = 0.5
batch_normalization = True
spatial_dropout = True
metric = 'dice'
epochs = 150

#Data loader parameters
p = 0.2
path = '/Lab1/Lab3/CT/'
fold1 = 'Image'
fold2 = 'Mask'

#Data augmentation parameters
rotation_range = 10
width_shift = 0.1
height_shift_range = 0.1,
rescale = 1./255
horizontal_flip = True

#Load the data
train_img, train_mask, test_img, test_mask = get_train_test_data(fold1, fold2, path, p,image_size, image_size)

#Data augmentation
train_datagen, val_datagen = DataAugmentation(rotation_range,width_shift,height_shift_range,rescale,horizontal_flip)


#Build the model
model = u_net(base,image_size, image_size, img_ch, batch_normalization, SDRate, spatial_dropout)

#Compile the model
model.compile(optimizer = Adam(lr=LR), loss = 'binary_crossentropy', metrics =[dice_coef, Recall(), Precision()] )

#Fit the data into the model
History = model.fit_generator(train_datagen.flow(train_img, train_mask,batch_size = batch_size), validation_data = val_datagen.flow(test_img, test_mask), epochs = epochs, verbose = 1)        

#Plot results
dice = True
recall = True
precision = True
plotter(History, dice, recall, precision)


# ### Task 6

# In[29]:


from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras.utils import to_categorical
#Model parameters
base = 16
image_size = 256
img_ch = 1
batch_size =8
LR = 0.0001
SDRate = 0.5
batch_normalization = True
spatial_dropout = True
epochs = 150

#Data loader parameters
p = 0.2
path = '/Lab1/Lab3/CT/'
fold1 = 'Image'
fold2 = 'Mask'

#Data augmentation parameters
rotation_range = 10
width_shift = 0.1
height_shift_range = 0.1,
rescale = 1./255
horizontal_flip = True

#Load the data
train_img, train_mask, test_img, test_mask = get_train_test_data(fold1, fold2, path, p,image_size, image_size)

#To one-hot-encoding
train_mask = to_categorical(train_mask, num_classes=3)
test_mask = to_categorical(test_mask, num_classes=3)

#Build the multi-classification model
model = u_net_3labels(base,image_size, image_size, img_ch, batch_normalization, SDRate, spatial_dropout)

#Compile the model
model.compile(optimizer = Adam(lr=LR), loss = 'categorical_crossentropy', metrics =[dice_coef, Recall(), Precision()] )

#Fit the data into the model
History = model.fit_generator(train_datagen.flow(train_img, train_mask,batch_size = batch_size), validation_data = val_datagen.flow(test_img, test_mask), epochs = epochs, verbose = 1)          

#Plot results
dice = True
recall = True
precision = True
plotter(History, dice, recall, precision)


# ### Task 7

# In[32]:


from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras.utils import to_categorical

#Model parameters
base = 16
image_size = 240
img_ch = 1
batch_size =8
LR = 0.0001
SDRate = 0.5
batch_normalization = True
spatial_dropout = True
epochs = 150

#Data loader parameters
p = 0.2
path = '/Lab1/Lab3/MRI/'
fold1 = 'Image'
fold2 = 'Mask'

#Data augmentation parameters
rotation_range = 10
width_shift = 0.1
height_shift_range = 0.1,
rescale = 1./255
horizontal_flip = True

#Load the data
train_img, train_mask, test_img, test_mask = get_train_test_data(fold1, fold2, path, p,image_size, image_size)

#Build the multi-classification model
model = u_net(base,image_size, image_size, img_ch, batch_normalization, SDRate, spatial_dropout)

#Compile the model
model.compile(optimizer = Adam(lr=LR), loss = 'binary_crossentropy', metrics =[dice_coef, Recall(), Precision()] )

#Fit the data into the model
History = model.fit_generator(train_datagen.flow(train_img, train_mask,batch_size = batch_size), validation_data = val_datagen.flow(test_img, test_mask), epochs = epochs, verbose = 1)          

#Plot results
dice = True
recall = True
precision = True
plotter(History, dice, recall, precision)


# In[ ]:




