# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 22:36:08 2019

@author: looly
"""

##model

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv2D, Input, concatenate, Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization, SpatialDropout2D
from tensorflow.keras.layers import MaxPooling2D, Reshape, ConvLSTM2D
import numpy as np
def u_net(Base,img_height, img_width, img_ch, batchNormalization, SDRate, spatial_dropout, number_of_labels):
    
    inputs = Input((img_height, img_width, img_ch))
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
    # up-sampling:
    c10 = Conv2DTranspose(filters=Base*8,kernel_size= (3, 3), strides=(2, 2), padding='same')(a9)
    # reshaping:
    x1 = Reshape(target_shape=(1, np.int32(img_height/8), np.int32(img_width/8), Base * 8))(a8)
    x2 = Reshape(target_shape=(1, np.int32(img_height/8), np.int32(img_width/8), Base * 8))(c10)
    # concatenation:
    c10 = concatenate([x1, x2], axis=1)
    # LSTM:
    c10 = ConvLSTM2D(filters=Base*4, kernel_size=(3, 3), padding='same', return_sequences=False, go_backwards=True)(c10)
    
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
    # up-sampling:
    c13 = Conv2DTranspose(filters=Base*4, kernel_size=(2,2), strides=(2,2), padding='same')(a11)
    # reshaping:
    x3 = Reshape(target_shape=(1, np.int32(img_height/4), np.int32(img_width/4),Base * 4))(a6)
    x4 = Reshape(target_shape=(1, np.int32(img_height/4), np.int32(img_width/4), Base * 4))(c13)
    # concatenation:
    c13 = concatenate([x3, x4], axis=1)
    # LSTM:
    c13 = ConvLSTM2D(filters=Base*4, kernel_size=(3, 3), padding='same', return_sequences=False, go_backwards=True)(c13)
    
    c14 = Conv2D(filters=Base*4,kernel_size=(3,3), strides=(1,1), padding='same')(c13)
    
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
    # up-sampling:
    c16 = Conv2DTranspose(filters=Base*2,
                     kernel_size=(2,2), strides=(2,2), padding='same')(a13)
    # reshaping:
    x5 = Reshape(target_shape=(1, np.int32(img_height/2), np.int32(img_width/2), Base * 2))(a4)
    x6 = Reshape(target_shape=(1, np.int32(img_height/2), np.int32(img_width/2), Base * 2))(c16)
    # concatenation:
    c16 = concatenate([x5, x6], axis=1)
    # LSTM:
    c16 = ConvLSTM2D(filters=Base*4, kernel_size= (3, 3), padding='same', return_sequences=False, go_backwards=True)(c16)
    
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
    # up-sampling:
    c19 = Conv2DTranspose(filters=Base,
                     kernel_size=(2,2), strides=(2,2), padding='same')(a15)
    # reshaping:
    x7 = Reshape(target_shape=(1, np.int32(img_height), np.int32(img_width), Base))(a2)
    x8 = Reshape(target_shape=(1, np.int32(img_height), np.int32(img_width), Base))(c19)
    # concatenation:
    c19 = concatenate([x7, x8], axis=1)
    # LSTM:
    c19 = ConvLSTM2D(filters=Base*4, kernel_size=(3, 3), padding='same', return_sequences=False, go_backwards=True)(c19)
    
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
    c22 = Conv2D(number_of_labels, kernel_size=(3,3), strides=(1,1), padding='same')(a17)
    a18 = Activation('sigmoid')(c22)
    
    model = Model(inputs,a18)
    
    model.summary()
    return model