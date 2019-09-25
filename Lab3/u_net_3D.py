# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 17:20:37 2019

@author: looly
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Input, Convolution3D, MaxPooling3D, Conv3DTranspose
from tensorflow.keras.layers import Reshape, Activation
from tensorflow.keras.layers import BatchNormalization


def u_net3D(img_height, img_width, img_ch, img_d, batchNormalization, Base, k_size=3):
    
    merge_axis = -1 # Feature maps are concatenated along last axis (for tf backend)
    data = Input((img_height, img_width, img_d, img_ch))
    
    #1ConvBlock
    conv1 = Convolution3D(padding='same', filters=Base, kernel_size=k_size)(data)
    if batchNormalization:
        conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    
    conv2 = Convolution3D(padding='same', filters=Base, kernel_size=k_size)(conv1)
    if batchNormalization:
        conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    
    #2ConvBlock
    conv3 = Convolution3D(padding='same', filters=Base*2, kernel_size=k_size)(pool1)
    if batchNormalization:
        conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
   
    conv4 = Convolution3D(padding='same', filters=Base*2, kernel_size=k_size)(conv3)
    if batchNormalization:
        conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)
    
    #3ConvBlock
    conv5 = Convolution3D(padding='same', filters=Base*4, kernel_size=k_size)(pool2)
    if batchNormalization:
        conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    
    conv6 = Convolution3D(padding='same', filters=Base*4, kernel_size=k_size)(conv5)
    if batchNormalization:
        conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)
    
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv6)
    
    #4ConvBlock
    conv7 = Convolution3D(padding='same', filters=Base*8, kernel_size=k_size)(pool3)
    if batchNormalization:
        conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)
    
    conv8 = Convolution3D(padding='same', filters=Base*8, kernel_size=k_size)(conv7)
    if batchNormalization:
        conv8 = BatchNormalization()(conv8)
    conv8 = Activation('relu')(conv8)
    
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv8)
    
    #Bottleneck
    conv9 = Convolution3D(padding='same', filters=Base*16, kernel_size=k_size)(pool4)
    if batchNormalization:
        conv9 = BatchNormalization()(conv9)
    conv9 = Activation('relu')(conv9)
    
    #Expansion
    #1ConvBlock
    up1 = Conv3DTranspose(filters= Base*8, kernel_size=(2, 2, 2),padding = 'same')(conv9)
    merged1 = concatenate([up1, conv8], axis=merge_axis)
    conv10 = Convolution3D(padding='same', filters=Base*8, kernel_size=k_size)(merged1)
    if batchNormalization:
        conv10 = BatchNormalization()(conv10)
    conv10 = Activation('relu')(conv10)
    
    conv11 = Convolution3D(padding='same', filters=Base*8, kernel_size=k_size)(conv10)
    if batchNormalization:
        conv11 = BatchNormalization()(conv11)
    conv11 = Activation('relu')(conv11)
    
    #2ConvBlock
    up2 = Conv3DTranspose(filters = Base*4, kernel_size=(2, 2, 2),padding = 'same')(conv11)
    merged2 = concatenate([up2, conv6], axis=merge_axis)
    conv12 = Convolution3D(padding='same', filters=Base*4, kernel_size=k_size)(merged2)
    if batchNormalization:
        conv12 = BatchNormalization()(conv12)
    conv12 = Activation('relu')(conv12)

    conv13 = Convolution3D(padding='same', filters=Base*4, kernel_size=k_size)(conv12)
    if batchNormalization:
        conv13 = BatchNormalization()(conv13)
    conv13 = Activation('relu')(conv13)
    
    #3ConvBlock
    up3 = Conv3DTranspose(filters = Base*2, kernel_size=(2, 2, 2),padding='same')(conv13)
    merged3 = concatenate([up3, conv4], axis=merge_axis)
    conv14 = Convolution3D(padding='same', filters=Base*2, kernel_size=k_size)(merged3)
    if batchNormalization:
        conv14 = BatchNormalization()(conv14)
    conv14 = Activation('relu')(conv14)
    
    conv15 = Convolution3D(padding='same', filters=64, kernel_size=k_size)(conv14)
    if batchNormalization:
        conv15 = BatchNormalization()(conv15)
    conv15 = Activation('relu')(conv15)

    #4ConvBlock
    up4 = Conv3DTranspose(filters= Base,kernel_size=(2, 2, 2),padding ='same')(conv15)
    merged4 = concatenate([up4, conv2], axis=merge_axis)
    conv16 = Convolution3D(padding='same', filters=Base, kernel_size=k_size)(merged4)
    if batchNormalization:
        conv16 = BatchNormalization()(conv16)
    conv16 = Activation('relu')(conv16)
    
    conv17 = Convolution3D(padding='same', filters=Base, kernel_size=k_size)(conv16)
    if batchNormalization:
        conv17 = BatchNormalization()(conv17)
    conv17 = Activation('relu')(conv17)
    
    #final layer
    conv18 = Convolution3D(padding='same', filters=3, kernel_size=k_size)(merged3)
    if batchNormalization:
        conv18 = BatchNormalization()(conv18)
    output = Reshape([-1, 2])(conv18)
    output = Activation('softmax')(output)
    output = Reshape(inp_shape[:-1] + (2,))(output)

    model = Model(data, output)
    return model