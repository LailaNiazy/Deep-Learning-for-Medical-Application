# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 17:18:15 2019

@author: looly
"""
##image loader for 3D

import os
import numpy as np
from random import shuffle
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split

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

def data_loader(fold1, fold2, data_path, p,img_h, img_w, img_d):
    
    images, masks = path_loader(fold1, fold2, data_path)
    train_x, test_x, train_y, test_y = get_train_data_shuffled(images, masks, p)
    
    train_img = []
    train_mask = []
    test_img = []
    test_mask = []
  
    for i in range(len(train_x)):
        image_name = train_x[i]
        img = imread(image_name, as_grey=True)
        img = resize(img, (img_h, img_w, img_d), anti_aliasing = True).astype('float32')
        train_img.append([np.array(img)]) 

        if i % 50 == 0:
             print('Reading: {0}/{1}  of train images'.format(i, len(train_x)))
    for j in range(len(train_y)):
        mask_name = train_y[j]
        mask = imread(mask_name, as_grey=True)
        mask = resize(img, (img_h, img_w, img_d), anti_aliasing = True).astype('float32')
        train_mask.append([np.array(mask)])
        
    for i in range(len(test_x)):
        image_name = test_x[i]
        img = imread(image_name, as_grey=True)
        img = resize(img, (img_h, img_w, img_d), anti_aliasing = True).astype('float32')
        test_img.append([np.array(img)]) 

        if i % 50 == 0:
             print('Reading: {0}/{1}  of test images'.format(i, len(test_x)))
                
    for j in range(len(test_y)):
        mask_name = test_y[j]
        mask = imread(mask_name, as_grey=True)
        mask = resize(img, (img_h, img_w,img_d), anti_aliasing = True).astype('float32')
        test_mask.append([np.array(mask)])        
    print('finish')
 
    return train_img, train_mask, test_img, test_mask

#preprocessing
def normalize(image):
    MIN_BOUND = -1000.0
    MAX_BOUND = 500.0
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    print(image.shape)
    image[image>1] = 1.
    image[image<0] = 0.
    return image

# Instantiating images and labels for the model.
def get_train_test_data(fold1, fold2, data_path, p,img_h, img_w, img_d):
    
    train_img, train_mask, test_img, test_mask = data_loader(fold1, fold2, data_path, p,img_h, img_w, img_d)
    Train_Img = np.zeros((len(train_img), img_h, img_w, img_d), dtype = np.float32)
    Test_Img = np.zeros((len(test_img), img_h, img_w, img_d), dtype = np.float32)

    Train_Label = np.zeros((len(train_mask),img_h, img_w, img_d), dtype = np.int32)
    Test_Label = np.zeros((len(test_mask),img_h, img_w, img_d), dtype = np.int32)

    for i in range(len(train_img)):
        Train_Img[i] = normalize(train_img[i][0])
        Train_Label[i] = normalize(train_mask[i][0])

    Train_Img = np.expand_dims(Train_Img, axis = 4)  
    Train_Label = np.expand_dims(Train_Label, axis = 4) 

    for j in range(len(test_img)):
        Test_Img[j] = normalize(test_img[j][0])
        Test_Label[j] = normalize(test_mask[j][0])

    Test_Img = np.expand_dims(Test_Img, axis = 4)
    Test_Label = np.expand_dims(Test_Label, axis = 4)
   

    return Train_Img, Train_Label, Test_Img, Test_Label