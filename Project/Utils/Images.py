import cv2 as cv2
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np
from random import shuffle
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split

import tensorflow as tf


tf.config.gpu.set_per_process_memory_fraction(0.3)
tf.config.gpu.set_per_process_memory_growth(True)

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


def create_weight_map(mask, radius, i):
 
    mask = np.uint16(mask)
    # morphology kernel
    kernel = np.ones((radius*2+1,radius*2+1))
    
    # dilate the mask -radius=2
    dilate = cv2.dilate(mask, kernel)
    
    # erode the mask
    erosion = cv2.erode(mask, kernel)
    
    # substract the eroded image from the dilated image
    substraction = dilate-erosion

    return substraction

# reading and resizing the training images with their corresponding labels
def data_loader(fold1, fold2, data_path,img_h, img_w):
    
    train_x, train_y = path_loader(fold1, fold2, data_path)
    #train_x, train_y = get_train_data_shuffled(images, masks)
    
    train_img = []
    train_mask = []

    for i in range(len(train_x)):
        image_name = train_x[i]
        img = imread(image_name, as_grey=True)
        img = resize(img, (img_h, img_w), anti_aliasing = True).astype('float32')
        img = img/255
        img -= img.mean()
        img /= img.std()
        train_img.append([np.array(img)]) 

        if i % 50 == 0:
             print('Reading: {0}/{1}  of train images'.format(i, len(train_x)))
                
    for j in range(len(train_y)):
        mask_name = train_y[j]
        mask = imread(mask_name, as_grey=True)
        mask = resize(mask, (img_h, img_w), anti_aliasing = True).astype('float32')
        train_mask.append([np.array(mask)])
        
        if j % 50 == 0:
             print('Reading: {0}/{1}  of train images'.format(j, len(train_y)))
        
    return train_img, train_mask

# Instantiating images and labels for the model.
def get_train_test_data(fold1, fold2, data_path, img_h, img_w):
    
    train_img, train_mask = data_loader(fold1, fold2, data_path, img_h, img_w)

    Train_Img = np.zeros((len(train_img), img_h, img_w), dtype = np.float32)
    Train_Label = np.zeros((len(train_mask),img_h, img_w), dtype = np.int32)
    Weight_Map = np.zeros((len(train_mask),img_h, img_w), dtype = np.int32)
    
    for i in range(len(train_img)):
        Train_Img[i] = train_img[i][0]
        mask_semi = train_mask[i][0]
        mask_semi[mask_semi>0]=1
        Train_Label[i] = mask_semi
        Weight_Map[i] = create_weight_map(Train_Label[i], 2, i)
    
    Train_Img = np.expand_dims(Train_Img, axis = 3)  
    Train_Label = np.expand_dims(Train_Label, axis = 3) 
    Weight_Map = np.expand_dims(Weight_Map, axis = 3) 
   
    return Train_Img, Train_Label, Weight_Map