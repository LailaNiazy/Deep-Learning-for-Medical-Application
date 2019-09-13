#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
tf.config.gpu.set_per_process_memory_fraction(0.3)
tf.config.gpu.set_per_process_memory_growth(True)


# ## Methods

# In[1]:


def plotter(History):
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

    #Train and test accuracy plot
    plt.figure(figsize=(4,4))
    plt.title("Accuracy Learning Curve")
    plt.plot(History.history["binary_accuracy"], label="binary_accuracy")
    plt.plot(History.history["val_binary_accuracy"], label="val_binary_accuracy")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(); 


# In[2]:


#Data loader

import os
import numpy as np
from random import shuffle
from skimage.io import imread
from skimage.transform import resize

def data_loader(img_w, img_h, label1, label2, data_path):
    #Creating data path
    train_data_path = os.path.join(data_path, 'train')   
    test_data_path = os.path.join(data_path, 'test')

    #Listing all file names in the path
    train_list = os.listdir(train_data_path)
    test_list = os.listdir(test_data_path)
    
    
    return get_train_test_data(train_data_path, test_data_path, train_list, test_list)
    
    

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
        train_img.append([np.array(img), gen_labels(image_name, label1, label2)]) 

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
        test_img.append([np.array(img), gen_labels(image_name, label1, label2)]) 

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

   


# In[1]:


# AlexNet building the model
# AlexNet Model
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Activation, Flatten, Conv2D, Dense, MaxPooling2D, MaxPooling2D,Dropout, BatchNormalization, SpatialDropout2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD, Adam


#Model definition
def modelAlexNet(img_ch, img_width, img_height, Base, batchNormalization, dropout, dropoutRate, spatial_dropout, SDRate):

    model = Sequential()
    
    model.add(Conv2D(filters=Base, input_shape=(img_width, img_height, img_ch),
                     kernel_size=(3,3), strides=(1,1), padding='same'))
    
    #Add batch Normalization
    if batchNormalization:
        model.add(BatchNormalization(axis=-1))
    
    model.add(Activation('relu'))
       
    #Add spatial Dropout
    if spatial_dropout:
        model.add(SpatialDropout2D(SDRate))

    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Conv2D(filters=Base, kernel_size=(3,3), strides=(1,1), padding='same'))
    
    #Add batch Normalization
    if batchNormalization:
        model.add(BatchNormalization(axis=-1))
   
    model.add(Activation('relu'))
    
    #Add spatial Dropout
    if spatial_dropout:
        model.add(SpatialDropout2D(SDRate))
    
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(filters=Base, kernel_size=(3,3), strides=(1,1), padding='same'))
    
    #Add batch Normalization
    if batchNormalization:
        model.add(BatchNormalization(axis=-1))
    
    model.add(Activation('relu'))
    
    #Add spatial Dropout
    if spatial_dropout:
        model.add(SpatialDropout2D(SDRate))
    

    model.add(Conv2D(filters=Base, kernel_size=(3,3), strides=(1,1), padding='same'))
   
    #Add batch Normalization
    if batchNormalization:
        model.add(BatchNormalization(axis=-1))
        
    model.add(Activation('relu'))
    
    #Add spatial Dropout
    if spatial_dropout:
        model.add(SpatialDropout2D(SDRate))
    

    model.add(Conv2D(filters=Base, kernel_size=(3,3), strides=(1,1), padding='same'))
   
    #Add batch Normalization
    if batchNormalization:
        model.add(BatchNormalization(axis=-1))
        
    model.add(Activation('relu'))
    
    #Add spatial Dropout
    if spatial_dropout:
        model.add(SpatialDropout2D(SDRate))
    
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dense(128))
    #Add batch Normalization
    if batchNormalization:
        model.add(BatchNormalization(axis=-1))
        
    model.add(Activation('relu'))
    
    #Add Dropout
    if dropout:
        model.add(Dropout(dropoutRate))

    model.add(Dense(64))
    
    #Add batch Normalization
    if batchNormalization:
        model.add(BatchNormalization(axis=-1))
        
    model.add(Activation('relu'))
    
    #Add Dropout
    if dropout:
        model.add(Dropout(dropoutRate))

    model.add(Dense(1))
    
    #Add batch Normalization
    if batchNormalization:
        model.add(BatchNormalization(axis=-1))
    
    model.add(Activation('sigmoid'))

    model.summary()   
    return model


# In[4]:


# VGG model


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D


#Model definition
def modelVGG(img_ch,img_width,img_height, batchNormalization, spatial_dropout, SDRate, dropout, dropoutRate):

    model = Sequential()
    
    # Conv Block 1
    # 1.1
    model.add(Conv2D(filters=64, input_shape=(img_width, img_height, img_ch),
                     kernel_size=(3,3), strides=(1,1), padding='same'))
     #Add batch Normalization
    if batchNormalization:
        model.add(BatchNormalization(axis=-1))
    
    model.add(Activation('relu'))
    
    #Add spatial Dropout
    if spatial_dropout:
        model.add(SpatialDropout2D(SDRate))
    
    #1.2
    model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same'))
    
    #Add batch Normalization
    if batchNormalization:
        model.add(BatchNormalization(axis=-1))
    
    model.add(Activation('relu'))
    
    #Add spatial Dropout
    if spatial_dropout:
        model.add(SpatialDropout2D(SDRate))
        
    #1.3    
    model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same'))
    
    #Add batch Normalization
    if batchNormalization:
        model.add(BatchNormalization(axis=-1))
    
    model.add(Activation('relu'))
    
    #Add spatial Dropout
    if spatial_dropout:
        model.add(SpatialDropout2D(SDRate))
        
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
     # Conv Block 2
        
    # 2.1
    model.add(Conv2D(filters=128, input_shape=(img_width, img_height, img_ch),
                     kernel_size=(3,3), strides=(1,1), padding='same'))
     #Add batch Normalization
    if batchNormalization:
        model.add(BatchNormalization(axis=-1))
    
    model.add(Activation('relu'))
    
    #Add spatial Dropout
    if spatial_dropout:
        model.add(SpatialDropout2D(SDRate))
    
    #2.2
    model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same'))
    
    #Add batch Normalization
    if batchNormalization:
        model.add(BatchNormalization(axis=-1))
    
    model.add(Activation('relu'))
    
    #Add spatial Dropout
    if spatial_dropout:
        model.add(SpatialDropout2D(SDRate))
        
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    #Conv Block 3
    
     # 3.1
    model.add(Conv2D(filters=256, input_shape=(img_width, img_height, img_ch),
                     kernel_size=(3,3), strides=(1,1), padding='same'))
     #Add batch Normalization
    if batchNormalization:
        model.add(BatchNormalization(axis=-1))
    
    model.add(Activation('relu'))
    
    #Add spatial Dropout
    if spatial_dropout:
        model.add(SpatialDropout2D(SDRate))
    
    #3.2
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same'))
    
    #Add batch Normalization
    if batchNormalization:
        model.add(BatchNormalization(axis=-1))
    
    model.add(Activation('relu'))
    
    #Add spatial Dropout
    if spatial_dropout:
        model.add(SpatialDropout2D(SDRate))
        
    #3.3    
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same'))
    
    #Add batch Normalization
    if batchNormalization:
        model.add(BatchNormalization(axis=-1))
    
    model.add(Activation('relu'))
    
    #Add spatial Dropout
    if spatial_dropout:
        model.add(SpatialDropout2D(SDRate))
        
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    #Conv Block 4
    
    # 4.1
    model.add(Conv2D(filters=512, input_shape=(img_width, img_height, img_ch),
                     kernel_size=(3,3), strides=(1,1), padding='same'))
     #Add batch Normalization
    if batchNormalization:
        model.add(BatchNormalization(axis=-1))
    
    model.add(Activation('relu'))
    
    #Add spatial Dropout
    if spatial_dropout:
        model.add(SpatialDropout2D(SDRate))
    
    #4.2
    model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same'))
    
    #Add batch Normalization
    if batchNormalization:
        model.add(BatchNormalization(axis=-1))
    
    model.add(Activation('relu'))
    
    #Add spatial Dropout
    if spatial_dropout:
        model.add(SpatialDropout2D(SDRate))
        
    #4.3    
    model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same'))
    
    #Add batch Normalization
    if batchNormalization:
        model.add(BatchNormalization(axis=-1))
    
    model.add(Activation('relu'))
    
    #Add spatial Dropout
    if spatial_dropout:
        model.add(SpatialDropout2D(SDRate))
        
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    #Conv Block 5
    
    # 5.1
    model.add(Conv2D(filters=512, input_shape=(img_width, img_height, img_ch),
                     kernel_size=(3,3), strides=(1,1), padding='same'))
     #Add batch Normalization
    if batchNormalization:
        model.add(BatchNormalization(axis=-1))
    
    model.add(Activation('relu'))
    
    #Add spatial Dropout
    if spatial_dropout:
        model.add(SpatialDropout2D(SDRate))
    
    #5.2
    model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same'))
    
    #Add batch Normalization
    if batchNormalization:
        model.add(BatchNormalization(axis=-1))
    
    model.add(Activation('relu'))
    
    #Add spatial Dropout
    if spatial_dropout:
        model.add(SpatialDropout2D(SDRate))
        
    #5.3    
    model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same'))
    
    #Add batch Normalization
    if batchNormalization:
        model.add(BatchNormalization(axis=-1))
    
    model.add(Activation('relu'))
    
    #Add spatial Dropout
    if spatial_dropout:
        model.add(SpatialDropout2D(SDRate))
        
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    #Dense Block
    
    model.add(Flatten())
    
    model.add(Dense(4096))
    model.add(Activation('relu'))
    
    #Add Dropout
    if dropout:
        model.add(Dropout(dropoutRate))
        
    model.add(Dense(4096))
    model.add(Activation('relu'))
    
    #Add Dropout
    if dropout:
        model.add(Dropout(dropoutRate))
     
    #Output
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.summary()
    return model


# In[5]:


# Execute a model
def execute(model, Input, Target ,loss_type, optim, LR, acc_metric,epoch, batch_s, x_test,y_test):
    
    #Compile the model
    model.compile(loss=loss_type,
                  optimizer = optim(lr=LR),
                  metrics=[acc_metric])
    
    History = model.fit(Input, Target, epochs = epoch, batch_size= batch_s, verbose=1,
                     validation_data=(x_test, y_test))
    return History


# In[23]:


# Load data
#init image dimensions
img_w, img_h = 128, 128                               
data_path = '/Lab1/Skin/'          
label1 = 'Mel'    
label2 = 'Nev'

x_train, x_test, y_train, y_test = data_loader(img_w, img_h, label1, label1, data_path)


# ## Tasks

# In[ ]:


## Task 1


# In[7]:


# Task 1a

img_width = 128
img_height = 128
img_ch = 1
Base = 8
batchNormalization = False
dropout = False
dropoutRate = 0
spatial_dropout = False
SDRate = 0
Input = x_train
Test_Input = x_test
Target = y_train
Test_Target = y_test
loss_type = 'binary_crossentropy'
optim = Adam
LR = 0.0001
acc_metric = 'binary_accuracy'
epoch = 50
batch_s = 8

#Model call
model_without_BN = modelAlexNet(img_ch, img_width, img_height, Base, batchNormalization, dropout, dropoutRate, spatial_dropout, SDRate)

#Model execution
History_without_BN = execute(model_without_BN, Input, Target ,loss_type, optim, LR, acc_metric,epoch, batch_s, x_test,y_test)

#Obtain plots
plotter(History_without_BN)


# In[ ]:


# Clear Variables

del model_without_BN, History_without_BN


# In[8]:


# Task 1b:

img_width = 128
img_height = 128
img_ch = 1
Base = 8
batchNormalization = True
dropout = False
dropoutRate = 0
spatial_dropout = False
SDRate = 0
Input = x_train
Test_Input = x_test
Target = y_train
Test_Target = y_test
loss_type = 'binary_crossentropy'
optim = Adam
LR = 0.0001
acc_metric = 'binary_accuracy'
epoch = 50
batch_s = 8

#Model call
model_with_BN = modelAlexNet(img_ch, img_width, img_height, Base, batchNormalization, dropout, dropoutRate, spatial_dropout, SDRate)

#Model execution
History_with_BN = execute(model_with_BN, Input, Target ,loss_type, optim, LR, acc_metric,epoch, batch_s, x_test,y_test)

#Obtain plots
plotter(History_with_BN)


# In[ ]:


# Clear Variables

del model_with_BN, History_with_BN


# In[ ]:


# Task 1b: The final training accuracy when adding batch normalization is: 1.
# The training accuracy from task 1a is reached at epoch number 8.
# The effect of the batch normalization layers is stabilize the learning process and reducing the number of training epochs required. It is very useful for deep architectures.


# In[9]:


# Task 1c

# Model without batch normalization
img_width = 128
img_height = 128
img_ch = 1
Base = 8
batchNormalization = False
dropout = False
dropoutRate = 0
spatial_dropout = False 
SDRate = 0
Input = x_train
Test_Input = x_test
Target = y_train
Test_Target = y_test
loss_type = 'binary_crossentropy'
optim = Adam
LR = 0.00001
acc_metric = 'binary_accuracy'
epoch = 80
batch_s = 8

#Model call
model_1c_NoBN = modelAlexNet(img_ch, img_width, img_height, Base, batchNormalization, dropout, dropoutRate, spatial_dropout, SDRate)

#Model execution
History_1c_NoBN = execute(model_1c_NoBN, Input, Target ,loss_type, optim, LR, acc_metric,epoch, batch_s, x_test,y_test)

#Obtain plots
plotter(History_1c_NoBN)


# In[ ]:


# Clear Variables

del model_1c_NoBN, History_1c_NoBN


# In[13]:


# Model with batch normalization
img_width = 128
img_height = 128
img_ch = 1
Base = 8
batchNormalization = True
dropout = False
dropoutRate = 0
spatial_dropout = False 
SDRate = 0
Input = x_train
Test_Input = x_test
Target = y_train
Test_Target = y_test
loss_type = 'binary_crossentropy'
optim = Adam
LR = 0.00001
acc_metric = 'binary_accuracy'
epoch = 80
batch_s = 8

#Model call
model_1c_BN = modelAlexNet(img_ch, img_width, img_height, Base, batchNormalization, dropout, dropoutRate, spatial_dropout, SDRate)

#Model execution
History_1c_BN = execute(model_1c_BN, Input, Target ,loss_type, optim, LR, acc_metric,epoch, batch_s, x_test,y_test)

#Obtain plots
plotter(History_1c_BN)


# In[ ]:


# Clear Variables

del model_1c_BN, History_1c_BN


# In[10]:


# Task 2 - Add dropout layer
#With BN

img_width = 128
img_height = 128
img_ch = 1
Base = 8
batchNormalization = True
dropout = True
dropoutRate = 0.4
spatial_dropout = False
SDRate = 0
Input = x_train
Test_Input = x_test
Target = y_train
Test_Target = y_test
loss_type = 'binary_crossentropy'
optim = Adam
LR = 0.00001
acc_metric = 'binary_accuracy'
epoch = 80
batch_s = 8

#Model call
model_2BN = modelAlexNet(img_ch, img_width, img_height, Base, batchNormalization, dropout, dropoutRate , spatial_dropout, SDRate)

#Model execution
History_2BN = execute(model_2BN, Input, Target ,loss_type, optim, LR, acc_metric,epoch, batch_s, x_test,y_test)

#Obtain plots
plotter(History_2BN)


# In[ ]:


# Clear Variables

del model_2BN, History_2BN


# In[14]:


#Without BN

img_width = 128
img_height = 128
img_ch = 1
Base = 8
batchNormalization = False
dropout = True
dropoutRate = 0.4
spatial_dropout = False
SDRate = 0.1
Input = x_train
Test_Input = x_test
Target = y_train
Test_Target = y_test
loss_type = 'binary_crossentropy'
optim = Adam
LR = 0.00001
acc_metric = 'binary_accuracy'
epoch = 80
batch_s = 8

#Model call
model_2NotBN = modelAlexNet(img_ch, img_width, img_height, Base, batchNormalization, dropout, dropoutRate, spatial_dropout, SDRate)

#Model execution
History_2NotBN = execute(model_2NotBN, Input, Target ,loss_type, optim, LR, acc_metric,epoch, batch_s, x_test,y_test)

#Obtain plots
plotter(History_2NotBN)


# In[ ]:


# clear variables

del model_2NotBN, History_2NotBN


# In[17]:


# Task 3
# Model with dropout
img_width = 128
img_height = 128
img_ch = 1
Base = 64
batchNormalization = False
dropout = True
dropoutRate = 0.4
spatial_dropout = True
SDRate = 0.1
Input = x_train
Test_Input = x_test
Target = y_train
Test_Target = y_test
loss_type = 'binary_crossentropy'
optim = Adam
LR = 0.00001
acc_metric = 'binary_accuracy'
epoch = 150
batch_s = 8

#Model call
model_3_dropout = modelAlexNet(img_ch, img_width, img_height, Base, batchNormalization, dropout, dropoutRate, spatial_dropout, SDRate)

#Model execution
History_3_dropout = execute(model_3_dropout, Input, Target ,loss_type, optim, LR, acc_metric,epoch, batch_s, x_test,y_test)

#Obtain plots
plotter(History_3_dropout)


# In[ ]:


# clear variables

del model_3_Dropout, History_3_Dropout


# In[18]:


# Model without dropout
img_width = 128
img_height = 128
img_ch = 1
Base = 64
batchNormalization = False
dropout = False
dropoutRate = 0.4
spatial_dropout = False
SDRate = 0.1
Input = x_train
Test_Input = x_test
Target = y_train
Test_Target = y_test
loss_type = 'binary_crossentropy'
optim = Adam
LR = 0.00001
acc_metric = 'binary_accuracy'
epoch = 150
batch_s = 8

#Model call
model_3_NoDropout = modelAlexNet(img_ch, img_width, img_height, Base, batchNormalization, dropout, dropoutRate, spatial_dropout, SDRate)

#Model execution
History_3_NoDropout = execute(model_3_NoDropout, Input, Target ,loss_type, optim, LR, acc_metric,epoch, batch_s, x_test,y_test)

#Obtain plots
plotter(History_3_NoDropout)


# In[ ]:


# clear variables

del model_3_NoDropout, History_3_NoDropout


# In[27]:


# Task 4

# VGG tests for skin images
img_width = 128
img_height = 128
img_ch = 1
batchNormalization = True
dropout = True
dropoutRate = 0.55
spatial_dropout = False
SDRate = 0.001
Input = x_train
Test_Input = x_test
Target = y_train
Test_Target = y_test
loss_type = 'binary_crossentropy'
optim = Adam
LR = 0.00001
acc_metric = 'binary_accuracy'
epoch = 20
batch_s = 8

VGG_skin = modelVGG(img_ch,img_width,img_height, batchNormalization, spatial_dropout, SDRate, dropout, dropoutRate)
History_VGG_skin = execute(VGG_skin, Input, Target ,loss_type, optim, LR, acc_metric,epoch, batch_s, x_test,y_test)
plotter(History_VGG_skin)


# In[15]:


# clear variables

del VGG_skin, History_VGG_skin


# In[16]:


# Load Bone images

# Load data

img_w, img_h = 128, 128                               
data_path = '/Lab1/Bone/'           
label1 = 'AFF'    
label2 = 'NFF'

x_train, x_test, y_train, y_test = data_loader(img_w, img_h, label1, label1, data_path)


# In[21]:


# VGG for bone images

img_width = 128
img_height = 128
img_ch = 1
batchNormalization = True
dropout = True
dropoutRate = 0.4
spatial_dropout = False
SDRate = 0.05
Input = x_train
Test_Input = x_test
Target = y_train
Test_Target = y_test
loss_type = 'binary_crossentropy'
optim = Adam
LR = 0.00001
acc_metric = 'binary_accuracy'
epoch = 20
batch_s = 8

VGG_bone = modelVGG(img_ch,img_width,img_height, batchNormalization, spatial_dropout, SDRate, dropout, dropoutRate)
History_VGG_bone = execute(VGG_bone, Input, Target ,loss_type, optim, LR, acc_metric,epoch, batch_s, x_test,y_test)
plotter(History_VGG_bone)


# In[22]:


# clear variables

del VGG_bone, History_VGG_bone


# ### Data Augmentation

# In[29]:


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
scale_factor = 0.5
image_rescaled = rescale(Img, scale_factor)
show_paired(Img, image_rescaled, "Rescaled")

# Roation
Angle = 25
image_rotated = rotate(Img, Angle)
show_paired(Img, image_rotated, "Rotated")

# Horizontal Flip
horizontal_flip = Img[:, ::-1]
show_paired(Img, horizontal_flip, 'Horizontal Flip')

# Vertical Flip
vertical_flip = Img[::-1, :]
show_paired(Img, vertical_flip, 'vertical Flip')


# Intensity rescaling
Min_Per, Max_Per = 3, 75
min_val, max_val = np.percentile(Img, (Min_Per, Max_Per))

better_contrast = exposure.rescale_intensity(Img, in_range=(min_val, max_val))
show_paired(Img, better_contrast, 'Intensity Rescaling')


# In[32]:


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
MyGen = ImageDataGenerator(featurewise_center = True, rotation_range = 90,
                         width_shift_range = 0.2,
                         horizontal_flip = True)


fix, ax = plt.subplots(1,count+1, figsize=(14,2))
images_flow = MyGen.flow(Img, batch_size=1)
for i, new_images in enumerate(images_flow):
    new_image = array_to_img(new_images[0], scale=True)
    ax[i].imshow(new_image,cmap="gray")
    if i >= count:
        break 


# In[8]:


# Task 6
# Import required libraries …
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img

# Data and model parameters
# image generator parameters
rotation_range = 10
width_shift = 0.1
height_shift_range = 0.1,
rescale = 1./255
horizontal_flip = True

# model parameters
img_ch = 1
img_width = 128
img_height = 128
batchNormalization = True
dropout = True
dropoutRate = 0.4
spatial_dropout = False
SDRate = 0
Base = 64
LR = 0.00001
b_size = 8
TRAIN_DIR = '/Lab1/Lab2/Skin/train/' 
VAL_DIR = '/Lab1/Lab2/Skin/validation/'

def DataAugmentation(rotation_range,width_shift,height_shift_range,rescale,horizontal_flip, TRAIN_DIR, VAL_DIR):
    
    #Train data
    train_datagen = ImageDataGenerator(rotation_range = rotation_range, width_shift_range = width_shift, height_shift_range=height_shift_range,
                                       horizontal_flip = horizontal_flip, rescale = rescale)
    train_generator = train_datagen.flow_from_directory(TRAIN_DIR,target_size=(128, 128))
    
    #Val data
    val_datagen = ImageDataGenerator(rescale = rescale)
    val_generator = val_datagen.flow_from_directory(VAL_DIR,target_size=(128, 128))
    
    
    return train_generator, val_generator 

train_generator, val_generator = DataAugmentation(rotation_range,width_shift,height_shift_range,rescale,horizontal_flip, TRAIN_DIR, VAL_DIR)

model = modelAlexNet(img_ch, img_width, img_height, Base, batchNormalization, dropout, dropoutRate, spatial_dropout, SDRate)

model.compile(loss='binary_crossentropy',optimizer = Adam(lr=LR), metrics=['binary_accuracy'])

History = model.fit_generator( train_generator, steps_per_epoch=2000, epochs=80, validation_data=val_generator, validation_steps=800)
    
plotter(History)


# In[ ]:


# Task 8
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, ZeroPadding2D
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import applications
import numpy as np


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
    

# parameters (TODO)
train_data_dir = '/Lab1/Lab2/Bone/train/'
validation_data_dir = '/Lab1/Lab2/Bone/validation/'
img_width, img_height = ???
epochs = ???
batch_size = ???
LR = ???
# number of data for each class
Len_C1_Train = get_length(train_data_dir,'AFF')
Len_C2_Train = ???
Len_C1_Val = ???
Len_C2_Val = ???

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


# To DO: Feature extraction from pretrained VGG (validation data)
get_ipython().run_line_magic('pinfo2', '')


# training a small MLP with extracted features from the pre-trained model
train_data = features_train
train_labels = np.array([0] * int(Len_C1_Train) + [1] * int(Len_C2_Train))

validation_data = features_validation
validation_labels = np.array([0] * int(Len_C1_Val) + [1] * int(Len_C2_Val))

# TODO: Building the MLP model
get_ipython().run_line_magic('pinfo2', '')

# TODO: Compile and train the model, plot learning curves
get_ipython().run_line_magic('pinfo2', '')


# In[ ]:


# Task 10
from tensorflow.keras import backend as K
from skimage.io import imread
from skimage.transform import resize
import cv2

Sample = '/Lab1/Lab2/Bone/train/AFF/14.jpg'
Img = imread(Sample)
Img = Img[:,:,0]
Img = Img/255
Img = resize(Img, (img_height, img_width), anti_aliasing = True).astype('float32')
Img = np.expand_dims(Img, axis = 2) 
Img = np.expand_dims(Img, axis = 0)
preds = model.predict(Img)
class_idx = np.argmax(preds[0])
print(class_idx)
class_output = model.output[:, class_idx]
last_conv_layer = model.get_layer("Last_ConvLayer")

grads = K.gradients(class_output, last_conv_layer.output)[0]
pooled_grads = K.mean(grads, axis=(0, 1, 2))
iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
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

