#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

try:
    # %tensorflow_version only exists in Colab.
    get_ipython().run_line_magic('tensorflow_version', '2.x')
    IS_COLAB = True
except Exception:
    IS_COLAB = False

# TensorFlow ≥2.0 is required
import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0"

if not tf.test.is_gpu_available():
    print("No GPU was detected. CNNs can be very slow without a GPU.")
    if IS_COLAB:
        print("Go to Runtime > Change runtime and select a GPU hardware accelerator.")

import keras 
from keras.layers import Dense, Conv2D, MaxPool2D, UpSampling2D, Dropout, Input, MaxPooling2D,ZeroPadding2D,Conv2DTranspose, concatenate
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm 
import numpy as np
import os
import re


# In[6]:


SIZE = 128


# In[7]:


# to get the files in proper order
def sorted_alphanumeric(data):  
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)',key)]
    return sorted(data,key = alphanum_key)


# In[ ]:


def get(path) :
    train_mask_path = path+'/mask_train'
    train_mask_image = []
    train_mask_file = sorted_alphanumeric(os.listdir(train_mask_path))
    for i in train_mask_file:

        image = cv2.imread(train_mask_path + '/' + i,1)

        # as opencv load image in bgr format converting it to rgb
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, (SIZE, SIZE))

        # normalizing image 
        image = image.astype('float32') / 255.0

        #appending normal normal image    
        train_mask_image.append(img_to_array(image))
        
    train_image_path = path+'/no_mask_train'
    train_image = []
    train_image_file = sorted_alphanumeric(os.listdir(train_image_path))
    try :
        train_image_file.remove('.DS_Store')
    except :
        pass
    
    for i in train_image_file:

        image = cv2.imread(train_image_path + '/' + i,1)

        # as opencv load image in bgr format converting it to rgb
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, (SIZE, SIZE))

        # normalizing image 
        image = image.astype('float32') / 255.0

        # appending normal sketch image
        train_image.append(img_to_array(image))  
        
        
    val_mask_path = path+'/mask_val'
    val_mask_image = []
    val_mask_file = sorted_alphanumeric(os.listdir(val_mask_path))
    for i in val_mask_file:
        image = cv2.imread(val_mask_path + '/' + i,1)

        # as opencv load image in bgr format converting it to rgb
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, (SIZE, SIZE))

        # normalizing image 
        image = image.astype('float32') / 255.0

        #appending normal normal image    
        val_mask_image.append(img_to_array(image))
        
    val_image_path = path+'/no_mask_val'
    val_image = []
    val_image_file = sorted_alphanumeric(os.listdir(val_image_path))
    try :
        val_image_file.remove('.DS_Store')
    except :
        pass
    for i in val_image_file:

        image = cv2.imread(val_image_path + '/' + i,1)

        # as opencv load image in bgr format converting it to rgb
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, (SIZE, SIZE))

        # normalizing image 
        image = image.astype('float32') / 255.0

        # appending normal sketch image
        val_image.append(img_to_array(image))
        
        
    test_mask_path = path+'/mask_test'
    test_mask_image = []
    test_mask_file = sorted_alphanumeric(os.listdir(test_mask_path))
    for i in test_mask_file:

        image = cv2.imread(test_mask_path + '/' + i,1)

        # as opencv load image in bgr format converting it to rgb
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, (SIZE, SIZE))

        # normalizing image 
        image = image.astype('float32') / 255.0

        #appending normal normal image    
        test_mask_image.append(img_to_array(image))
        
    test_image_path = path+'/no_mask_test'
    test_image = []
    test_image_file = sorted_alphanumeric(os.listdir(test_image_path))
    try :
        test_image_file.remove('.DS_Store')
    except :
        pass
    for i in test_image_file:

        image = cv2.imread(test_image_path + '/' + i,1)

        # as opencv load image in bgr format converting it to rgb
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, (SIZE, SIZE))

        # normalizing image 
        image = image.astype('float32') / 255.0

        # appending normal sketch image
        test_image.append(img_to_array(image))
    
    train_mask_image = np.reshape(train_mask_image,(len(train_mask_image),SIZE,SIZE,3))
    train_image = np.reshape(train_image, (len(train_image),SIZE,SIZE,3))
    print('Train no mask image shape:',train_image.shape)

    val_mask_image = np.reshape(val_mask_image,(len(val_mask_image),SIZE,SIZE,3))
    val_image = np.reshape(val_image, (len(val_image),SIZE,SIZE,3))
    print('Validation no mask image shape:',val_image.shape)

    test_mask_image = np.reshape(test_mask_image,(len(test_mask_image),SIZE,SIZE,3))
    test_image = np.reshape(test_image, (len(test_image),SIZE,SIZE,3))
    print('Test no mask image shape',test_image.shape)
    
    return train_mask_image, train_image, val_mask_image, val_image, test_mask_image, test_image

