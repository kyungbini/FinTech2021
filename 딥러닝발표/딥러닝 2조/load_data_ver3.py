#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
assert sys.version_info >= (3, 5)

from keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import re


# In[ ]:


def preProcessImage(img):
  rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  # 흰색 
  lower_1 = np.array([200,200,200])
  upper_1 = np.array([255, 255, 255])

  #밝은 하늘색
  lower_2 = np.array([130,190,210])
  upper_2 = np.array([160, 255, 255])
  
  #진한 네이비색
  lower_3 = np.array([60,120,130])
  upper_3 = np.array([100,168,200])

  lower_4 = np.array([0,0,0])
  upper_4 = np.array([0,0,0])

  #제외할 생상영역

  lower_not = np.array([0,0,0])
  upper_not = np.array([0,0,0])


  #각 조건에 따라 선택된 영역들
  red_select_1 = cv2.inRange(rgb,lower_1,upper_1)
  red_select_2 = cv2.inRange(rgb,lower_2,upper_2)
  red_select_3 = cv2.inRange(rgb,lower_3,upper_3)
  red_select_4 = cv2.inRange(rgb,lower_4,upper_4)
  red_select_not = cv2.inRange(rgb, lower_not, upper_not)
  red_exclude = cv2.bitwise_not(red_select_not)

  #조건들 순차적으로 합쳐주기
  red_select_aug_1 = cv2.bitwise_or(red_select_1, red_select_2)
  red_select_aug_2 = cv2.bitwise_or(red_select_aug_1, red_select_3) 
  red_select_aug_3 = cv2.bitwise_or(red_select_aug_2, red_select_4) 
  red_select_aug_4 = cv2.bitwise_and(red_select_aug_3, red_exclude)

  #red_select_aug_4 = cv2.bitwise_not(red_select_aug_4)
  kernel = np.ones((7,7))
  red_select_aug_4 = cv2.dilate(red_select_aug_4, kernel, iterations = 1)
  contours, hierarchy = cv2.findContours(red_select_aug_4, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

  
  #윤곽선 중 면적 2000 이상인 윤곽선 내부만 살려서 원본 이미지에 검게 칠하기
  for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 2000:
        #cv2.drawContours(original, cnt, -1, (255, 0 ,255), 3)
        cv2.drawContours(img ,[cnt], 0, (0,0,0), -1)
    


# In[6]:


SIZE = 128


# In[4]:


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
        preProcessImage(image)

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
        preProcessImage(image)

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
        preProcessImage(image)

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

