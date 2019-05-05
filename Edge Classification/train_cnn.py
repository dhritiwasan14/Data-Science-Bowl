from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Conv2D
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
PATCH_SIZE = 5
CHANNELS = 3

train_datagen = ImageDataGenerator()

model = Sequential()

#add model layers
model.add(Conv2D(64, kernel_size=PATCH_SIZE, activation='relu', input_shape=(None, None, CHANNELS)))
model.add(Conv2D(1, kernel_size=1, activation='relu', input_shape=(None, None, 64)))
model.add(Activation('sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

import os
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import scipy.misc
import random
import math

df = pd.read_csv('0.csv')

data = []
labels = []
FINAL_SIZE = 256

for idx, row in df.iterrows():
    o = row['image_id']
    if o[0] == '.':
        continue
        
    d = '/home/ashiswin/Documents/Data Science Bowl 2018/stage1_train/' + o
    
    img = cv2.imread(d + '/images/' + o + '.png')
    
    if img.shape[0] > FINAL_SIZE or img.shape[1] > FINAL_SIZE:
        continue
    
    padding_1 = (math.floor((FINAL_SIZE - img.shape[0]) / 2) + (PATCH_SIZE // 2), math.ceil((FINAL_SIZE - img.shape[0]) / 2) + (PATCH_SIZE // 2))
    padding_2 = (math.floor((FINAL_SIZE - img.shape[1]) / 2) + (PATCH_SIZE // 2), math.ceil((FINAL_SIZE - img.shape[1]) / 2) + (PATCH_SIZE // 2))
    
    img_pad = np.pad(img, (padding_1, padding_2, (0, 0)), 'constant', constant_values=(0, 0))
    
    mask = cv2.imread(d + '/thin_mask.png', 0)
    mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)[1]
    
    padding_1 = (math.floor((FINAL_SIZE - img.shape[0]) / 2), math.ceil((FINAL_SIZE - img.shape[0]) / 2))
    padding_2 = (math.floor((FINAL_SIZE - img.shape[1]) / 2), math.ceil((FINAL_SIZE - img.shape[1]) / 2))
    
    mask_pad = np.pad(mask, (padding_1, padding_2), 'constant', constant_values=(0, 0))
    data.append(img_pad)
    labels.append(mask_pad.reshape(mask_pad.shape[0], mask_pad.shape[1], 1))
    
data = np.array(data)
labels = np.array(labels)

print(data.shape)
print(labels.shape)

model.fit(data, labels, epochs=10, batch_size=4)

#model.fit_generator(train_datagen.flow(x = data, y = labels), epochs=10)
