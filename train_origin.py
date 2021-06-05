import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os
import glob

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Flatten, Dense
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard


base_path = "C:/Users/user/anaconda3/dataset-resized4/dataset-resized2" #資料集路徑
img_list = glob.glob(os.path.join(base_path, '*/*.jpg'))
#tensorboard = TensorBoard(log_dir='my_log')#將訓練的模型輸出至my_log資料夾

#ImageDataGenerator():資料擴增的功能，利用現有的資料經過旋轉、翻轉、縮放等方式增加更多的訓練資料
#                     小幅修改現有樣本來產生更多樣本
train_datagen = ImageDataGenerator(                      
    rescale=1./225,                   
    shear_range=10,               #隨機順時針傾斜圖片0~10度
    zoom_range=0.1,               #隨機水平或垂直縮放影像10%
    width_shift_range=0.1,        #隨機向左或向右平移10%寬度以內的像素
    height_shift_range=0.1,       #隨機向上或向下平移10%寬度以內的像素
    horizontal_flip=True,         #隨機水平翻轉圖片
    vertical_flip=True,           #隨機垂直翻轉圖片
    validation_split=0.1)         #取訓練集最後面10%做驗證

test_datagen = ImageDataGenerator(
    rescale=1./255, 
    validation_split=0.1)

#建立生成訓練資料的走訪器
#flow_from_directory(路徑, 圖像被縮放成(300,300), 一次訊量需要的樣本數):以資料夾路徑視為參數,
#                    返回標籤數組的形式, , 可選參數_打亂數據進行和進行變換時的隨機變數種子)
train_generator = train_datagen.flow_from_directory(
    base_path,                    #資料集路徑
    target_size=(300, 300),       #調整影像大小成 300 x300
    batch_size=32,                #每批次要生成32筆資料
    class_mode='categorical',     #指定多元分類方式:(bottle, metal, paper, glass)
    subset='training',            #只生成前75%的訓練資料
    seed=0)          

validation_generator = test_datagen.flow_from_directory(
    base_path,                    #資料及路徑
    target_size=(300, 300),       #調整影像大小成 300 x300
    batch_size=32,               #每批次要生成32筆資料
    class_mode='categorical',     #指定多元分類方式:(bottle, metal, paper, glass)
    subset='validation',          #只生成後25%的驗證資料
    seed=0)

labels = (train_generator.class_indices)       #可以獲取base_path裡面對應的資料夾順序
labels = dict((v,k) for k,v in labels.items()) #取出資料項

print(labels)#印出標籤

#model_1
#範例模型
model = Sequential([                   #建立序列是模型物件
    #第一個卷基層
    Conv2D(filters=32,                 #32個神經元，輸出32張特徵圖
           kernel_size=3,              #卷積核尺寸3x3
           padding='same',             #使用填補法
           activation='relu',          #啟動函數
           input_shape=(300, 300, 3)), #輸入圖片尺寸為300x300(3channals)
    #第一個最大池化層
    MaxPooling2D(pool_size=2),         #檢視視窗2x2
    
    #第二個卷基層
    Conv2D(filters=64,                 #64個神經元，輸出32張特徵圖
           kernel_size=3,              #卷積核尺寸3x3
           padding='same',             #使用填補法
           activation='relu'),         #啟動函數
    
    #第二個最大池化層
    MaxPooling2D(pool_size=2),         #檢視視窗2x2

    #第三個卷基層
    Conv2D(filters=32,                 #32個神經元，輸出32張特徵圖
           kernel_size=3,              #卷積核尺寸3x3
           padding='same',             #使用填補法
           activation='relu'),         #啟動函數
    
    #第三個最大池化層
    MaxPooling2D(pool_size=2),         #檢視視窗2x2
 
    #第四個卷基層
    Conv2D(filters=32,                 #32個神經元，輸出32張特徵圖
           kernel_size=3,              #卷積核尺寸3x3
           padding='same',             #使用填補法
           activation='relu'),         #啟動函數
    
    #第四個最大池化層
    MaxPooling2D(pool_size=2),         #檢視視窗2x2
    #展平層
    Flatten(),                         #將特徵圖拉平
    
    #密集層
    Dense(64, activation='relu'),      #64個神經元, 啟動函數
    #密集層
    Dense(4, activation='softmax')     #4個神經元, 啟動函數
])
#模型總結(摘要)
model.summary()

#模型編譯 
model.compile(loss='categorical_crossentropy',             #損失函數
              optimizer='adam',                            #優化器
              metrics=['acc'])                             #用精準度評估模型的指標

#模型訓練
model.fit_generator(train_generator,                       #訓練生成器
                    epochs=50,                             #週期
                    steps_per_epoch=13093/128,            #圖片量除以batch_size(批次大小)
                                                           #指定每周期需要訓練多少次
                                                           
                    validation_data=validation_generator,  #驗證資料生成器
                    validation_steps=1453//128,            #指定每個周期要驗證多少批次
                    #callbacks=[tensorboard]               #tensorboard應用(監測)
                    )               
#模型儲存
model.save('C:/Users/user/anaconda3/files/object_detection_model6.h5')

