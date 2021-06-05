from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os
import glob

from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Flatten, Dense
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard

base_path = "C:/Users/user/anaconda3/dataset-resized4/dataset-resized2"
img_list = glob.glob(os.path.join(base_path, '*/*.jpg'))
tensorboard = TensorBoard(log_dir='my_log')

train_datagen = ImageDataGenerator(                      
    rescale=1./225,                   
    shear_range=10,               
    zoom_range=0.1,               
    width_shift_range=0.1,        
    height_shift_range=0.1,       
    horizontal_flip=True,         
    vertical_flip=True,           
    validation_split=0.1)        

test_datagen = ImageDataGenerator(
    rescale=1./255, 
    validation_split=0.1)

                    
train_generator = train_datagen.flow_from_directory(
    base_path,                    
    target_size=(300, 300),       
    batch_size=512,                
    class_mode='categorical',     
    subset='training',            
    seed=0)          

validation_generator = test_datagen.flow_from_directory(
    base_path,                   
    target_size=(300, 300),       
    batch_size=512,                
    class_mode='categorical',     
    subset='validation',          
    seed=0)

labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())

print(labels)

model = load_model('C:/Users/user/anaconda3/files/object_detection_model5_8.h5')


model.summary()
model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=0.001),
                      metrics=['accuracy'])

model.fit_generator(train_generator,
                    epochs=50,
                    steps_per_epoch=13312//256,
                    validation_data=validation_generator,
                    validation_steps=1477//256,
                    callbacks=[tensorboard])
                    
model.save('C:/Users/user/anaconda3/files/object_detection_model5_9.h5')
