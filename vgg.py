import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
from tensorflow.keras import layers
import keras as K
from keras.layers import Flatten, Dense, Input
from keras.models import Model
from keras.preprocessing import sequence, image
from keras import layers
from keras.optimizers import Adam
from keras.models import load_model


from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
                             
                             samplewise_center=True,
                             rescale=1./255,
                             validation_split=0.2          
                             )


train_generator = datagen.flow_from_directory(
        '../final_train/train',  # this is the target directory
        color_mode="rgb",
        target_size=(224, 224),  # all images will be resized to 150x150
        batch_size=16,
        shuffle=True,
        class_mode='categorical',
        seed=45,
        classes=['0','1','2','3','4'],
	    subset="training"       
       )



validation_generator = datagen.flow_from_directory(
        '../final_train/validation',  # this is the target directory
        color_mode="rgb",
        target_size=(224, 224),  # all images will be resized to 150x150
        batch_size=16,
        shuffle=True,
        class_mode='categorical',
        seed=45,
        classes=['0','1','2','3','4'],
        subset="validation"       
       )


import tensorflow as tf
import os
import keras
from keras.callbacks import Callback
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D,BatchNormalization
from keras import backend as K
img_width, img_height = 224, 224


if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

'''

model = keras.applications.vgg19.VGG19(include_top=False, weights='vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5',input_shape=input_shape,pooling='avg')




for layer in model.layers[0:-1]:
    layer.trainable = False

x = model.output
model.summary()


x = Dense(16)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)          
x = Dropout(0.25)(x)


x = Dense(8)(x)
x = Activation('relu')(x)          
predictions = Dense(5, activation='softmax')(x)

model= Model(input = model.input, output = predictions)

filepath="vgg_model.hdf5" # checkpoint
checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_categorical_accuracy', verbose=1, save_best_only=True, mode='max')


model.compile(
          
          optimizer=Adam(lr=0.0001),
          loss='categorical_crossentropy',
          metrics=['categorical_accuracy'])

model.summary()

'''

model=load_model('vgg_model.hdf5')

test_generator= datagen.flow_from_directory(
        '../final_train/test',
        color_mode="rgb",
        target_size=(224, 224),
        class_mode='categorical',
        batch_size=1,
        #shuffle=True,
        classes=['0','1','2','3','4']
          )

predicted_scores = model.predict_generator(test_generator, test_generator.samples//1)
Evaluate_score = model.evaluate_generator(test_generator , test_generator.samples //1,use_multiprocessing=True)

print(predicted_scores)
print(Evaluate_score)
