#pip install keras_efficientnets
#pip install git+https://github.com/titu1994/keras-efficientnets.git
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras import layers
import keras as K
from keras.layers import Flatten, Dense, Input
from keras.models import Model
from keras.preprocessing import sequence, image
from keras import layers
from keras.models import load_model


from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
                             samplewise_center=True,
                             rescale=1./255,
                             validation_split=0.2,
                                      
                             )


train_generator = datagen.flow_from_directory(
        'final_dataset/',  # this is the target directory
        color_mode="rgb",
        target_size=(224, 224),  # all images will be resized to 150x150
        batch_size=32,
        shuffle=True,
        class_mode='categorical',
        seed=45,
        classes=['0','1','2','3','4'],
        
        subset="training"
	     
       )



validation_generator = datagen.flow_from_directory(
        'final_dataset/',  # this is the target directory
        color_mode="rgb",
        target_size=(224, 224),  # all images will be resized to 150x150
        batch_size=32,
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
from PIL import Image
img_width, img_height = 224, 224

"""
if K.image_data_format() == 'channels_first':
    input_shape = (1, img_width, img_height)
else:
    input_shape = (img_width, img_height, 1)
    
"""
from keras_efficientnets import EfficientNetB0

model = EfficientNetB0(include_top=False, weights='imagenet',pooling='avg',input_shape=(224,224,3))

for layer in model.layers[0:-1]:
    layer.trainable = True

x = model.output

x = Dense(64)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)          
x = Dropout(0.3)(x)

x = Dense(32)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)          
x = Dropout(0.3)(x)

x = Dense(16)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)          
x = Dropout(0.3)(x)


x = Dense(8)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)          
x = Dropout(0.3)(x)

predictions = Dense(5, activation='softmax')(x)

model= Model(input = model.input, output = predictions)
filepath="effi_model.hdf5" # checkpoint
checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_categorical_accuracy', verbose=1, save_best_only=True, mode='max')

adam = keras.optimizers.Adam(lr=0.001)
model.summary()
model.compile(
          
          optimizer=adam,
          loss='categorical_crossentropy',
          metrics=['categorical_accuracy'])

model.summary()


final_model=model.fit_generator(train_generator,
                    steps_per_epoch=train_generator.samples //32,
                    validation_data = validation_generator,
                    validation_steps=validation_generator.samples // 32,
                    epochs=50,
                    verbose=1,
                    shuffle=True,
                    callbacks=[checkpoint])

model.save('end_effi_model.hdf5')