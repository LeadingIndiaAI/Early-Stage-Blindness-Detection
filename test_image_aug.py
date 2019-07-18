from keras.models import load_model
from keras_efficientnets import __init__,efficientnet,custom_objects,optimize,config

model=load_model("effi_model.hdf5")

def crop_image1(img,tol=8):
    mask = img>tol
    return img[np.ix_(mask.any(1),mask.any(0))]
import os
import numpy as np
import cv2
tst_imgs=os.listdir("C:\\Users\\Ankit Singh Vohra\\Desktop\\test_images")

from keras.preprocessing.image import save_img
from keras.preprocessing.image import img_to_array

desired_size=1500
IMG_SIZE=299
def pre_processing(train_img,i): 
    im = cv2.cvtColor(train_img, cv2.COLOR_BGR2RGB)
    old_size = im.shape[:2] # old_size is in (height, width) format
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    #new_size should be in (width, height) format
    im = cv2.resize(im, (new_size[1], new_size[0]))
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,value=color)
    #image = cv2.cvtColor(new_im, cv2.COLOR_BGR2RGB)
    image = cv2.cvtColor(new_im, cv2.COLOR_RGB2GRAY)
    image = crop_image1(image)
    image = cv2.resize(image, (IMG_SIZE,IMG_SIZE))
    image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , IMG_SIZE/10) ,-4 ,128)  
    
    img_array = img_to_array(image)
    save_img('test_images/'+i, img_array)#images = numpy.empty(len(tst_imgs), dtype=object)

for n in range(len(tst_imgs)):
    tmp = cv2.imread("C:\\Users\\Ankit Singh Vohra\\Desktop\\test_images\\"+tst_imgs[n])
    pre_processing(tmp,tst_imgs[n])
    print(n)

from keras.preprocessing import sequence, image
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
                            samplewise_center=True,
                            horizontal_flip=True,
                            vertical_flip=True,
                            rotation_range=180,
                            shear_range=0.9,
                            brightness_range=(0.5, 1.0),
                            fill_mode='constant',
                            width_shift_range=0.05,
                            height_shift_range=0.05
                            )


test_generator= datagen.flow_from_directory(
       '',  # this is the target directory
       color_mode="rgb",
       target_size=(299, 299),
       batch_size=1,
       class_mode='categorical',
       classes=['test_images'],
       shuffle=False)
predict = model.predict_generator(test_generator,steps = 1)




