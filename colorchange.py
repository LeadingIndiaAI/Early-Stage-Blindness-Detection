
def crop_image1(img,tol=8):
    # img is image data
    # tol  is tolerance
        
    mask = img>tol
    return img[numpy.ix_(mask.any(1),mask.any(0))]

from keras.preprocessing.image import save_img
from keras.preprocessing.image import img_to_array

IMG_SIZE=299
def pre_processing(train_img,i): 
  image = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)
  image = crop_image1(image)
  image = cv2.resize(image, (IMG_SIZE,IMG_SIZE))
  
  image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , IMG_SIZE/10) ,-4 ,128)
  
  img_array = img_to_array(image)
  save_img("/mnt/c/Users/Rohan/Desktop/bennettcopy/dataset/new_grayscaled/4/"+ str(i) +".png" , img_array) 




from os import listdir
from os.path import isfile, join
import numpy
import cv2
 
mypath='/mnt/c/Users/Rohan/Desktop/bennettcopy/dataset/final_train/4/'
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
images = numpy.empty(len(onlyfiles), dtype=object)
for n in range(0, len(onlyfiles)):
  images[n] = cv2.imread( join(mypath,onlyfiles[n]) )
  pre_processing(images[n],n+1)
