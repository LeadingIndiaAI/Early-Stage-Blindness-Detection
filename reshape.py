from keras.preprocessing.image import save_img
from keras.preprocessing.image import img_to_array

desired_size=1500
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





    img_array = img_to_array(new_im)
    save_img("/mnt/c/Users/Rohan/Desktop/bennettcopy/new_test/"+ str(i) +".png" , img_array) 



from os import listdir
from os.path import isfile, join
import numpy
import cv2
 
mypath='/mnt/c/Users/Rohan/Desktop/bennettcopy/test_images/'
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
images = numpy.empty(len(onlyfiles), dtype=object)
for n in range(0, len(onlyfiles)):
  images[n] = cv2.imread( join(mypath,onlyfiles[n]) )
  pre_processing(images[n],n+1)
