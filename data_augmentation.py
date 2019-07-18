from keras.preprocessing import sequence, image
import random


from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
                             
                             samplewise_center=True,
                             
                             horizontal_flip=True,
                             vertical_flip=True,
                             rotation_range=180,
                             shear_range=0.9,
                             brightness_range=(0.5, 1.0),
                             #zoom_range=[0.85,0.9],
                             fill_mode='constant',
                             width_shift_range=0.05,
                             height_shift_range=0.05
                             )

i=0
for batch in datagen.flow_from_directory(
        '/mnt/c/Users/Rohan/Desktop/bennettcopy/dataset/final_train',  # this is the target directory
        color_mode="rgb",
        target_size=(299, 299),
        batch_size=1,
        class_mode='binary',
        classes=['4'],
        save_to_dir='/mnt/c/Users/Rohan/Desktop/bennettcopy/dataset/new_augmented/4/' ,
        #save_prefix='aug',
        save_format='png',
        shuffle=False,
        #seed=random.randint(1,99)
        ):
      
    i += 1
    
    
    if i > 1505:
      break
        



