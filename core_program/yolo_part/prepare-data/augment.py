from numpy import expand_dims
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


dir = "/home/kittipong/dataset/Color/"
i = 15
img_data = os.listdir(dir)
for k in range(len(img_data)):
    img = cv2.imread('/home/kittipong/dataset/Color/color_image'+str(k)+'.png')
    file1 = open("/home/kittipong/dataset/label/color_image"+str(k)+".txt","r")
    label =  file1.read()
    data = img_to_array(img)
# expand dimension to one sample
    samples = expand_dims(data, 0)
# create image data augmentation generator
    datagen = ImageDataGenerator(brightness_range=[0.2,1.0])
# prepare iterator
    it = datagen.flow(samples, batch_size=1)
    for j in range(9):
        batch = it.next()
        image = batch[0].astype('uint8')
        cv2.imwrite('/home/kittipong/dataset/augment/color_image'+str(i)+'.png', image)
        f= open("/home/kittipong/dataset/label/color_image"+str(i)+'.txt',"w")
        f.write(label)
        f.close()
        i=i+1
    file1.close()