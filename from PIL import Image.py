# install pillow module  
# pip install Pillow

import numpy as np            ###import numpy module
from PIL import Image as im          #importing Image module from Pillow library (pillow is a fork of the PIL library)

# Created array using arange function of size 2700(60*45)
arr = np.arange(2700, dtype=np.uint8)
print(type(arr))

arr = np.reshape(arr , (60,45))         #changing the shape of array

print(np.shape(arr))                    #print shape of array

# Creating image object of above array
# Image.fromarray() function to convert the array PIL image object
data = im.fromarray(arr)
data.save('arr_pic.png')

array = np.random.randint(255, size=(400, 400),dtype=np.uint8)
image = im.fromarray(array)             #converting array to image object
image.show()                            #displaying array image

# Creating RGB Images
# RGB images are usually stored as 3 dimensional arrays of 8-bit unsigned integers. 
# The shape of the array is: height x width x 3

color_arr = np.zeros((100, 200, 3), dtype=np.uint8)
color_arr[:,:100 ] = (255, 128, 0)      # Orange left side 
color_arr[:,100:] = (0, 0, 255)         # Blue right side

# Saving an RGB image using PIL
img = im.fromarray(color_arr)        # convert the array PIL image object
img.save('testrgb.png')

for width in range(200):
    for height in range(100):
        color_arr[height, width, 2] = height

trans_img = im.fromarray(color_arr)
trans_img.save('trans_rgb.png')

# Reading Images:

imag = im.open('trans_rgb.png')     #opening imag object
arr = np.array(img)                 #converting image object to array
print(arr.shape)

# Creating RGBA images (Red Green Blue Alpha)
# Alpha value of 255 will make the pixel fully opaque, value 0 will make it fully transparent
# Set transparency

array = np.zeros((50, 100, 4), dtype=np.uint8)
array[:,:50] = [255, 128, 0, 255]   #Orange left side
array[:,50:] = [0, 0, 255, 255]     #Blue right side

# Set transparency depending on x position
for width in range(2):
    for height in range(1):
        array[height, width, 3] = width

rgba_img = im.fromarray(array)
rgba_img.save('rgba.png')


# File Handling
import random

k = random.randint(5,50,(3,4,2))  ## creating array using randint function
np.save('array',k)   ### saving array

m = np.load('array.npy')  ### loading array
print(m)

p = random.rand(4,3)   ### creating array range (0,1)
np.savetxt('array.txt', p ,fmt='%f',delimiter=' , ',encoding= 'latin1')     ### saving array in txt file
with open('array.txt') as f:
  print(f.read())
#np.load('array.txt',allow_pickle=True)      

np.savetxt('array.csv',p,fmt='%f',delimiter=' , ',header='C1,C2,C3',comments='')   ### saving array in csv file
with open('array.csv') as f:
  print(f.read())
#np.load('array.csv',allow_pickle=True)


