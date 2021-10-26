import numpy as np
import cv2
import math

"""
¢ 模糊圖片:50%
     Average Blur:20%
     Medium Blur:20%
     Gaussian Blur:10% (可用OpenCV函數)
     比較分析三種模糊方法在人臉照片下的差別:5%
"""
# https://docs.opencv.org/4.5.3/d4/d86/group__imgproc__filter.html
# https://zh.wikipedia.org/wiki/%E9%AB%98%E6%96%AF%E6%A8%A1%E7%B3%8A
# https://numpy.org/doc/stable/reference/random/generated/numpy.random.normal.html

def gaussian (x,mean=0,std=1):
    return ((math.sqrt(2*math.pi)*std)**-1)* math.exp(-0.5*((((x-mean)/std)**2)))

## Read image file
# img = cv2.imread("l_hires.jpg")
img = cv2.imread("test_img.png")

filter_size = 5
convol_size = filter_size//2
new_img = np.ones_like(img,dtype=np.uint8)
filter=np.ones((filter_size,filter_size)) # Medium Blur

std = 0.8
mean = 0
sum = 0

print(filter_size,convol_size)
name = "Resize_GaussianBlur_{}x{}_to_{}x{}".format(img.shape[0],img.shape[1],new_img.shape[0],new_img.shape[1])
# print(name)

for i in range(filter.shape[0]):
    for j in range(filter.shape[1]):
        filter[i,j]=gaussian(np.sqrt((i-convol_size)**2 + (j-convol_size)**2) ,mean,std)
filter = filter/np.sum(filter)
filter = np.int32((filter/filter[0,0]))
filter_sum = np.sum(filter)
print(filter)
print(filter_sum)


for color_clannels in range(img.shape[-1]):
    for i in range(convol_size,img.shape[0]-convol_size):
        for j in range(convol_size,img.shape[1]-convol_size):
            new_img[i,j,color_clannels] = np.sum((img[i-convol_size:i+1+convol_size,j-convol_size:j+1+convol_size,color_clannels]*filter)) \
                / (filter_sum)
            # new_img[i,j,color_clannels] = \
            # np.sum((img[i-convol_size:i+1+convol_size,j-convol_size:j+1+convol_size,color_clannels]*filter))*(filter_size*filter_size)

# print(new_img)

print("done, show img")
cv2.imwrite('{}.png'.format(name), new_img)
# cv2.imshow('hello',new_img)
# cv2.waitKey(0)


