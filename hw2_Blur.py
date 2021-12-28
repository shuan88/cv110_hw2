import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

"""
¢ 模糊圖片:50%
     Average Blur:20%
     Medium Blur:20%
     Gaussian Blur:10% (可用OpenCV函數)
     比較分析三種模糊方法在人臉照片下的差別:5%
"""
# https://docs.opencv.org/4.5.3/d4/d86/group__imgproc__filter.html

def AverageBlur(img,filter_size):
    convol_size = filter_size//2
    new_img = np.ones_like(img,dtype=np.uint8)
    filter=np.ones((filter_size,filter_size)) # Medium Blur
    filter_sum = np.sum(filter)
    for color_clannels in range(img.shape[-1]):
        for i in range(convol_size,img.shape[0]-convol_size):
            for j in range(convol_size,img.shape[1]-convol_size):
                new_img[i,j,color_clannels] = np.sum((img[i-convol_size:i+1+convol_size,j-convol_size:j+1+convol_size,color_clannels]*filter)) \
                    / (filter_sum)
    return new_img

def MediumBlur(img,filter_size):
    convol_size = filter_size//2
    new_img = np.ones_like(img,dtype=np.uint8)
    filter=np.ones((filter_size,filter_size)) # Medium Blur
    for color_clannels in range(img.shape[-1]):
        for i in range(convol_size,img.shape[0]-convol_size):
            for j in range(convol_size,img.shape[1]-convol_size):
                new_img[i,j,color_clannels] = \
                    np.median((img[i-convol_size:i+1+convol_size,\
                        j-convol_size:j+1+convol_size,color_clannels]*filter))
    return new_img

def gaussian (x,mean=0,std=1):
    return ((math.sqrt(2*math.pi)*std)**-1)* math.exp(-0.5*((((x-mean)/std)**2)))

def GaussianBlur(img,filter_size,mean=0,std=1):
    filter=np.ones((filter_size,filter_size))
    convol_size = filter_size//2
    new_img = np.ones_like(img,dtype=np.uint8)
    for i in range(filter.shape[0]):
        for j in range(filter.shape[1]):
            filter[i,j]=gaussian(np.sqrt((i-convol_size)**2 + (j-convol_size)**2) ,mean,std)
    filter = filter/np.sum(filter)
    filter = np.int32((filter/filter[0,0]))
    filter_sum = np.sum(filter)
    for color_clannels in range(img.shape[-1]):
        for i in range(convol_size,img.shape[0]-convol_size):
            for j in range(convol_size,img.shape[1]-convol_size):
                new_img[i,j,color_clannels] = np.sum((img[i-convol_size:i+1+convol_size,j-convol_size:j+1+convol_size,color_clannels]*filter)) \
                    / (filter_sum)
    return new_img

## Read image file
img = cv2.imread("./img/myface.JPG")
# img = cv2.imread("./img/3.jpeg")
# img = cv2.imread("./img/lena.png")

filter_size = 5

fig = plt.figure()

new_img = AverageBlur(img, filter_size)
name = "./myface/AverageBlur_filter{}".format(filter_size)
print(name)
print("done, show img")
# cv2.imwrite('{}.png'.format(name), new_img)
ax1 = fig.add_subplot(1,3,1)
ax1.imshow(cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.xlabel("AverageBlur")


new_img = MediumBlur(img, filter_size)
name = "./myface/MediumBlur_filter{}".format(filter_size)
print(name)
print("done, show img")
# cv2.imwrite('{}.png'.format(name), new_img)
ax2 = fig.add_subplot(1,3,2)
ax2.imshow(cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.xlabel("MediumBlur")

new_img = GaussianBlur(img, filter_size)
name = "./myface/GaussianBlur_filter{}".format(filter_size)
print(name)
print("done, show img")
# cv2.imwrite('{}.png'.format(name), new_img)
ax3 = fig.add_subplot(1,3,3)
ax3.imshow(cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.xlabel("GaussianBlur")

# plt.savefig("figure3.png")
plt.show()

# cv2.imshow('hello',new_img)
# cv2.waitKey(0)

