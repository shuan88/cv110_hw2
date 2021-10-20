import numpy as np
import cv2
"""
¢ 模糊圖片:50%
     Average Blur:20%
     Medium Blur:20%
     Gaussian Blur:10% (可用OpenCV函數)
     比較分析三種模糊方法在人臉照片下的差別:5%
"""
# https://docs.opencv.org/4.5.3/d4/d86/group__imgproc__filter.html

## Read image file
img = cv2.imread("l_hires.jpg")
# img = cv2.imread("test_img.png")

filter_size = 7
convol_size = filter_size//2
new_img = np.ones_like(img,dtype=np.uint8)
filter=np.ones((filter_size,filter_size)) # Medium Blur
filter_sum = np.sum(filter)

# name = "Resize_AverageBlur_{}x{}_to_{}x{}".format(img.shape[0],img.shape[1],new_img.shape[0],new_img.shape[1])
# print(name)

filter=np.ones((filter_size,filter_size)) # Average Blur
for color_clannels in range(img.shape[-1]):
    for i in range(convol_size,img.shape[0]-convol_size):
        for j in range(convol_size,img.shape[1]-convol_size):
            new_img[i,j,color_clannels] = np.sum((img[i-convol_size:i+1+convol_size,j-convol_size:j+1+convol_size,color_clannels]*filter)) \
                / (filter_sum)

print("done, show img")
# cv2.imwrite('{}.tiff'.format(name), new_img)
cv2.imshow('hello',new_img)
cv2.waitKey(0)

