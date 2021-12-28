import numpy as np
import cv2
from matplotlib import pyplot as plt

"""
¢ 放大圖片:40%
     最近鄰:20%
     線性:20%
     比較線性內插時從不同方向進行內插的結果:5%
在輸出的圖片右下角加上屬於自己 的簽名(利用圖檔) 10%
"""

# def get_threshold(input,threshold=0.5):
#     return np.average(input)

# img = cv2.imread("l_hires.jpg")
img = cv2.imread("test_img.png")

def sign(src,threshold=128):
    name_img = cv2.imread("name.jpeg",0)
    name_img = cv2.resize(name_img,(src.shape[0]//2,src.shape[1]//2))
    for i in range(name_img.shape[0]):
        for j in range(name_img.shape[1]):
            if name_img[i,j] < threshold:
                src[name_img.shape[0]+i,name_img.shape[1]+j,:] = [255,255,255]
    return src

new_img = sign(img)

cv2.imshow('hello',new_img)
cv2.waitKey(0)


