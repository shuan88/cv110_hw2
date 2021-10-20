import numpy as np
import cv2
"""
¢ 放大圖片:40%
     最近鄰:20%
     線性:20%
     比較線性內插時從不同方向進行內插的結果:5%
在輸出的圖片右下角加上屬於自己 的簽名(利用圖檔) 10%
"""

# img = cv2.imread("l_hires.jpg")
img = cv2.imread("test_img.png")
print(img.shape[:])

scale = 11.1 # scale of new image
new_x = np.int16(img.shape[0]*scale)
new_y = np.int16(img.shape[1]*scale)
new_img = np.ones((new_x, new_y,img.shape[-1]),dtype=np.uint8)
print(new_img.shape[:])

for i in range(new_img.shape[0]):
    x_near = np.int16(i/scale)
    for j in range(new_img.shape[1]):
        y_near = np.int16(j/scale)
        new_img[i,j,:] = img[x_near,y_near,:]

print("done, show img")
# print(new_img)
cv2.imshow('hello',new_img)
cv2.waitKey(0)