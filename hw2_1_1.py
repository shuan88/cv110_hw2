import numpy as np
import cv2
"""
¢ 放大圖片:40%
     最近鄰:20%
     線性:20%
     比較線性內插時從不同方向進行內插的結果:5%
在輸出的圖片右下角加上屬於自己 的簽名(利用圖檔) 10%
"""
def sign(src,threshold=128):
    name_img = cv2.imread("name.jpeg",0)
    name_img = cv2.resize(name_img,(src.shape[0]//2,src.shape[1]//2))
    for i in range(name_img.shape[0]):
        for j in range(name_img.shape[1]):
            if name_img[i,j] < threshold:
                src[name_img.shape[0]+i,name_img.shape[1]+j,:] = [255,255,255]
    return src

# img = cv2.imread("l_hires.jpg")
img = cv2.imread("test_img.png")
print(img.shape[:])

scale = 2.0 # scale of new image
new_x = np.int16(img.shape[0]*scale)
new_y = np.int16(img.shape[1]*scale)
new_img = np.ones((new_x, new_y,img.shape[-1]),dtype=np.uint8)
print(new_img.shape[:])

name = "Resize_NN_{}x{}_to_{}x{}".format(img.shape[0],img.shape[1],new_img.shape[0],new_img.shape[1])

for i in range(new_img.shape[0]):
    x_near = np.int16(i/scale)
    for j in range(new_img.shape[1]):
        y_near = np.int16(j/scale)
        new_img[i,j,:] = img[x_near,y_near,:]

new_img = sign(new_img)
cv2.imwrite('{}.png'.format(name), new_img)
print("done, show img")
# print(new_img)
cv2.imshow('hello',new_img)
cv2.waitKey(0)
