import numpy as np
import cv2
"""
¢ 放大圖片:40%
     最近鄰:20%
     線性:20%
     比較線性內插時從不同方向進行內插的結果:5%
在輸出的圖片右下角加上屬於自己 的簽名(利用圖檔) 10%
"""

def linear_interpolation(A, B ,Y):
    return (Y-A)/abs(B-A)

## Read image file
# img = cv2.imread("l_hires.jpg")
img = cv2.imread("test_img.png")


scale = 2.0 # scale of new image

new_x = np.int16(img.shape[0]*scale)
new_y = np.int16(img.shape[1]*scale)
new_img = np.ones((new_x, new_y,img.shape[-1]),dtype=np.uint8)

name = "Resize_LI_{}x{}_to_{}x{}".format(img.shape[0],img.shape[1],new_img.shape[0],new_img.shape[1])
print(name)
  

for i in range(new_img.shape[0]):
    x_near = i/scale
    # if x_near %1 > 0.0:
    xa_wegiht= linear_interpolation (np.int16(x_near),np.int16(x_near+1),x_near)
    # print(xa_wegiht)
    for j in range(new_img.shape[1]):
        y_near = j/scale
        # if y_near %1 > 0.0:
        ya_wegiht= linear_interpolation (np.int16(y_near),np.int16(y_near+1),y_near)
        # new_img[i,j,:] = img[x_near,y_near,:]
        if (x_near+1) < img.shape[0] and (y_near+1) < img.shape[1]:
            # print(x_near,y_near)
            new_img[i,j,:] = (img[np.int16(x_near),np.int16(y_near),:]*xa_wegiht*ya_wegiht) \
                    + (img[np.int16(x_near+1),np.int16(y_near),:]*(1-xa_wegiht)*ya_wegiht) \
                    + (img[np.int16(x_near),np.int16(y_near+1),:]*(xa_wegiht)*(1-ya_wegiht)) \
                    + (img[np.int16(x_near+1),np.int16(y_near+1),:]*(1-xa_wegiht)*(1-ya_wegiht))
                    ## A B C D
            # print(x_near,y_near)
        elif (x_near) >= img.shape[0] :
            new_img[i,j,:] = (img[np.int16(x_near),np.int16(y_near),:]*ya_wegiht) \
                    + (img[np.int16(x_near),np.int16(y_near+1),:]*(1-ya_wegiht))
        elif (y_near) >= img.shape[1] :
            new_img[i,j,:] = (img[np.int16(x_near),np.int16(y_near),:]*xa_wegiht) \
                    + (img[np.int16(x_near+1),np.int16(y_near),:]*(1-xa_wegiht))
        else:
            new_img[i,j,:] = img[np.int16(x_near),np.int16(y_near),:]
    

print("done, show img")
cv2.imwrite('{}.png'.format(name), new_img)
cv2.imshow('hello',new_img)
cv2.waitKey(0)

