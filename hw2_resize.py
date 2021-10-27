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
    # name_img = cv2.resize(name_img,(src.shape[0]//2,src.shape[1]//2))
    name_img = cv2.resize(name_img,(src.shape[1]//2,src.shape[0]//2))
    print(name_img.shape[:])
    for i in range(name_img.shape[0]):
        for j in range(name_img.shape[1]):
            if name_img[i,j] < threshold:
                src[name_img.shape[0]+i,name_img.shape[1]+j,:] = [255,255,255]
                # src[name_img.shape[1]+i,name_img.shape[0]+j,:] = [255,255,255]
    return src

# def nearest_neighbor(input,scale):
#     new_x = np.int16(input.shape[0]*scale)
#     new_y = np.int16(input.shape[1]*scale)
#     new_img = np.ones((new_x, new_y,input.shape[-1]),dtype=np.uint8)
#     for i in range(new_img.shape[0]):
#         x_near = np.int16(i/scale)
#         for j in range(new_img.shape[1]):
#             y_near = np.int16(j/scale)
#             new_img[i,j,:] = input[x_near,y_near,:]
#     return new_img

def nearest_neighbor(input,scale_x,scale_y):
    new_x = np.int16(input.shape[0]*scale_x)
    new_y = np.int16(input.shape[1]*scale_y)
    new_img = np.ones((new_x, new_y,input.shape[-1]),dtype=np.uint8)
    for i in range(new_img.shape[0]):
        x_near = np.int16(i/scale_x)
        for j in range(new_img.shape[1]):
            y_near = np.int16(j/scale_y)
            new_img[i,j,:] = input[x_near,y_near,:]
    return new_img

def linear_interpolation(A, B ,Y):
    return (Y-A)/abs(B-A)

def linear (input,scale_x,scale_y):
    new_x = np.int16(input.shape[0]*scale_x)
    new_y = np.int16(input.shape[1]*scale_y)
    new_img = np.ones((new_x, new_y,input.shape[-1]),dtype=np.uint8)
    for i in range(new_img.shape[0]):
        x_near = i/scale_x
        xa_wegiht= linear_interpolation (np.int16(x_near),np.int16(x_near+1),x_near)
        for j in range(new_img.shape[1]):
            y_near = j/scale_y
            # if y_near %1 > 0.0:
            ya_wegiht= linear_interpolation (np.int16(y_near),np.int16(y_near+1),y_near)
            # new_input[i,j,:] = input[x_near,y_near,:]
            if (x_near+1) < input.shape[0] and (y_near+1) < input.shape[1]:
                # print(x_near,y_near)
                new_img[i,j,:] = (input[np.int16(x_near),np.int16(y_near),:]*xa_wegiht*ya_wegiht) \
                        + (input[np.int16(x_near+1),np.int16(y_near),:]*(1-xa_wegiht)*ya_wegiht) \
                        + (input[np.int16(x_near),np.int16(y_near+1),:]*(xa_wegiht)*(1-ya_wegiht)) \
                        + (input[np.int16(x_near+1),np.int16(y_near+1),:]*(1-xa_wegiht)*(1-ya_wegiht))
                # new_img[i,j,:] = (input[np.int16(x_near),np.int16(y_near),:]*xa_wegiht*ya_wegiht) \
                #         + (input[np.int16(x_near+1),np.int16(y_near),:]*(1-xa_wegiht)*ya_wegiht) \
                #         + (input[np.int16(x_near),np.int16(y_near+1),:]*(xa_wegiht)*(1-ya_wegiht)) \
                #         + (input[np.int16(x_near+1),np.int16(y_near+1),:]*(1-xa_wegiht)*(1-ya_wegiht))
            """
                # elif (x_near) >= input.shape[0] :
                #     new_img[i,j,:] = (input[np.int16(x_near),np.int16(y_near),:]*ya_wegiht) \
                #             + (input[np.int16(x_near),np.int16(y_near+1),:]*(1-ya_wegiht))
                # elif (y_near) >= input.shape[1] :
                #     new_img[i,j,:] = (input[np.int16(x_near),np.int16(y_near),:]*xa_wegiht) \
                #             + (input[np.int16(x_near+1),np.int16(y_near),:]*(1-xa_wegiht))
                # else:
                #     new_img[i,j,:] = input[np.int16(x_near),np.int16(y_near),:]
            """
    return new_img


img = cv2.imread("./img/lena.png")
print(img.shape[:])

scale = 2.0 # scale of new image

# new_img = sign(nearest_neighbor(img,scale,scale+1.5))
# name = "./output/Resize_NN"

# new_img = sign(nearest_neighbor(img,scale))
# name = "./output/Resize_NN"

new_img = sign(linear (img,scale,scale+1.5))
# name = "./output/Resize_Linear"
name = "./output/Resize_Linear_scale{}".format(scale)

cv2.imwrite('{}.png'.format(name), new_img)

print("done, show img")
# print(new_img)
cv2.imshow('hello',new_img)
cv2.waitKey(0)
