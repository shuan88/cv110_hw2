import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler


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
    for i in range(convol_size,img.shape[0]-convol_size):
        for j in range(convol_size,img.shape[1]-convol_size):
            new_img[i,j] = np.sum((img[i-convol_size:i+1+convol_size,j-convol_size:j+1+convol_size]*filter)) \
                / (filter_sum)
    return new_img

def Normalized(data , scaler = True):
    scaler = MinMaxScaler()
    scaler.fit(data)
    return scaler.transform(data)
    

def sobel(img , T = False):
    new_img = np.zeros_like(img,dtype=np.uint8)
    # img = np.ones_like(img,dtype=np.uint8)
    filter = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    if T :
        filter = np.transpose(filter)
    print(filter)
    for i in range(1,img.shape[0]-1):
        for j in range(1,img.shape[1]-1):
            result = np.sum((img[i-1:i+2,j-1:j+2]*filter))
            if result > 0:
                new_img[i,j] = result
    # new_img = Normalized(new_img)*255
    return new_img

def magnitude(img_1 , img_2):
    new_img = np.zeros_like(img_1,dtype=np.uint8)
    for i in range(1,new_img.shape[0]-1):
        for j in range(1,new_img.shape[1]-1):
            new_img[i,j] =  np.sqrt(img_1[i,j]**2 + img_2[i,j]**2)
    
    return new_img

img = cv2.imread("img/Bikesgray.jpg",0)
# img = cv2.imread("img/sobel_1.jpeg",0)
print(img.shape[:])

img = GaussianBlur(img, 7)

# new_img_y = sobel(img) # y
# new_img_x = sobel(img , True) # x
# new_img = magnitude(new_img_x , new_img_y) # x


fig = plt.figure()

new_img_y = sobel(img) # y
ax1 = fig.add_subplot(1,3,1)
ax1.imshow(cv2.cvtColor(new_img_y, cv2.COLOR_BGR2RGB))
plt.axis('off')


new_img_x = sobel(img , True) # x
ax2 = fig.add_subplot(1,3,2)
ax2.imshow(cv2.cvtColor(new_img_x, cv2.COLOR_BGR2RGB))
plt.axis('off')


new_img = magnitude(new_img_x , new_img_y) # x
ax3 = fig.add_subplot(1,3,3)
ax3.imshow(cv2.cvtColor(new_img_x, cv2.COLOR_BGR2RGB))
plt.axis('off')




plt.show()
# print("done show image") 
# cv2.imshow('hello',new_img)
# cv2.waitKey(0)

