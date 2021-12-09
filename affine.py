import cv2
import numpy as np
import argparse

file_name = "img/lena.png"

# 讀取圖片 laod image
img_original = cv2.imread(file_name)

image = cv2.resize(img_original, (256, 256), interpolation=cv2.INTER_AREA)
cv2.imshow('Result', image)


# image = back
# image = back[: image.shape[0] ,: image.shape[1],: ] +image


# We use warpAffine to transform
# the image using the matrix, T

T = np.float32([[1, 0, 0], [0, 1, 60]])
height, width = image.shape[:2]
img_translation = cv2.warpAffine(image, T, (width*2, height*2))
cv2.imshow('Translation', img_translation)

# 按下任意鍵則關閉所有視窗
cv2.waitKey(0)
cv2.destroyAllWindows()