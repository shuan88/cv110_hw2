# cv110_hw2_D0748284

原始圖片
![](test_img.png)
# 1. 放大圖片：40%
## 1-1 最近鄰：20%
* [code](hw2_1_1.py)
* 更改倍率:`scale = 2.0`
### Result
![](./output/Resize_NN.png)

## 1-2 線性：20%
* [code](hw2_1_2.py)
* 更改倍率:`scale = 11.1 # scale of new image`
### Result
![](output/Resize_Linear.png)


# 2.  模糊圖片：50%
## Average Blur
* [code](hw2_2_1_Average_Blur.py)
### Result
![](./output/AverageBlur_filter7.png)
![](./output3/AverageBlur_filter5.png)
![](./myface/AverageBlur_filter5.png)

## Medium Blur
* [code](hw2_2_2_Median_Blur.py)
### Result
![](./output/MediumBlur_filter7.png)
![](./output3/MediumBlur_filter5.png)
![](./myface/MediumBlur_filter5.png)

## Gaussian Blur
* [code](hw2_2_3_Gaussian_Blur.py)
### Result
![](./output/GaussianBlur_filter7.png)
![](./output3/GaussianBlur_filter5.png)
![](./myface/GaussianBlur_filter5.png)

# bouns sign name  15%

new_img = sign(new_img)

``` python
def sign(src,threshold=128):
    name_img = cv2.imread("name.jpeg",0)
    name_img = cv2.resize(name_img,(src.shape[0]//2,src.shape[1]//2))
    for i in range(name_img.shape[0]):
        for j in range(name_img.shape[1]):
            if name_img[i,j] < threshold:
                src[name_img.shape[0]+i,name_img.shape[1]+j,:] = [255,255,255]
    return src
```
