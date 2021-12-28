@jit(nopython=True)
def Non_Maximum_Suppression(img,map):
    reuslt_img = np.zeros_like(img)
    for i in range(1,img.shape[0]-1):
        for j in range(1,img.shape[1]-1):
                if map[i,j] == 1:
                    x,y = [i+1,j]
                    while (map[x,y] == map[i,j] and x<img.shape[0]-1 and y<img.shape[1]-1):
                        x +=1
                    # map[range(i,x+1) , j] = -1
                    map[i:x , j] = 0
                    x = np.argmax(img[i:x,y]) + i 
                    y = y 
                    reuslt_img[x,y] = img[x,y]
                    
                elif map[i,j] == 2:
                    x,y = [i,j+1]
                    while (map[x,y]==map[i,j] and x<img.shape[0]-1 and y<img.shape[1]-1):
                        y +=1
                    map[i, j:y] = 0
                    x = i
                    y = np.argmax(img[x, j:y ]) + j
                    reuslt_img[x,y] = img[x,y]

                elif map[i,j] == 3:
                    # print(i,j)
                    x,y = [i+1,j+1]
                    while (map[x,y]==map[i,j] and x<img.shape[0]-1 and y<img.shape[1]-1):
                        x +=1
                        y +=1
                    map[range(i,x+1),range(j,y+1)] = 0
                    # print(np.argmax(img[range(i,x+1),range(j,y+1)]))
                    # x = i + np.argmax(img[range(i,x+1),range(j,y+1)])
                    # y = j + x-i
                    arg = np.argmax(img[range(i,x+1),range(j,y+1)])
                    x = i+arg
                    y = j+arg
                    reuslt_img[x,y] = img[x,y]
    for i in range(1 , img.shape[0]-1):
        for j in range(img.shape[1]-1 , 1 , -1):
            if map[i,j] == 4:
                x,y = [i,j]
                while (map[x,y] == map[i,j] and y<img.shape[1]-1 and x>0 and y>0 and x<img.shape[0]-1):
                    x = x+1
                    y -= 1
                map[range(i,x+1),range(j,y-1,-1)] = -1
                arg = np.argmax(img[range(i,x+1),range(j,y-1,-1)])
                x = i + arg
                y = j - arg
                reuslt_img[x,y] = img[x,y]
    return reuslt_img