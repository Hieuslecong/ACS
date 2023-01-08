import cv2 
import glob
import matplotlib.pyplot as plt 
import os
import numpy as np
import imutils
from tqdm import tqdm
from sklearn.metrics import f1_score,confusion_matrix,fbeta_score
def Accuracy1(GT,seg,beta):
    r = []
    p = []
    F = 0
    #[x,y] = np.argwhere(GT >0)
    GT[GT >0] = 1
    GT = np.ndarray.flatten(GT)
    #[x,y] = np.argwhere(seg > 0)
    seg[seg > 0] = 1
    seg = np.ndarray.flatten(seg)
    CM = confusion_matrix(GT,seg)
    c = np.shape(CM)
    for i in range(c[1]):
        if (np.sum(CM[i,:]) == 0):
            r.append(0)
        else:
            a = CM[i,i]/(np.sum(CM[i,:]))
            r.append(a)
        if (np.sum(CM[:,i]) == 0):
            p.append(0)
        else:
            p.append(CM[i,i]/(np.sum(CM[:,i])))
    F = (1+beta)*(np.mean(r)*np.mean(p))/(beta*np.mean(p)+np.mean(r))
    return F,np.mean(p),np.mean(r)

numlist = [0] * 250
list_fall=[]
img_paper = glob.glob(r'D:\Tai Xuong0\crack datasets\CRACK500\test/'+'*.png')
img_result = glob.glob(r'D:\pix2pixHD\dataset\SNAT_R4_Crack500_Results\FCN/'+'*.png')
for k ,img_path in enumerate(img_result):
    print('%s/%s     %s' %(k,len(img_result),img_path))
    
    a=256
    name12= (os.path.split(img_path)[-1]).split(".")[0]
    #print(name12)
    image_result = cv2.imread(img_path)
    image_result=cv2.resize(image_result,(a,a))
    image_result[image_result>0]=1
    #print(np.unique(image_result))
    min_img=0
    min_2=100000000
    img_out = []
    name_out = []
    img_out1=[]
    loop = tqdm(img_paper)
    for batch_idx , path_real in enumerate(loop):
    #for path_real in loop:
        name_img = (os.path.split(path_real)[-1]).split(".")[0]
        image_real=cv2.imread(path_real)
        for angle in np.arange(0, 360,90):
            rotated = imutils.rotate(image_real, angle)
            image_real=cv2.resize(image_real,(a,a))
            image_real[image_real>0]=1
            #print(np.unique(image_real))
            #print(aaaa)
            img_test= image_real - image_result

            #print(np.unique(img_test))
            a1,b1,c1=Accuracy1(image_real,image_result,0.3)
            
            #print(a1)
            point = len(np.argwhere(img_test==255))
            if a1>min_img:
                min_img= a1
                img_out = image_real
                name_out = name_img
            if point<min_2:
                min_2=point
                img_out1 = image_real
                name1 = name_img
        
        loop.set_postfix(f1_score=min_img)
    if min_img<0.7:
        list_fall.append(name12)
    print(list_fall)
    numlist[int(name12)] = name_out
    print(min_img,min_2,name_out)
    rows, cols = 1, 3
    # plt.subplot(rows, cols, 1)
    # plt.imshow(image_result*255)
    # plt.title("Figure 1")
    # plt.subplot(rows, cols, 2)
    # plt.imshow(img_out*255)
    # plt.title("Figure 2")
    # plt.subplot(rows, cols, 3)
    # plt.imshow(img_out1*255)
    # plt.title("Figure 3")


    # plt.show()  
with open("list_image_crack500.txt", "w") as f:
    for item in numlist:
        # write each item on a new line
        f.write("%s\n" % item)
    print('Done')
print(list_fall)
    