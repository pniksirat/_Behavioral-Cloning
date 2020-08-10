#!/usr/bin/env python
import cv2
import numpy as np
import matplotlib.image as mpimg
import random
import warnings

#looked up discussion on the forum:
#https://stackoverflow.com/questions/32609098/how-to-fast-change-image-brightness-with-python-opencv
# changing the V channel randomly
def augment_brightness(image):
    if random.randint(0,10)<5:
        image_m = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
        image_m = np.array(image_m, dtype = np.float64)
        #create a distribution at 0.5 med and dev=0.3
        random_bright = np.random.uniform()*0.4+0.5
        image_m[:,:,2] = image_m[:,:,2]*random_bright

        lim = 255 - random_bright
        image_m[:,:,2][image_m[:,:,2]>lim] = 255
        image_m[:,:,2][image_m[:,:,2]<lim] += random_bright
   
    #image_m[:,:,2][image_m[:,:,2]>255]  = 255
        image_m = cv2.cvtColor(np.array(image_m, dtype = np.uint8),cv2.COLOR_HSV2RGB)
    else:
        image_m=image
    cv2.imwrite('images/brightness3.png', image_m)
    return image_m

def random_flip(image, steering_angle):
    if random.randint(0,10)<4:    
        image = cv2.flip(image, 1)
        #if flipped then stearing angle needs to be updated
        steering_angle = -steering_angle
    #cv2.imwrite('images/flipped.png', image)
    return image, steering_angle

#shadow3 creates a blocks of solid shade not so effective
def rand_shadow3(img):
    
    #if random.randint(0,10)<5:
    imgSize=img.shape
    fit_=[]
    px1=0
    c=0
    for i in range(2):
         px1=np.random.randint(px1,imgSize[1])
         px2=np.random.randint(px1,imgSize[1])
         if px2<c:
            px2=px2+abs(px2-c)
         c=px2
         py1=np.int32(imgSize[0]/1.7)
         py2=np.int32(imgSize[0])
         print((px1,py1),(px2,py2))
         #poly.append((px1,py1),(px2,py2))
         #poly.append((px2,py2))
         fit_.append( np.polyfit((px1,py1),(px2,py2), 1))
    # np.polyfit() returns the coefficients [A, B] of the fit
    
    # Find the region inside the lines
    XX, YY = np.meshgrid(np.arange(0, imgSize[1]), np.arange(0, imgSize[0]))
    region_thresholds = (YY > (XX*fit_[0][0] + fit_[0][1])) & \
                         (YY > (XX*fit_[1][0] + fit_[1][1]))
    region_select = np.copy(img)
    region_select[region_thresholds]=[80,80,80]
    print(fit_)
    #cv2.imwrite('shadow.png', region_select)
      
    return img

#shadow 2 makes more reasonable shades
def rand_shadow2(img):
    warnings.filterwarnings('ignore')
    if random.randint(0,10)<5:
        imgSize=img.shape
        fit_=[]
        px1=np.random.randint(0,imgSize[1])
        px2=np.random.randint(0,imgSize[1])
        #py1=np.int32(imgSize[0]/1.5)
        py1=70
        py2=np.int32(imgSize[0])
        #print((px1,py1),(px2,py2))

        fit_.append( np.polyfit((px1,py1),(px2,py2), 1))
        # np.polyfit() returns the coefficients [A, B] of the fit
        m=fit_[0][0]
        b=np.random.randint(-50,50)+fit_[0][1]
        fit_.append(np.array([m,b]))
        # Find the region inside the lines
        #print(fit_)
   
        XX, YY = np.meshgrid(np.arange(0, imgSize[1]), np.arange(0, imgSize[0]))
        region_thresholds = (YY > (XX*fit_[0][0] + fit_[0][1])) & \
                            (YY > (XX*fit_[1][0] + fit_[1][1]))
        region_select = np.copy(img)
        img_hls = cv2.cvtColor(region_select,cv2.COLOR_RGB2HLS)
        #image_hls[:,:,1][region_thresholds]=image_hls[:,:,1]*0.2
        #print(image_hls[:,:,1].shape)
        img_hls[:,:,1][region_thresholds]=img_hls[:,:,1][region_thresholds]*0.45
        img_shade= cv2.cvtColor(img_hls,cv2.COLOR_HLS2RGB) 
        #cv2.imwrite('shadow2.png', img_shade)
    else:
        img_shade= np.copy(img)
     
    return img_shade

"""
def rand_shadow(img):
    
    #if random.randint(0,10)<5:
    imgSize=img.shape
    poly=[]
    for i in range(2):
         px1=np.random.randint(0,imgSize[1])
         px2=np.random.randint(0,imgSize[1])
         py1=np.int32(imgSize[0]/2)
         py2=np.int32(imgSize[0])
         poly.append((px1,py1))
         poly.append((px2,py2))
    print(poly)
    print(imgSize)
    mask=np.zeros_like(img)
    cv2.fillPoly(img,np.array([poly]), (80,80,80))
    #masked_edges=cv2.bitwise_and(img,mask)
    cv2.imwrite('shadow.png', img)
    cv2.imwrite('shadow2.png', mask)
      
    return img
"""


def random_shear(image,steering):
    rows,cols,ch = image.shape
    dx = np.random.randint(-cols/2,cols/2+1)
    #print('dx',dx,'shape', image.shape)
    random_point = [cols/2+dx,rows/2]
    pts1 = np.float32([[0,rows],[cols,rows],[cols/2,rows/2]])
    pts2 = np.float32([[0,rows],[cols,rows],random_point])
    dsteering = dx/(rows/2) * 360/(2*np.pi*25.0) / 6.0    
    M = cv2.getAffineTransform(pts1,pts2)
    image = cv2.warpAffine(image,M,(cols,rows),borderMode=1)
    steering +=dsteering
    #cv2.imwrite('warped.png', image)
    
    return image,steering


#********testing*********************
"""
img = mpimg.imread('test1.jpg')
img2=augment_brightness(img)
img=random_shear(img,0)
random_flip(img2, 0.2)


img = mpimg.imread('images/center_930.jpg')
img=augment_brightness(img)

#rand_shadow2(img)
"""