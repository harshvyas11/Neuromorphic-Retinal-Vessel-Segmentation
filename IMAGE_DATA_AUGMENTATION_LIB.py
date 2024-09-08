# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 09:30:11 2024

@author: Admin
"""



import  cv2
import numpy as np


def getRandomAffineTransform_2D(translation, rotation_range, scale_range, im_shape):
    height, width = im_shape[:2]
    center = (width/2-width*np.random.uniform(-translation,translation), 
              height/2-height*np.random.uniform(-translation,translation))
    
    rotation = np.random.uniform(rotation_range[0],rotation_range[1])
    scale = np.random.uniform(scale_range[0],scale_range[1])
    
    AffineMat = cv2.getRotationMatrix2D(center, rotation, scale)
    
    return AffineMat


def Augment_RandomAffine_2D(img, mask=None, translation=0.05, rotation_range=[-10,10], scale_range=[0.95, 1.05], interpolation=cv2.INTER_NEAREST):

    AffineMat = getRandomAffineTransform_2D(translation, 
                                            rotation_range, 
                                            scale_range, 
                                            im_shape=img.shape)
    height, width = img.shape[:2]
    img = cv2.warpAffine(img,AffineMat,dsize=(height, width), flags=interpolation)
    if mask is not None:
        mask = cv2.warpAffine(mask,AffineMat,dsize=(height, width), 
                              flags=interpolation,
                              borderMode=cv2.BORDER_CONSTANT, 
                              borderValue=0)
        #print('returning img and mask')
        return img, mask
    else:
        return img
        


#Augment_RandomAffine_2D(img, mask=None, translation=0.05, rotation_range=[-10,10], scale_range=[0.95, 1.05], interpolation=cv2.INTER_NEAREST):

#AffineMat = getRandomAffineTransform_2D(translation=0.05, rotation_range=[-10,10], scale_range=[0.95, 1.05], im_shape=[256, 256])
#print(AffineMat)