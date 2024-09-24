# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 15:24:31 2023

@author: Roya Arian, email: royaarian101@gmail.com
"""

import cv2
import numpy as np
from preprocess.cross_image import cross_image
import matplotlib.pyplot as plt

def shifting_image(image_ref, image):
    
    shift = cross_image(image_ref, image)
    
    M = np.float32([[1, 0, 0], [0, 1, shift]])
     
    (rows, cols) = image.shape[:2]
 
    # warpAffine does appropriate shifting given the
    # translation matrix.
    res = cv2.warpAffine(image, M, (cols, rows))
    
    if __name__ == '__main__':
        plt.subplot(2, 1, 1)
        plt.imshow(image_ref[:,:,0], cmap='gray')
        plt.subplot(2, 1, 2)
        plt.imshow(res[:,:,0], cmap='gray')
    
    return res
