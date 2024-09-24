# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 15:18:30 2023

@author: Roya Arian, email: royaarian101@gmail.com
"""

import numpy as np
import scipy.signal

def cross_image(im1, im2):
   # get rid of the color channels by performing a grayscale transform
   # the type cast into 'float' is to avoid overflows
   im1_gray = np.sum(im1.astype('float'), axis=2)
   im2_gray = np.sum(im2.astype('float'), axis=2)

   # get rid of the averages, otherwise the results are not good
   im1_gray -= np.mean(im1_gray)
   im2_gray -= np.mean(im2_gray)

   # calculate the correlation image; note the flipping of onw of the images
   
   corr_img_self = scipy.signal.fftconvolve(im1_gray, im1_gray[::-1,::-1], mode='same')
   corr_img = scipy.signal.fftconvolve(im2_gray, im1_gray[::-1,::-1], mode='same')
   maxpoint = np.unravel_index(np.argmax(corr_img_self), corr_img_self.shape)
   shift = np.unravel_index(np.argmax(corr_img), corr_img.shape)
   
   return maxpoint[0] - shift[0]
