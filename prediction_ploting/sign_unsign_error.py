# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 12:56:42 2023

@author: Roya Arian, email: royaarian101@gmail.com
"""

import numpy as np
res = 1250
#461 pixels the whole image, 69 pixels = 200 um, so 461 pixels = 1317
#we have used just 437 pixels of image, so 437 pixels ~= 1250 um
    
def sign_unsign_error(segmented_lines_quick, true_lines_quick): 
    where_are_NaNs = np.isnan(segmented_lines_quick)
    segmented_lines_quick[where_are_NaNs] = 0
    where_are_NaNs = np.isnan(true_lines_quick)
    true_lines_quick[where_are_NaNs] = 0

    signed_error=np.zeros((np.size(true_lines_quick,0), np.size(true_lines_quick,1)))
    unsigned_error=np.zeros((np.size(true_lines_quick,0), np.size(true_lines_quick,1)))

    mean_signed_error_micro = np.zeros((np.size(true_lines_quick,1)))
    mean_signed_error_pixel=np.zeros((np.size(true_lines_quick,1)))
    mean_unsigned_error_micro = np.zeros((np.size(true_lines_quick,1)))
    mean_unsigned_error_pixel = np.zeros((np.size(true_lines_quick,1)))
    
    for i in range (np.size(true_lines_quick,1)):
        n = 0
        for j in range (np.size(true_lines_quick,0)):
            signed_error[n,i] = np.sum(np.subtract(true_lines_quick[j,i,:], segmented_lines_quick [j,i,:]))/(np.size(true_lines_quick,2))
            unsigned_error[n,i] = abs (np.sum(np.subtract(true_lines_quick[j,i,:], segmented_lines_quick [j,i,:]))/(np.size(true_lines_quick,2)))
            n = n+1
            
        mean_signed_error_pixel[i]= np.sum( signed_error[:,i] )/n
        mean_signed_error_micro[i] = (mean_signed_error_pixel[i])*res/np.size(true_lines_quick,2)
        
        mean_unsigned_error_pixel[i]= np.sum(unsigned_error[:,i])/n
        mean_unsigned_error_micro[i]= (mean_unsigned_error_pixel[i])*res/np.size(true_lines_quick,2)

        print("mean_unsigned_error_ab_overall Layer ", i+1 , "(in pixel) = ",mean_unsigned_error_pixel[i])
        print("mean_unsigned_error_overall Layer ", i+1 , "(in micro) = ",mean_unsigned_error_micro[i])
        
        
        print("mean_signed_error_ab_overall Layer ", i+1 , "(in pixel) = ",mean_signed_error_pixel[i])
        print("mean_signed_error_overall Layer ", i+1 , "(in micro) = ",mean_signed_error_micro[i])

        
    return mean_signed_error_pixel, mean_signed_error_micro, \
        mean_unsigned_error_pixel, mean_unsigned_error_micro
        