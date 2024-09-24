# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 12:05:24 2023

@author: Roya Arian, email: royaarian101@gmail.com
"""

from skimage import measure
import numpy as np

def calculating_predicted_layers(preds_test_t, number_class):
 
    Largest_area = np.zeros_like((preds_test_t)) # removing small white balls
            
    for i in range (preds_test_t.shape[0]):
        for l in range(number_class):
            # removing small white balls
            labels_mask = measure.label(preds_test_t[i,:,:,l])                       
            regions = measure.regionprops(labels_mask)
            regions.sort(key=lambda x: x.area, reverse=True)
            if len(regions) > 1:
                for rg in regions[1:]:
                    labels_mask[rg.coords[:,0], rg.coords[:,1]] = 0
            labels_mask[labels_mask!=0] = 1
            if (l!=1):
                # inverting the image
                # removing small black balls
                img_not = 1 - np.asarray(labels_mask)
                # finding the largest area
                labels_mask = measure.label(img_not)                       
                regions = measure.regionprops(labels_mask)
                regions.sort(key=lambda x: x.area, reverse=True)
                if len(regions) > 1:
                    for rg in regions[1:]:
                        labels_mask[rg.coords[:,0], rg.coords[:,1]] = 0
                labels_mask[labels_mask!=0] = 1
                Largest_area_not = labels_mask
                
                # inverting the image again
                Largest_area[i,:,:,l] = 1 - np.asarray(Largest_area_not)
            else:
                Largest_area[i,:,:,l] = np.asarray(labels_mask)
            
    # One Hot Decoding
    Largest_area_dc = np.argmax(Largest_area, axis = 3) 
    return Largest_area_dc
            