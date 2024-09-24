# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 12:36:04 2023

@author: Roya Arian, email: royaarian101@gmail.com
"""
import numpy as np
import matplotlib.pyplot as plt

def ploting_pred_boundaries(segmented_lines_quick, true_lines_quick, x_test, name):
    for j in range (x_test.shape[0]):
        for num_seglayer in range(np.size(segmented_lines_quick,1)):
            pred_layer = segmented_lines_quick[j,num_seglayer,:]
            true_layer=true_lines_quick[j,num_seglayer,:]
            plt.plot(pred_layer,'b')
            plt.plot(true_layer,'r')
            plt.title(f'{name[j]}')
            plt.imshow(x_test[j,:,:],cmap='gray')
        plt.show()