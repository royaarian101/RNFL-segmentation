# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 11:59:32 2022

@author: Roya Arian
"""
import numpy as np
from skimage.transform import resize

    
def preparing(x, y, number_class):
    if number_class==2:
        l = 255
    elif number_class==3:
        l = 85
        
    data  = []
    label = []
    name  = []
    for i in x:
        for j in x[i]:
            data.append(x[i][j])
            name.append(j)           
            label_temp = np.zeros((*(np.shape(x[i][j]))[:2], number_class))
            for c in range(number_class):
                label_temp[:,:,c] = np.where(np.round(y[i][j]/l)[:,:,0]==c, 1, 0)
            
            label.append(label_temp)
    
    data  = np.reshape(data, np.shape(data))
    label = np.reshape(label, np.shape(label))
    data  = resize((data),(np.shape(data)[0], np.shape(data)[1], np.shape(data)[2], 1),\
        mode = 'constant', preserve_range = True)
    
    return data, label, name
