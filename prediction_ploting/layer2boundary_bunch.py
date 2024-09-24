# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 12:26:40 2023

@author: Roya Arian, email: royaarian101@gmail.com
"""
import numpy as np


# layer2boundary_bunch_Makeboundry function
width = 200
def layer2boundary_bunch(layer, number_class, im_size):
    image_size=im_size-1
    segmented_lines_quick = np.zeros((layer.shape[0], 2 , layer.shape[2]))
    loc_image = layer.copy()
    loc=np.where(loc_image==0,-1,1)


    for sampel in range(layer.shape[0]):
        
        if number_class == 2:
            for j in range(np.shape(layer)[1]):
                a = np.where(layer[sampel,:,j] == 1) 
                if len(a[0])>1:
                    segmented_lines_quick[sampel, 0 , j] = a[0][0]
                if len(a[0])>=2:
                    segmented_lines_quick[sampel, 1 , j] = a[0][-1]
                elif len(a[0])==1:
                    segmented_lines_quick[sampel, 1 , j] = a[0][0]
                else:
                    if j-3>0:
                        segmented_lines_quick[sampel, 0 , j] = np.where(layer[sampel,:,j-3] == 1)[0][0]
                        segmented_lines_quick[sampel, 1 , j] = np.where(layer[sampel,:,j-3] == 1)[0][-1]
                    else:
                        segmented_lines_quick[sampel, 0 , j] = np.where(layer[sampel,:,j+1] == 1)[0][0]
                        segmented_lines_quick[sampel, 1 , j] = np.where(layer[sampel,:,j+1] == 1)[0][-1]
                    
        else:
            boundries = np.zeros((width,image_size+1))
            last_boundries = np.zeros((image_size+1))
            last_boundries=np.where(last_boundries==0,np.nan,last_boundries) 
            for i in range (layer.shape[2]):
                if (len(np.where(np.diff(np.sign(loc[sampel,:,i])))[0])!=0):
                    b = np.where(np.diff(np.sign(loc[sampel,:,i])))[0] 
                    boundries [0:len(b),i] = b 
                if (len(boundries[1])>image_size+1):
                    boundries [:,i] = boundries [:,0:i-1]
                boundries=np.where(boundries==0,np.nan,boundries) 
                last_boundries[i] = boundries[0,i]
                segmented_lines_quick[sampel, 0 , :] = last_boundries
        
    
    if number_class == 3:
 
    # Finding second boundary 
        loc_image = layer.copy()
        loc=np.where(loc_image==1,1,-1)
        image_size=im_size-1
        
        for sampel in range(layer.shape[0]):
            boundries = np.zeros((width,image_size+1))
            last_boundries = np.zeros((image_size+1))
            last_boundries=np.where(last_boundries==0,np.nan,last_boundries) 
            for i in range (layer.shape[2]):
                if (len(np.where(np.diff(np.sign(loc[sampel,:,i])))[0])!=0):
                    b = np.where(np.diff(np.sign(loc[sampel,:,i])))[0] 
                    boundries [0:len(b),i] = b 
                if (len(boundries[1])>image_size+1):
                    boundries [:,i] = boundries [:,0:i-1]
                boundries=np.where(boundries==0,np.nan,boundries) 
                last_boundries[i] = boundries[0,i]
                segmented_lines_quick[sampel, 1 , :] = last_boundries

    return segmented_lines_quick