# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 11:27:38 2023

@author: Roya Arian, email: royaarian101@gmail.com
"""

import numpy as np

# parametes
k=500
q=20000
alpha=2
beta=0.8

def weights_Definition(y_train, number_class):
    w1=np.zeros((number_class))
    w2=np.zeros((number_class))
    w3=np.zeros((number_class))
    w4=np.zeros((number_class))
    
    # w1: assign w1=1 for background and w1=2 for choroid layer  
    for j in range (number_class):
        if ((j==0) or (j==number_class-1)):
            w1[j]= 1
        else:
            w1[j]= 2
    fij=0        
    for j in range (number_class):
        for i in range (np.size(y_train,0)):
            fij = fij+ np.sum(y_train[i][:,:,j])/(np.size(y_train,1)*np.size(y_train,2))
        fj = fij/(np.size(y_train,0))
        w2[j] = k/fj
        w3[j] = np.log(q/fj)    
        w4[j] = (1-np.power(beta,fj))/(1-beta)
    
    return w1, w2, w3, w4   
    
def weights(y_train, number_class, w=5):
    switcher={
        1:np.ones((number_class)),                         # first version of Balanced Cross Entropy
        2:weights_Definition(y_train, number_class)[0],    # Balanced Cross Entropy
        3:weights_Definition(y_train, number_class)[1],    # Inverse class freq. linear (k=500)
        4:weights_Definition(y_train, number_class)[2],    # Inverse class freq. logarithmic (q=20000)
        5:weights_Definition(y_train, number_class)[3]     # Effective Number of Object Class (beta=0.8)
      }
    return switcher.get(w,"weights")
    