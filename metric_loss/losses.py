# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 10:11:54 2023
loss functions
@author: Roya Arian, email: royaarian101@gmail.com
"""


#### loss functions ####
from keras import backend as K
import os
import numpy as np
os.environ['KERAS_BACKEND'] = 'theano'

def combined_loss(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    
    weights = K.variable(weights)
        
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    
    def dice_coef(y_true, y_pred, smooth=1):
        """
        Dice = (2*|X & Y|)/ (|X|+ |Y|)
              =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
        ref: https://arxiv.org/pdf/1606.04797v1.pdf
        """
        intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
        return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1)\
                                               + K.sum(K.square(y_pred),-1) + smooth)
    
    def dice_coef_loss(y_true, y_pred, smooth=1):
        """
        Dice = (2*|X & Y|)/ (|X|+ |Y|)
              =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
        ref: https://arxiv.org/pdf/1606.04797v1.pdf
        """
        intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
        return 1 - (2. * intersection + smooth) / (K.sum(K.square(y_true),-1)\
                                               + K.sum(K.square(y_pred),-1) + smooth)
           
        
    def total_variation_loss(y_pred):
        img_width=np.size(y_pred,1)
        img_height=np.size(y_pred,1)
        a = K.square(
            y_pred[:, :img_height - 1, :img_width - 1, :] -
            y_pred[:, 1:, :img_width - 1, :])
        b = K.square(
            y_pred[:, :img_height - 1, :img_width - 1, :] -
            y_pred[:, :img_height - 1, 1:, :])
        return K.sum(K.pow(a + b, 1.25))/(255*128*128)  
    
    
    def tversky_loss(y_true, y_pred, alpha=0.3, beta=0.7, smooth=1e-10):
        y_true = K.flatten(y_true) 
        y_pred = K.flatten(y_pred) 
        truepos = K.sum(y_true * y_pred) 
        fp_and_fn = alpha * K.sum(y_pred * (1 - y_true)) + beta * K.sum((1 - y_pred) * y_true) 
        answer = (truepos + smooth) / ((truepos + smooth) + fp_and_fn) 
        return -answer
    
    def WCCE_dice_loss(y_true, y_pred):
        return loss(y_true, y_pred)+dice_coef_loss(y_true, y_pred)
    
    
    def WCCE_dice_tv_loss(y_true, y_pred):
        return loss(y_true, y_pred)+dice_coef_loss(y_true, y_pred)+total_variation_loss( y_pred)
    
    
    def WCCE_dice_tversky_loss(y_true, y_pred):
        return loss(y_true, y_pred)+dice_coef_loss(y_true, y_pred)+tversky_loss(y_true, y_pred, alpha=0.3, beta=0.7, smooth=1e-10)
    
    def all_losses(y_true, y_pred):
        return loss(y_true, y_pred)+dice_coef_loss(y_true, y_pred)+\
            tversky_loss(y_true, y_pred, alpha=0.3, beta=0.7, smooth=1e-10)\
                +total_variation_loss( y_pred)
    
    return WCCE_dice_loss, WCCE_dice_tv_loss, WCCE_dice_tversky_loss, dice_coef, dice_coef_loss, all_losses