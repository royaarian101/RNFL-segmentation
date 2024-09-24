# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 08:48:31 2023

Utilized functions

@author:Roya Arian, royaarian101@gmail.com
"""


import numpy as np

def dice_coef(y_true, y_pred, smooth=1):
  """
  This function calculates the Dice coefficient for RNFL layer
  
  
  Dice = (2*|X & Y|)/ (|X|+ |Y|)
        =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
  ref: https://arxiv.org/pdf/1606.04797v1.pdf
  """
  intersection = np.sum(np.abs(y_true[:,:,:,1] * y_pred[:,:,:,1]))
  return (2. * intersection + smooth) / (np.sum(np.square(y_true[:,:,:,1]))\
                                          + np.sum(np.square(y_pred[:,:,:,1])) + smooth)


def get_iou(y_true, y_pred):
    """
    This function calculates the Jacard coefficient for RNFL layer
    
    IOU = (|X & Y|)/ (|X U Y|)

          
    """
    EPS = 1e-12
    intersection = np.sum((y_true[:,:,:,1])*(y_pred[:,:,:,1]))
    union = np.sum(np.maximum((y_true[:,:,:,1]), (y_pred[:,:,:,1])))
    iou = float(intersection)/(union + EPS)

    return iou


def Initialize(number_class, nfold, im_size):
    
    """
    This function Initializes some evaluation parameters for the segmentation model
    
    mae: Mean Absolute Error
    pred: predicted by the proposed model
    device: predicted by the Heidelberg Device
    true: predicted by the proposed model
    iou: Jacard
    
    """
    
    d = dict()
    
    ####### for test
    d['kfold_dice_test'] = np.zeros((nfold))
    d['kfold_iou_test']  = np.zeros((nfold))

    d['mae_test']            = np.zeros((nfold))

    d['mae_test_edema']      = np.zeros((nfold))
    d['mae_test_glaucoma']   = np.zeros((nfold))
    d['mae_test_normal']     = np.zeros((nfold))

    d['error_r_folds']          = np.zeros((nfold))
    d['error_i_folds']          = np.zeros((nfold))
    d['error_r_edema_folds']    = np.zeros((nfold))
    d['error_i_edema_folds']    = np.zeros((nfold))
    d['error_r_glaucoma_folds'] = np.zeros((nfold))
    d['error_i_glaucoma_folds'] = np.zeros((nfold))
    d['error_r_normal_folds']   = np.zeros((nfold))
    d['error_i_normal_folds']   = np.zeros((nfold))


    d['thickness_pred_test'] = np.zeros((nfold))
    d['thickness_true_test'] = np.zeros((nfold))

    d['thickness_pred_test_edema'] = np.zeros((nfold))
    d['thickness_true_test_edema'] = np.zeros((nfold))

    d['thickness_pred_test_glaucoma'] = np.zeros((nfold))
    d['thickness_true_test_glaucoma'] = np.zeros((nfold))

    d['thickness_pred_test_normal'] = np.zeros((nfold))
    d['thickness_true_test_normal'] = np.zeros((nfold))


  
    return d




def printing(d, im_size, resolution):
    
    """
   This function prints some evaluation parameters for the bi-modal classifier
    
    mae: Mean Absolute Error
    pred: predicted by the proposed model
    device: predicted by the Heidelberg Device
    true: predicted by the proposed model
    iou: Jacard
    """
    
    print(f"\n\nmean_mae_test      = {(np.mean(d['mae_test'])*resolution)/im_size}")
    print(f"mean_mae_test_edema    = {(np.mean(d['mae_test_edema'])*resolution)/im_size}")
    print(f"mean_mae_test_glaucoma = {(np.mean(d['mae_test_glaucoma'])*resolution)/im_size}")
    print(f"mean_mae_test_normal   = {(np.mean(d['mae_test_normal'])*resolution)/im_size}")
    
    
    print(f"\n\nthickness_pred_test = {(np.mean(d['thickness_pred_test'], axis=0)*resolution)/im_size}")
    print(f"thickness_true_test     = {(np.mean(d['thickness_true_test'], axis=0)*resolution)/im_size}",end='\n')
    
    print(f"\nthickness_pred_test_edema    = {(np.mean(d['thickness_pred_test_edema'], axis=0)*resolution)/im_size}")
    print(f"thickness_true_test_edema      = {(np.mean(d['thickness_true_test_edema'], axis=0)*resolution)/im_size}",end='\n')
    print(f"\nthickness_pred_test_glaucoma = {(np.mean(d['thickness_pred_test_glaucoma'], axis=0)*resolution)/im_size}")
    print(f"thickness_true_test_glaucoma   = {(np.mean(d['thickness_true_test_glaucoma'], axis=0)*resolution)/im_size}",end='\n')
    print(f"\nthickness_pred_test_normal   = {(np.mean(d['thickness_pred_test_normal'], axis=0)*resolution)/im_size}")
    print(f"thickness_true_test_normal     = {(np.mean(d['thickness_true_test_normal'], axis=0)*resolution)/im_size}",end='\n')
    

    print(f"\n\nRNFL_unsigned_error_device = {(np.mean(d['error_r_folds'])/im_size)*resolution}")
    print(f"ILM_unsigned_error_device  = {(np.mean(d['error_i_folds'])/im_size)*resolution}")
    
    print(f"\n\nRNFL_unsigned_error_device_edema = {(np.mean(d['error_r_edema_folds'])/im_size)*resolution}")
    print(f"ILM_unsigned_error_device_edema  = {(np.mean(d['error_i_edema_folds'])/im_size)*resolution}")
    
    print(f"\n\nRNFL_unsigned_error_device_glaucoma = {(np.mean(d['error_r_glaucoma_folds'])/im_size)*resolution}")
    print(f"ILM_unsigned_error_device_glaucoma = {(np.mean(d['error_i_glaucoma_folds'])/im_size)*resolution}")
    
    print(f"\n\nRNFL_unsigned_error_device_normal = {(np.mean(d['error_r_normal_folds'])/im_size)*resolution}")
    print(f"ILM_unsigned_error_device_normal = {(np.mean(d['error_i_normal_folds'])/im_size)*resolution}")
    
    print(f"\n\n\nkfold_dice_test = {np.mean(d['kfold_dice_test'])}")
    print(f"kfold_iou_test  = {np.mean(d['kfold_iou_test'])}")




