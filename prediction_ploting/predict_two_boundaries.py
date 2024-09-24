# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 12:23:53 2023

@author: Roya Arian, email: royaarian101@gmail.com
"""
import numpy as np
from prediction_ploting.layer2boundary_bunch import layer2boundary_bunch

def predict_two_boundaries(predictions, number_class, im_size):

    # Normalize masks
    predictions_normal = np.zeros_like(predictions)

    prediction_local = predictions
    prediction_local = (prediction_local - np.min(prediction_local))/(np.max(prediction_local) - np.min(prediction_local))
    predictions_normal = prediction_local
    segmented_lines = layer2boundary_bunch(predictions_normal, number_class, im_size)
    return segmented_lines