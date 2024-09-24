# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 09:53:00 2023
RNFL segmentation
@author: Roya Arian, email: royaarian101@gmail.com
"""
#import
import os
from pathlib import Path
import segmentation_models
import numpy as np # linear algebra
import matplotlib.pyplot as plt
from keras.layers import Input
import pickle
from sklearn.model_selection import KFold
import metric_loss
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import prediction_ploting
import preprocess
import sklearn
import utils


####### set below parameters
batch_size    = 16   # Please tune the batch-size due to your data
epochs        = 100  # Please tune the number of epochs
im_size       = 128  # Please choose among 128, 256 or 512
number_class  = 3    # Please choose number of classes 2 or 3
learning_rate = 1e-4 # Please tune the learning rate due to your data
resolution    = 1250 # Please enter your image resolution
nfold         = 5    # Please enter number of folds

# initial some parameters
d = utils.Initialize(number_class, nfold, im_size)



### pickle data path

pkl_data_path= str(Path(os.getcwd()).parent) + "\\Data\\pkl_data"

file = open(os.path.join(pkl_data_path, "train_dataset.pkl"), 'rb')
subjects = pickle.load(file)

file = open(os.path.join(pkl_data_path, "train_dataset_inpainted.pkl"), 'rb')
aligned_images_subject = pickle.load(file)

file = open(os.path.join(pkl_data_path, "train_lable.pkl"), 'rb')
label_3class = pickle.load(file)


file = open(os.path.join(pkl_data_path, "test_dataset_inpainted.pkl"), 'rb')
aligned_images_test = pickle.load(file)

file = open(os.path.join(pkl_data_path, "test_lable.pkl"), 'rb')
label_3class_test = pickle.load(file)

file = open(os.path.join(pkl_data_path, "class_labels_test.pkl"), 'rb')
class_labels_test = pickle.load(file)


########## test data
aligned_images_test, label_3class_test, name_test = preprocess.preparing(aligned_images_test, label_3class_test, number_class)
y_test_dc = np.argmax(label_3class_test, axis = 3)
x_test = aligned_images_test[:,:,:,0]
true_lines_quick_test = prediction_ploting.predict_two_boundaries(y_test_dc, number_class, im_size)
Largest_area_dc_test  = np.zeros((nfold, np.shape(aligned_images_test)[0], im_size, im_size))
####################################################################
# Applying kfold
####################################################################


############# Kfold-cross-validation
kf_nfold = KFold(n_splits=nfold, shuffle=True)

n = 0
### Kfold
for train_index, val_index in kf_nfold.split(aligned_images_subject,label_3class):
    n += 1
    # print(val_index)  # you can watch validation index using this comment
    print('%dth fold' % n)


    x_train = {k: aligned_images_subject[list(aligned_images_subject.keys())[k]] for k in train_index}
    x_valid = {k: aligned_images_subject[list(aligned_images_subject.keys())[k]] for k in val_index}

    y_train = {k: label_3class[list(label_3class.keys())[k]] for k in train_index}
    y_valid = {k: label_3class[list(label_3class.keys())[k]] for k in val_index}

    ################## preparing
    x_train, y_train, name_train = preprocess.preparing(x_train, y_train, number_class)
    x_valid, y_valid, name_valid = preprocess.preparing(x_valid, y_valid, number_class)

    ################## shuffling
    indices = np.random.permutation (len (x_train))
    x_train = x_train [indices]
    y_train = y_train [indices]


    if np.max(x_train[1,:,:,0])>1:
      x_train /= 255
      x_valid /= 255
    # One Hot Decoding of the test labels
    y_valid_dc = np.argmax(y_valid, axis = 3)
    x_validd = x_valid[:,:,:,0]
    true_lines_quick_valid = prediction_ploting.predict_two_boundaries(y_valid_dc, number_class, im_size)
    
    ####################################################################
    # Choosing weights for WCCE
    ####################################################################
    # please choose one of following weigths as w in metric_loss.weights.weights function
        # w=1   # first version of Balanced Cross Entropy
        # w=2   # Balanced Cross Entropy
        # w=3   # Inverse class freq. linear (k=500)
        # w=4   # Inverse class freq. logarithmic (q=20000)
        # w=5   # Effective Number of Object Class (beta=0.8)
    weights = metric_loss.weights(y_train, number_class, w=5)

    ####################################################################
    # Choosing loss function
    ####################################################################

    WCCE_dice_loss, WCCE_dice_tversky_loss, WCCE_dice_tv_loss , dice_coeff, dice_coef_loss, all_losses = metric_loss.losses.combined_loss(weights)

    ##### Model Definition
    input_img = Input((im_size, im_size,1))
    model = segmentation_models.get_unet(input_img, number_class, n_filters=32, dropout=0.1, batchnorm=True)

    # choose one of the loss functions
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=[WCCE_dice_loss], metrics=[dice_coeff])


    callbacks = [EarlyStopping(patience=20 , verbose=1), ReduceLROnPlateau(factor=0.1, patience=10, min_lr=1e-6, verbose=1),
        ModelCheckpoint(f'rnfl{n}.weights.h5', verbose=1, save_best_only=True, save_weights_only=True)
    ]

    results = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=callbacks,\
                        validation_data=(x_valid, y_valid))


    plt.figure(figsize=(8, 8))
    plt.title("Learning curve")
    plt.plot(results.history["loss"], label="loss")
    plt.plot(results.history["val_loss"], label="val_loss")
    plt.plot( np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="r", label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("log_loss")
    plt.legend()


    ###############################################################
    ###############################################################
    # Prediction on test data
    ###############################################################
    ###############################################################
    # load the best model
    model.load_weights(f'rnfl{n}.weights.h5')

    preds_test = model.predict(aligned_images_test, verbose=1)
    preds_test_t = (preds_test > 0.5).astype(np.float64)

    d['kfold_dice_test'][n-1] = utils.dice_coef(label_3class_test, preds_test_t)
    d['kfold_iou_test'][n-1]  = utils.get_iou(label_3class_test, preds_test_t)
    ########################################################
    # Calculating predicted layers
    ########################################################
    # Calculating and One Hot Decoding of the predicted labels
    Largest_area_dc_test[n-1] = prediction_ploting.calculating_predicted_layers(preds_test_t, number_class)
    #####################################
    # finging boundaries
    #####################################
    # 2 boundaries of real labels
    d['segmented_lines_quick_test'] = prediction_ploting.predict_two_boundaries(Largest_area_dc_test[n-1], number_class, im_size)


    # prediction_ploting.ploting_pred_boundaries(segmented_lines_quick_valid,\
    #                                             true_lines_quick_valid, x_validd, name_valid)

    t   = np.zeros((len(d['segmented_lines_quick_test']))) # predicted lines
    t_t = np.zeros((len(d['segmented_lines_quick_test']))) # true lines

    ##### for each class
    t_edema         = []
    t_edema_true    = []

    t_normal        = []
    t_normal_true   = []

    t_glaucoma      = []
    t_glaucoma_true = []

    error_r = np.zeros((len(class_labels_test)))
    error_i = np.zeros((len(class_labels_test)))

    error_r_edema    = []
    error_i_edema    = []
    error_r_glaucoma = []
    error_i_glaucoma = []
    error_r_normal   = []
    error_i_normal   = []

    for i in range(len(d['segmented_lines_quick_test'])):
         error_r[i] = np.mean(np.abs((d['segmented_lines_quick_test'][i,1,:] - true_lines_quick_test[i,1,:])))
         error_i[i] = np.mean(np.abs((d['segmented_lines_quick_test'][i,0,:] - true_lines_quick_test[i,0,:])))
         t[i]   = np.mean(np.abs(d['segmented_lines_quick_test'][i,1,:] - d['segmented_lines_quick_test'][i,0,:]))
         t_t[i] = np.mean(np.abs(true_lines_quick_test[i,1,:] - true_lines_quick_test[i,0,:]))

         if class_labels_test[i] == 2:
            t_edema.append(t[i])
            t_edema_true.append(t_t[i])
            error_r_edema.append(error_r[i])
            error_i_edema.append(error_i[i])

         elif class_labels_test[i] == 1:
            t_glaucoma.append(t[i])
            t_glaucoma_true.append(t_t[i])
            error_r_glaucoma.append(error_r[i])
            error_i_glaucoma.append(error_i[i])

         else:
            t_normal.append(t[i])
            t_normal_true.append(t_t[i])

            error_r_normal.append(error_r[i])
            error_i_normal.append(error_i[i])


    t_t = t_t[~np.isnan(t)]
    t   = t[~np.isnan(t)]
    t   = t[~np.isnan(t_t)]
    t_t = t_t[~np.isnan(t_t)]


    t_edema_true    = np.array(t_edema_true)
    t_edema_true    = t_edema_true[~np.isnan(t_edema)]
    t_edema         = np.array(t_edema)
    t_edema         = t_edema[~np.isnan(t_edema)]
    t_edema         = t_edema[~np.isnan(t_edema_true)]
    t_edema_true    = t_edema_true[~np.isnan(t_edema_true)]



    t_glaucoma_true  = np.array(t_glaucoma_true)
    t_glaucoma_true  = t_glaucoma_true[~np.isnan(t_glaucoma)]
    t_glaucoma       = np.array(t_glaucoma)
    t_glaucoma       = t_glaucoma[~np.isnan(t_glaucoma)]
    t_glaucoma       = t_glaucoma[~np.isnan(t_glaucoma_true)]
    t_glaucoma_true  = t_glaucoma_true[~np.isnan(t_glaucoma_true)]



    t_normal_true   = np.array(t_normal_true)
    t_normal_true   = t_normal_true[~np.isnan(t_normal)]
    t_normal        = np.array(t_normal)
    t_normal        = t_normal[~np.isnan(t_normal)]
    t_normal        = t_normal[~np.isnan(t_normal_true)]
    t_normal_true   = t_normal_true[~np.isnan(t_normal_true)]


    d['mae_test'][n-1] = sklearn.metrics.mean_absolute_error(t, t_t)
    print (f"mae_test = {d['mae_test'][n-1]*resolution/im_size}")

    d['mae_test_edema'][n-1]      = sklearn.metrics.mean_absolute_error(t_edema, t_edema_true)
    print (f"mae_test_edema  = {d['mae_test_edema'][n-1]*resolution/im_size}")
    d['mae_test_glaucoma'][n-1]   = sklearn.metrics.mean_absolute_error(t_glaucoma, t_glaucoma_true)
    print (f"mae_test_glaucoma = {d['mae_test_glaucoma'][n-1]*resolution/im_size}")
    d['mae_test_normal'][n-1]     = sklearn.metrics.mean_absolute_error(t_normal, t_normal_true)
    print (f"mae_test_normal = {d['mae_test_normal'][n-1]*resolution/im_size}")


    d['thickness_pred_test'][n-1] = np.mean(t)
    d['thickness_true_test'][n-1] = np.mean(t_t)

    d['thickness_pred_test_edema'][n-1] = np.mean(t_edema)
    d['thickness_true_test_edema'][n-1] = np.mean(t_edema_true)

    d['thickness_pred_test_glaucoma'][n-1] = np.mean(t_glaucoma)
    d['thickness_true_test_glaucoma'][n-1] = np.mean(t_glaucoma_true)

    d['thickness_pred_test_normal'][n-1] = np.mean(t_normal)
    d['thickness_true_test_normal'][n-1] = np.mean(t_normal_true)


    d['error_r_folds'][n-1] = np.mean(error_r)
    d['error_i']            = error_i[~np.isnan(error_i)]
    d['error_i_folds'][n-1] = np.mean(error_i)

    d['error_r_edema_folds'][n-1] = np.mean(error_r_edema)
    d['error_i_edema ']           = np.array(error_i_edema)
    d['error_i_edema']            = error_i_edema[~np.isnan(error_i_edema)]
    d['error_i_edema_folds'][n-1] = np.mean(error_i_edema)

    d['error_r_glaucoma_folds'][n-1] = np.mean(error_r_glaucoma)
    d['error_i_glaucoma']            = np.array(error_i_glaucoma)
    d['error_i_glaucoma']            = error_i_glaucoma[~np.isnan(error_i_glaucoma)]
    d['error_i_glaucoma_folds'][n-1] = np.mean(error_i_glaucoma)

    d['error_r_normal_folds'][n-1] = np.mean(error_r_normal)
    d['error_i_normal']            = np.array(error_i_normal)
    d['error_i_normal']            = error_i_normal[~np.isnan(error_i_normal)]
    d['error_i_normal_folds'][n-1] = np.mean(error_i_normal)
    
    

########################################
#     Metrics printing
########################################
utils.printing(d, im_size, resolution)

####################### Ploting
Largest_area_dc_test_mean = np.mean(Largest_area_dc_test, axis=0)
segmented_lines = prediction_ploting.predict_two_boundaries(Largest_area_dc_test_mean, number_class, im_size)

prediction_ploting.ploting_pred_boundaries(segmented_lines,\
                                            true_lines_quick_test, x_test, name_test)


