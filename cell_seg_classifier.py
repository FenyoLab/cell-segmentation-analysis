#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 14:01:16 2017

@author: keriabermudez
"""
from sklearn.base import BaseEstimator, ClassifierMixin
import cell_segmentation as cellseg
from sklearn.grid_search import GridSearchCV
from skimage import io
import numpy as np


class CellClassifier(BaseEstimator, ClassifierMixin):  
    """An example of classifier"""

    def __init__(self,shape = (2114, 1639),kernel_1=7,kernel_2=3):
        """
        Called when initializing the classifier
        """
        self.kernel_1 = kernel_1
        self.kernel_2 = kernel_2
        self.shape = shape
        self._y_train = 0

    def calc_label(self, X):
        new_X = X.copy()
        image = new_X.reshape(self.shape)
        cl1, gaussian_blur_cl1, segmented_zlevel, centers = cellseg.enhance_blur_segment(image,enhance = False, blur = False, kernel = self.kernel_1, n_intensities = 2)
        labeled = cellseg.watershedsegment(segmented_zlevel,smooth_distance = True,kernel =self.kernel_2) ##increase kernel size if it is oversegmenting
        y_2d  = cellseg.draw_contours(labeled,segmented_zlevel, with_labels = False, color = 0,width = 1 )
        y_2d[y_2d == 255] = 1
        y = y_2d.flatten()
        return y

    def fit(self, X, y):       
        self._y_train = y
        return self

    
    def predict(self, X, y=None):
        y = self.calc_label(X)

        return y
    
    def score(self,X,true_y):
        y_predict = self.predict(X)
        np.sum(y_predict == self._y_train)/float(len(self._y_train))
        

# %%
#Reading Image Stack


path= '/Users/keriabermudez/Dropbox/Projects/Gregory_Brittingham/ERK_KTR/'

image_dapi = 'ERK_DAPI2 (Converted).mov'
image_erk = 'ERK_KTR1 (Converted).mov'

path_results = path +'Results/'


#%%
#crop
dapi = cellseg.video_to_tif(path+image_dapi,0)
erk = cellseg.video_to_tif(path+image_erk,1)
dapi = dapi[:,500:,500:]
erk = erk[:,500:,500:]


# %% 
# Creating ztsack of intesitiy image and color zstack

dapi_color = np.zeros((dapi.shape[0], dapi.shape[1], dapi.shape[2],3), dtype = np.uint8)
dapi_color[:,:,:,2 ] = dapi.copy()

erk_color = np.zeros((erk.shape[0], erk.shape[1], erk.shape[2],3), dtype = np.uint8)
erk_color[:,:,:,1 ] = erk.copy()
#%%
# 2) Try paremeters in one slice or zlevel

z  = 0
zlevel_image_dapi = dapi[z].copy()
zlevel_image_erk = erk[z].copy()

zlevel_color_erk = erk_color[z].copy()
zlevel_color_dapi = dapi_color[z].copy()

#%%
# Enhance, Blur, and Segment 

cl1, gaussian_blur_cl1, segmented_zlevel, centers = cellseg.enhance_blur_segment(zlevel_image_dapi,enhance = False, blur = True, kernel = 7, n_intensities = 2)
io.imshow(cl1)
io.imshow(gaussian_blur_cl1)
io.imshow(segmented_zlevel)
centers

# Test Region Detection parameters

labeled = cellseg.watershedsegment(segmented_zlevel,smooth_distance = True,kernel = 3) ##increase kernel size if it is oversegmenting


#%% Draw Contours  

zlevel_image_color_regions_d  = cellseg.draw_contours(labeled,segmented_zlevel, with_labels = False, color = 0,width = 2 )
x_train = zlevel_image_dapi.flatten()
y_2d_train  = cellseg.draw_contours(labeled,segmented_zlevel, with_labels = False, color = 0,width = 1 )
y_2d_train[y_2d_train == 255] = 1
y_train = y_2d_train.flatten()


tuned_params = {"kernel_1" : [7,11,21,31,41,51],"kernel_2" : [7,11,21,31,41,51]}

gs = GridSearchCV(CellClassifier(), tuned_params)
gs.fit(x_train, y_train)
gs.best_params_