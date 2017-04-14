#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 05:47:04 2017

@author: keriabermudez
"""# -*- coding: utf-8 -*-



import cell_segmentation as cellseg
from skimage import io
import numpy as np
import os
import re


path = '/Users/keriabermudez/Dropbox/Projects/Gregs/VIDEOS/Videos_July1/Part A/'
path_results = '/Users/keriabermudez/Dropbox/Projects/Gregs/VIDEOS/Videos_July1/Part_A_Results_April_2017/'


listfiles = []

for list_file in os.listdir(path):
    if list_file.endswith(".tif"):
        listfiles.append(list_file)
        
for n_file,n_file_image in enumerate(listfiles):
           
     file_exists = os.path.isfile(path_results+n_file_image[0:-4]+'.csv') 
     if file_exists:
    
         continue
     
     if re.search('MLN',n_file_image):
         #Reading Image
         green  = io.imread(path+n_file_image)
    
         #Creating ztsack of intenistiy image and color zstack
         green_color = np.zeros((green.shape[0], green.shape[1], green.shape[2],3), dtype = np.uint8)
         green_color[:,:,:,1 ] = green.copy()
         
        # 2) Cell Tracking With Blob Detection, Note: This can take a long time to run

         ct = cellseg.cell_tracking(green,green,green_color)

        #Setting Segmentation Parameters
         ct.set_segment_param(enhance = False, blur = True, kernel = 31, n_intensities = 2)
        
        #Setting Blob Parameters
         ct.set_blob_param(max_sigma=50,min_sigma=40,num_sigma=5,threshold=.01,overlap=0.4)
        
        #Track with Blob
        
         ct.track_with_blob()
         ct.draw_trajectories()
        
         #Save zstack color as a video
         ct.create_video(path_results, n_file_image[0:-4]+'_Blob_Detection', fps = 2)
         io.imsave(path_results+n_file_image[0:-4]+'_Blob_Detection.tif',ct.zstack_color)
         
         table_postions = ct.positions_table
         table_postions.to_csv(path_results+ n_file_image[0:-4]+'.csv')
         print('finished file ' + n_file_image)
         
