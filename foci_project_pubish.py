# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 16:11:03 2016

@author: keriabermudez
"""

from skimage import  io
import pandas as pd
from pandas import DataFrame
import os
import cell_segmentation as cellseg
import foci_class   as foci  
import cv2


#%%
"""In this example the nuclei are labeled with DAPI in separate images with 'c2' at the end of the name and Foci are in images 
with c1 at the end of the name. Also there are two groups: No Dox, and Dox. No Dox Images have a 0  and Dox have a 1"""

path = os.getcwd()

path= path+'/images/foci_example/'

scale = 0.099
nucleus_files = []


for files in os.listdir(path):
    if files.endswith("c2.tif"):
        nucleus_files.append(files)
#Create Class
NO_DOX = foci.Foci_class('NO_DOX',scale, 80, 1000,4,path)
DOX = foci.Foci_class('DOX',scale, 80, 1000, 4 ,path)

for file_name in nucleus_files:
    
    nucleus_img = path + file_name   
    name = file_name[:-4]
    group = file_name[0]
    
    #Segment the nuclei
    nuclei_img = cv2.imread(nucleus_img)
    nuclei_img = cv2.cvtColor(nuclei_img, cv2.COLOR_BGR2GRAY)
    enhanced, gaussian_blur_cl1, segmented, thresholds = cellseg.enhance_blur_segment(nuclei_img)
    
    #Remove small regions
    segmented = cellseg.remove_regions(segmented,150)
    
    #Label nuclei
    labels_all =  cellseg.watershedsegment(segmented,smooth_distance = True,kernel = 16)
    
    foci_img_path = path +file_name[:-6]+'c1.tif'
    foci_img = io.imread(foci_img_path)
    foci_intesity = foci_img[:,:,1]
    name = os.path.basename(foci_img_path)[:-4] 
    if group  == '0':
        NO_DOX.foci_results(labels_all,foci_intesity, name) 
    if group  == '1':
        DOX.foci_results(labels_all,foci_intesity, name)
  
    
table_DOX = DataFrame(DOX.all_nuclei)       
table_DOX = table_DOX.T

table_NO_DOX = DataFrame(NO_DOX.all_nuclei)       
table_NO_DOX = table_NO_DOX.T


#%%
table =pd.concat([table_DOX,table_NO_DOX])
table.to_csv(path+'Results_Dox_No_Dox.csv')

#    
  