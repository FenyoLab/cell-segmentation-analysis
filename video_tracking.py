
""""
# Created on Tue Mar 21 15:46:39 2017

# @author: keriabermudez

# This script is an example of how to use the cell tracking 

"""

import cell_segmentation as cellseg
from skimage import io
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
import ast
# %%
#Reading Image Stack

path = os.getcwd()

path= path+'/images/tracking_example/'

image = 'GFP-CCND1.tif'
path_results = path +'Results/'

green  = io.imread(path+image)
green.shape
green = green[0:100]

# %% 
# Creating ztsack of intesitiy image and color zstack

green_color = np.zeros((green.shape[0], green.shape[1], green.shape[2],3), dtype = np.uint8)
green_color[:,:,:,1 ] = green.copy()

#%%
# 2) Try paremeters in one slice or zlevel

z  = 0
zlevel_image = green[z].copy()
zlevel_image_color = green_color[z].copy()

#%%
# Enhance, Blur, and Segment 

cl1, gaussian_blur_cl1, segmented_zlevel, centers = cellseg.enhance_blur_segment(zlevel_image,enhance = False, blur = True, kernel = 61, n_intensities = 2)
io.imshow(cl1)
io.imshow(gaussian_blur_cl1)
io.imshow(segmented_zlevel)
centers
    
#%%
# Test Region Detection parameters

labeled = cellseg.watershedsegment(segmented_zlevel,smooth_distance = True,kernel = 31) ##increase kernel size if it is oversegmenting

#%% Draw Contours 


zlevel_image_color_regions  = cellseg.draw_contours(labeled,zlevel_image_color, with_labels = False, color = (255,0,0),width = 3 )
io.imshow(zlevel_image_color_regions)
io.imsave(path_results+'Z-level_Regions'+str(z)+'.tif',zlevel_image_color_regions)

# Get Measurements
positions_regions = cellseg.regions_measure(labeled)

#%%
# Test Blob Detection parameters


zlevel_image_color_marked =  cellseg.draw_blob_log(gaussian_blur_cl1,zlevel_image_color,  with_labels = True,max_sigma=50,  min_sigma=40,num_sigma=5,threshold=.01,overlap=0.4,color_blobs = (0,0,255),width =3)
io.imshow(zlevel_image_color_marked)



 #Get Measurements
positions_blobs = cellseg.blob_log_measure(gaussian_blur_cl1, max_sigma=50,  min_sigma=40,num_sigma=5,threshold=.01,overlap=0.4 )


#%%
# 2) Cell Tracking With Blob Detection, Note: This can take a long time to run

ct = cellseg.cell_tracking(green,green,green_color)

# Setting Segmentation Parameters
ct.set_segment_param(enhance = False, blur = True, kernel = 61, n_intensities = 2)

# Setting Blob Parameters
ct.set_blob_param(max_sigma=50,min_sigma=40,num_sigma=5,threshold=.01,overlap=0.4)

# Track with Blob
ct.track_with_blob()
ct.draw_trajectories()

# Save zstack color as a video
ct.create_video(path_results, 'Blob_Detection', fps = 3)
table_positions  = ct.positions_table
table_positions.to_csv(path_results+image[:-4]+'Positions_Table.csv')

#%%
# To look at a cell and how the insensity changes with time. First, select the label, then track, and create video

positions_table = pd.read_csv(path_results+image[:-4]+'_Positions_Table.csv', index_col = 0,converters={"track_window": ast.literal_eval})
ct = cellseg.cell_tracking(green,green,green_color)
ct.positions_table = positions_table
label = 11
ct.reset_drawing()

mpl.rcParams.update(mpl.rcParamsDefault)
stack_graph = ct.track_blob_labeled(label,size =3.0)
cellseg.create_video(path_results, 'Tracked_blob_' + str(label),stack_graph)

#%%
# 3) Cell Tracking With Tracking Window
z = 0
track_window =  (680,370,137,175)

ct2 = cellseg.cell_tracking(green,green,green_color)
ct2.set_watershed_param(smooth_distance = True, kernel = 31)

#Track cells by returning a zstack with the marked cell and the intensity measurements
tracked_label, measurements = ct2.track_window(z,track_window,enhance_bool = True,kernel_size = 61)
stack_graph = ct2.track_window_graph(1,z,track_window,enhance_bool = True, blur_bool = True, kernel_size = 61)

#Save Zstack and video
io.imsave(path_results+ 'Tracked_window.tif',stack_graph)
cellseg.create_video(path_results, 'Tracked_window',stack_graph)

