
""""
# Created on Tue Mar 21 15:46:39 2017

# @author: keriabermudez

# This script is an example of how to use the cell tracking 

"""
import cell_segmentation as cellseg
from skimage import io
import numpy as np


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

cl1, gaussian_blur_cl1, segmented_zlevel, centers = cellseg.enhance_blur_segment(zlevel_image_dapi,enhance = True, blur = True, kernel = 7, n_intensities = 2)
io.imshow(cl1)
io.imshow(gaussian_blur_cl1)
io.imshow(segmented_zlevel)
centers
    
#%%
# Test Region Detection parameters

labeled = cellseg.watershedsegment(segmented_zlevel,smooth_distance = True,kernel = 3) ##increase kernel size if it is oversegmenting


#%% Draw Contours  

zlevel_image_color_regions_d  = cellseg.draw_contours(labeled,segmented_zlevel, with_labels = False, color = 0,width = 2 )

zlevel_image_color_regions_d  = cellseg.draw_contours(labeled,zlevel_color_dapi.copy(), with_labels = True, color = (255,0,0),width = 2 )
zlevel_image_color_regions_e  = cellseg.draw_contours(labeled,zlevel_color_erk.copy(), with_labels = True, color = (255,0,0),width = 2 )

io.imshow(zlevel_image_color_regions_d)
io.imshow(zlevel_image_color_regions_e)

io.imsave(path_results+'Z-level_Regions_'+str(z)+'dapi.tif',zlevel_image_color_regions_d)
io.imsave(path_results+'Z-level_Regions_'+str(z)+'erk.tif',zlevel_image_color_regions_e)

# Get Measurements
positions_regions = cellseg.regions_measure(labeled,zlevel_image_erk)

#%%
# Test Blob Detection parameters

zlevel_image_color_marked =  cellseg.draw_blob_log(cl1,zlevel_color_dapi.copy(),  with_labels = True,max_sigma=30,  min_sigma=20,num_sigma=10,threshold=.01,overlap=0.6,color_blobs = (255,0,0),width =3)
io.imshow(zlevel_image_color_marked)
io.imsave(path_results+'Z-level_Blobs_'+str(z)+'.tif',zlevel_image_color_marked)

 #Get Measurements
positions_blobs = cellseg.blob_log_measure(cl1,zlevel_image_erk, max_sigma=30,  min_sigma=20,num_sigma=10,threshold=.01,overlap=0.6 )


#%%
# 2) Cell Tracking With Blob Detection, Note: This can take a long time to run

ct = cellseg.cell_tracking(erk,dapi,erk_color)

# Setting Segmentation Parameters
ct.set_segment_param(enhance = False, blur = False, n_intensities = 2)

# Setting Blob Parameters
ct.set_blob_param(max_sigma=30,min_sigma=20,num_sigma=10,threshold=.01,overlap=0.6)

# Track with Blob
ct.track_with_blob(min_slices =1,color_blobs= (255,0,0))
ct.draw_trajectories(color_trajectory= (255,255,0) )

table_positions  = ct.positions_table
io.imsave(path_results+'Blobs.tif',ct.zstack_color)

#table_positions.to_csv(path_results+image[:-4]+'Positions_Table.csv')
#%%
# 3) Cell Tracking With Regions

ct2 = cellseg.cell_tracking(erk,dapi,dapi_color.copy())

# Setting Segmentation Parameters
ct2.set_segment_param(enhance = False, blur = False, n_intensities = 2)
ct2.set_watershed_param(smooth_distance = True, kernel = 3)

# Track with Blob
ct2.track_with_regions(min_slices =1, color_contours= (255,0,0))
ct2.draw_trajectories(color_trajectory= (255,255,0) )


#table_positions.to_csv(path_results+image[:-4]+'Positions_Table.csv')
io.imsave(path_results+'Regions.tif',ct2.zstack_color)


