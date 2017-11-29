# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 13:34:56 2017

@author: sarahkeegan
"""

import cell_segmentation as cellseg
import io
import base64
from IPython.display import HTML
from skimage import io as sio
import numpy as np
import os
import matplotlib.pyplot as plt
import ast
import pandas as pd
import seaborn as sns

path= '/Users/sarahkeegan/fenyolab/code/cell-segmentation-analysis/images/tracking_example/'

image = 'GFP-CCND1.tif'
path_results = path +'Results/'

green  = sio.imread(path+image)
green_color = np.zeros((green.shape[0], green.shape[1], green.shape[2],3), dtype = np.uint8)
green_color[:,:,:,1 ] = green.copy()

#ct = cellseg.cell_tracking(green,green,green_color)
#ct.set_segment_param(enhance = False, blur = True, kernel = 61, n_intensities = 2)
#ct.set_blob_param(max_sigma=50,min_sigma=40,num_sigma=5,threshold=.01,overlap=0.4)
#ct.track_with_blob()
##ct.draw_trajectories()
##ct.create_video(path_results, 'Blob_Detection', fps = 2)
##table_positions  = ct.positions_table
#
#ct.positions_table.to_csv(path_results+image[:-4]+'_Positions_Table.csv')

###
track_window =  (680,370,137,175)
z=0

ct2 = cellseg.cell_tracking(green,green,green_color)
ct2.set_watershed_param(smooth_distance = True, kernel = 31)

#Track cells by returning a zstack with the marked cell and the intensity measurements
tracked_label, measurements = ct2.track_window(z,track_window,enhance_bool = True,kernel_size = 61)
stack_graph = ct2.track_window_graph(1,z,track_window,enhance_bool = True, blur_bool = True, kernel_size = 61)

#Save Zstack and video
sio.imsave(path_results+ 'Tracked_window.tif',stack_graph)
#cellseg.create_video(path_results, 'Tracked_window',stack_graph)