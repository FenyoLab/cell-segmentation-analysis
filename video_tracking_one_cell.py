#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 17:02:38 2017

@author: keriabermudez
"""

import cell_segmentation as cellseg
from skimage import io
import numpy as np
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import ast

mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['axes.linewidth'] = 0.75
mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['font.size'] = 12
mpl.rcParams['pdf.fonttype'] = 42
sns.set_context("talk")
sns.set_style("whitegrid")

#%%
       
path = '/Users/keriabermudez/Dropbox/Projects/Gregs/VIDEOS/Videos_July1/Part A/'
path_results = '/Users/keriabermudez/Dropbox/Projects/Gregs/VIDEOS/Videos_July1/Part_A_Results_April_2017/'
path_graph = '/Users/keriabermudez/Dropbox/Projects/Gregs/VIDEOS/Videos_July1/Part_A_Results_April_2017/Graphs/'


#%%
name_to_search = 'LMB_3'
label_n = 10

mpl.rcParams.update(mpl.rcParamsDefault)

for list_file in os.listdir(path_results):
    if list_file.endswith(".csv"):
        if list_file.find(name_to_search) != -1 :
            file_name = list_file
        
           
            table = pd.read_csv(path_results+file_name, index_col = 0,converters={"track_window": ast.literal_eval})
            #file_name   =    listfiles[0]   
            
            name = file_name[46::]
            name = name[0:-4]
            if name.find('GSK+MLN') != -1:
                group = 'GSK+MLN'
            elif name.find('GSK') != -1 and name.find('GSK+MLN') == -1:
                group = 'GSK'
            elif name.find('LMB+MLN')!= -1:
                group = 'LMB+MLN'
            elif name.find('LMB')!= -1 and name.find('LMB+MLN') == -1:
                group = 'LMB'
            elif name.find('MLN')!= -1 and name.find('LMB+MLN') == -1 and name.find('GSK+MLN') == -1:
                group = 'MLN'  
            elif name.find('NT')!= -1:
                group ='NT'
            else:
                'Print No Group'
                    
            table['image'] = name
            table['group'] = group
            #Reading Image
            green  = io.imread(path +file_name[0:-4]+'.tif')
            
            #Creating ztsack of intesity image and color zstack
            green_color = np.zeros((green.shape[0], green.shape[1], green.shape[2],3), dtype = np.uint8)
            green_color[:,:,:,1 ] = green.copy()
            
            ct = cellseg.cell_tracking(green,green,green_color)
            ct.positions_table = table.copy()
            ct.track_with_blob(75)
            ct.set_segment_param(enhance = False, blur = True, kernel = 31, n_intensities = 2)
            #Setting Segmentation Parameters
            #fig, ax1  = plt.subplots(nrows=1,ncols=1, figsize=(2.5,2.5))
            g = sns.lmplot(x="z", y="mean_intensity", hue="label", data=ct.positions_table,fit_reg = False, size=5, legend = False);
               
            
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.title(name)
            ct.reset_drawing()
            labeled = ct.track_blob_labeled(label_n,levels_after =4)
            cellseg.create_video(path_graph, name+'Label_'+str(label_n),labeled,fps=4)
            #
#%% Modeling


label=   ct.positions_table[ct.positions_table.label == label_n]

x = label['z'].values
y =label['mean_intensity'].values
z = np.polyfit(x, y, 4)
p = np.poly1d(z)
xp = np.linspace(-2,170, 200)
plt.plot(x, y, '.', xp, p(xp), '-')
