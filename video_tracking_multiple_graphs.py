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
listfiles = []

tables = []

for list_file in os.listdir(path_results):
    if list_file.endswith(".csv"):
        listfiles.append(list_file)
        file_name   =    list_file 
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
        tables.append(ct.positions_table )
        kws = dict(s=10)
        g = sns.FacetGrid(ct.positions_table, col="label", col_wrap=5)
        g.map(plt.scatter, "z", "mean_intensity", alpha=0.5,**kws)
        g.savefig(path_graph+name+'labels.pdf')

        g = sns.lmplot(x="z", y="mean_intensity", hue="label", data=ct.positions_table,fit_reg = False, size=5, legend = False);
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title(name)
        g.savefig(path_graph+name+'hue.pdf')

all_tables  = pd.concat(tables) 
all_tables.to_csv(path_graph+'Summary_Results.csv')       
#g= sns.lmplot(x="z", y="mean_intensity", col="label", data=ct.positions_table,fit_reg = False, size=5, legend = False,col_wrap=4);
#g.savefig(path_results+file_name[0:-4]+'grid.pdf')

#%%
all_tables = all_tables.sort_values(by ='group')
kws = dict(s=10)
g = sns.FacetGrid(all_tables, col="group",hue ='label')
g.map(plt.scatter, "z", "mean_intensity", alpha=0.5,**kws)
g.savefig(path_graph+'All_Results_By_Group.pdf')

#%%
kws = dict(s=10)
g = sns.FacetGrid(all_tables, col="image" , col_wrap=5)
g.map(plt.scatter, "z", "mean_intensity", alpha=0.5,**kws)
g.add_legend();
g.savefig(path_graph+'All_Results_By_Stack.pdf')

