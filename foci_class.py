#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 11:02:35 2017

@author: keriabermudez
"""

import numpy as np
import cv2
from skimage.filters import threshold_otsu
from skimage.measure import regionprops
import mahotas as mh
import cell_segmentation as cellseg
from skimage import io

"""Class to measure foci in cells """

class Foci_class: 
   
    
    def __init__(self,group_name, scale,max_foci_coverage,min_cell_area,min_foci_area,path_results):
        """
        group_name: String
            The name of the group
        scale: float 
            The pixel scale. For instance if one pixel is represents  0.099um then write  0.099
        max_foci_coverage: float 
            Maximum foci coverage. For example if max_foci_coverage = 80, don't count foci if they cover 80 percent or more.
        min_cell_area: int
            Minimun cell area in pixels. Only analyze cells that are greater than the min_cell_area in number of pixels.
        min_foci_area: int
            Minimun foci area in pixels. Only analyze foci that are greater than the min_foci_area in number of pixels.
        path_results: String
            Where you want to save the images
        """
        self.all_nuclei = {}
        self.group_name = group_name
        self.path_results =path_results
        self.foci_areas = []
        self.foci_sum_intensities = []
        self.foci_mean_intensities = []
        self.scale = scale
        self.max_foci_coverage = max_foci_coverage
        self.min_cell_area = min_cell_area
        self.min_foci_area = min_foci_area
    
    def __add_area(self, area):        
        self.foci_areas.append(area)
    
    def __add_sum_intenisity(self,sum_int):        
        self.foci_sum_intensities.append(sum_int)
        
    def __add_mean_intensity(self,mean):        
        self.foci_mean_intensities.append(mean)
        
    def __areas_funct(labeled_image):
        areas = []
        for region in regionprops(labeled_image):
            areas.append(region.area)
        areas = np.array(areas)
        return areas    
    
    def foci_results(self,labels_all,foci_intesity,name):
        """Function that analyzes the foci
        labels_all: (N, M) ndarray
            Where nuclei are labeled
        foci_intesity: (N, M) ndarray 
            Intensity image of foci
        name: String
            Name of image
        
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        foci_mask = np.zeros(labels_all.shape, dtype = np.bool)
        Foci_are_marked = np.zeros((foci_intesity.shape[0],foci_intesity.shape[1],3), dtype =np.uint8)
        Foci_are_marked[:,:,1] = foci_intesity

        for region in regionprops(labels_all,foci_intesity):
            variables = {}
            nucleus_mask = labels_all == region.label 
            if len(np.unique(foci_intesity[nucleus_mask])) == 1:
                continue
            th = threshold_otsu(foci_intesity[nucleus_mask])
            th_mask = foci_intesity > th
            
            (min_row, min_col, max_row, max_col) = region.bbox
            
            inside_nucleus = np.logical_and(th_mask,nucleus_mask)
            
            #labeled_foci,foci_number = mh.label(inside_nucleus, np.ones((3,3), bool))
            ##remove 4 pixels
            inside_nucleus = cellseg.remove_regions(inside_nucleus,self.min_foci_area)
            inside_nucleus = inside_nucleus > 0
            labeled_foci,foci_number = mh.label(inside_nucleus, np.ones((3,3), bool))
    
            #print(foci_number)
            if foci_number == 1:
                continue
            foci_areas = []
            foci_areas_scaled = []
            foci_intensities = []
            foci_mean_intensities = []
    
            for foci in regionprops(labeled_foci,foci_intesity):
                foci_areas.append(foci.area)
                foci_areas_scaled.append(foci.area*self.scale**2)
                foci_mean_intensities.append(foci.mean_intensity)
                coords = foci.coords
                intensities = 0
                for coord in coords:
                    intensities += foci_intesity[coord[0],coord[1]]
                foci_intensities.append(intensities) 
            
                self.__add_area(foci.area*self.scale**2)
                self.__add_mean_intensity(foci.mean_intensity)
                self.__add_sum_intenisity(foci_intensities)
           
            nucleus_area_scaled = float(region.area) * self.scale**2
            nuclear_mean_int = float(region.mean_intensity)
            
            foci_intensities =np.array(foci_intensities,np.float32)
        
            foci_areas_scaled = np.array(foci_areas_scaled)
            foci_areas = np.array(foci_areas, np.float32)
    
            total_foci_areas_scaled = foci_areas_scaled.sum()
            foci_areas_perc = (foci_areas_scaled.sum()/nucleus_area_scaled)*100
            total_foci_int = foci_intensities.sum()
            
            if foci_areas_perc < self.max_foci_coverage and region.area > self.min_cell_area:# 80 and 10000
                
                variables['File_Name']  = name
                variables['Group']  = self.group_name
                variables['Threshold'] = th
                variables['Nucleus_Area_px='+str(self.scale)] = nucleus_area_scaled
                variables['Nuclear_Mean_Int'] = nuclear_mean_int
                variables['Foci_Number'] = foci_number 
    
                #Total
                variables['Total_Foci_Areas_px='+str(self.scale)] = total_foci_areas_scaled
                variables['Total_Foci_Sum_Int'] = total_foci_int
                variables['Total_Foci_Mean_Int'] = total_foci_int/foci_areas.sum()
    
                variables['Foci_Areas_%'] = foci_areas_perc
                 #Arrays
                variables['Foci_Areas_Array_px='+str(self.scale)] = foci_areas_scaled
                variables['Foci_Sum_Int_Array'] = foci_intensities
                variables['Foci_Mean_Int_Array'] = foci_mean_intensities    
                self.all_nuclei[name+'_'+str(region.label)] = variables
                
                foci_mask[inside_nucleus] = True
        
                cv2.putText(Foci_are_marked,str(region.label),(min_col,min_row), font, 2,(255,255,255),2,cv2.LINE_AA)
            else:
                cv2.putText(Foci_are_marked,str(region.label),(min_col,min_row), font, 2,(0,0,255),2,cv2.LINE_AA)
    
        all_labeled_foci,all_foci_number = mh.label(foci_mask, np.ones((3,3), bool))
        im2, foci_contours, hierarchy = cv2.findContours(all_labeled_foci,cv2.RETR_FLOODFILL,cv2.CHAIN_APPROX_SIMPLE)
        im2, nuclei_contours, hierarchy = cv2.findContours(np.int32(labels_all),cv2.RETR_FLOODFILL,cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(Foci_are_marked, foci_contours, -1, (255,0,0), 1)
        cv2.drawContours(Foci_are_marked, nuclei_contours, -1, (0,255,255), 2)
        Foci_are_marked[:,:,1] = foci_intesity
    
        io.imsave(self.path_results+name+'_Marked_Foci.tif',Foci_are_marked)
        
    