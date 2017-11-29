# -*- coding: utf-8 -*-

"""
Created on Fri Dec 18 10:09:41 2015

@author: keriabermudez
"""

import cv2
import numpy as np
from skimage import feature,segmentation,filters
from scipy import ndimage 
from scipy.spatial import distance
import mahotas as mh
from skimage.measure import regionprops
from sklearn.neighbors import NearestNeighbors
from pandas import DataFrame 
from skimage.feature import blob_dog, blob_log
from scipy.spatial import distance
from math import sqrt
from skimage.filters import threshold_otsu
from pandas import Series
import matplotlib as mpl
import matplotlib.pyplot as plt
import io
import base64
from IPython.display import HTML
from skimage import img_as_ubyte,color,exposure
from skimage import io as sio

"""Series of functions to segment cells or nuclei"""

font = cv2.FONT_HERSHEY_SIMPLEX  

def remove_regions(segmented_image, area, size='smaller'):
        
    """Removes regions that are smaller than the area, or greater and equal to the area
        ----------
        segmented_image:(N, M) ndarray
            Segmented image or labled image
        area: int
            Area of cuttoff
        
        Returns
        -------
        segmented_image: (N, M) ndarray
            
        """
    if len(np.unique(segmented_image)) >2:
        labeled_image = segmented_image
    else:
        labeled_image, num_objects= mh.label(segmented_image, np.ones((3,3), bool))
    if size == 'smaller': 
        for region in regionprops(labeled_image):
            if region.area < area: #less than mean area - std or greater than
                for coord in region.coords:
                    segmented_image[coord[0],coord[1]] = 0
        return segmented_image
    else:
        for region in regionprops(labeled_image):
            if region.area >= area: #less than mean area - std or greater than
                for coord in region.coords:
                    segmented_image[coord[0],coord[1]] = 0
        return segmented_image
        
def kmeans_img(cl1, K):
     
    """Uses the kmeans clustering algorithm to separate the images in K intensities
        ----------
        cl1:(N, M) ndarray
            Intensity image
        K: int
           number of K intensities you wan to divide  the image
        
        Returns
        -------
        segmented_image: (N, M) ndarray
            
        """
    Z = cl1.reshape((-1))
    
    # convert to np.float32
    Z = np.float32(Z)
    
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
   
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((cl1.shape))
    center  = np.sort(center, axis = 0)
    center = center[::-1]
    return res2, center

def watershedsegment(thresh,smooth_distance = True,kernel = 3,min_dist=10):
    
    """Uses the watershed segmentation  algorithm to separate cells or nuclei in K intensities
        ----------
        thresh:(N, M) ndarray
            Segmented image
        
        smooth_distance: bool
            If distances will be smoothed
        
        kernel: int 
            Kernel size for smoothing the distances map
        
        Returns
        -------
        labeled_image: (N, M) ndarray
            The label image will have each cell or nuclei labled with a number
        """
    distances = mh.distance(thresh)
   
    if smooth_distance:
        distance = ndimage.gaussian_filter(distances, kernel)
    else:
        distance = distances
    
    
    maxima = feature.peak_local_max(distance, indices=False, exclude_border=False, min_distance=min_dist)
    surface = distance.max() - distance
    spots, t = mh.label(maxima) 
    areas, lines = mh.cwatershed(surface, spots, return_lines=True)
    
    labeled_clusters, num_clusters= mh.label(thresh, np.ones((3,3), bool))
    joined_labels = segmentation.join_segmentations(areas, labeled_clusters)
    labeled_nucl = joined_labels * thresh

    for index, intensity in enumerate(np.unique(labeled_nucl)):
            labeled_nucl[labeled_nucl == intensity] = index   
    return labeled_nucl

def enhance_blur(img, enhance = True, blur = True, kernel = 61):
    
    """Uses the watershed segmentation  algorithm to sperate cells or nuclei in K intensities
        ----------
        img: (N, M) ndarray
             RGB image
        
        enhance: bool
            If the image will be enhanced using the CLAHE (Contrast Limited Adaptive Histogram Equalization) from OpenCV
        
        blur: bool 
            If the image will be blurred using the gaussian blur
        
        kernel: int 
           Kernel size of the gaussian blur
           
        n_intensities: int
            Number of intensities you want to segment. If you want to segment into two intensities Otsu is used
            If the intensities are greater than 2, then kmeans_img function is used to segment
        
        Returns
        -------
        cl1: (N, M) ndarray
            enhanced image
        
        gaussian_blur_cl1:  (N, M) ndarray
            blurred image
        
        segmented: (N, M) ndarray
            segmented images
            
        centers: int or list of int
            Thresholds or centers for the kmeans  
     """
    
    if enhance:
        clahe = cv2.createCLAHE() # tileGridSize=(8,8)
        cl1 = clahe.apply(img)
    else:
        cl1 = img.copy()
    
    if blur:
        gaussian_blur_cl1 = cv2.GaussianBlur(cl1,(kernel,kernel),0, 0)        
    else:
        gaussian_blur_cl1 = cl1.copy()
        
    return gaussian_blur_cl1
    
def enhance_blur_segment(img,enhance = True, blur = True, kernel = 61, n_intensities = 2):
    
    """Uses the watershed segmentation algorithm to sperate cells or nuclei in K intensities
        ----------
        img: (N, M) ndarray
             RGB image
        
        enhance: bool
            If the image will be enhanced using the CLAHE (Contrast Limited Adaptive Histogram Equalization) from OpenCV
        
        blur: bool 
            If the image will be blurred using the gaussian blur
        
        kernel: int 
           Kernel size of the gaussian blur
           
        n_intensities: int
            Number of intensities you want to segment. If you want to segment into two imatensities Otsu is used
            If the intensities are greater than 2, then kmeans_img function is used to segement
        
        Returns
        -------
        cl1: (N, M) ndarray
            enhanced image
        
        gaussian_blur_cl1:  (N, M) ndarray
            blurred image
        
        segmented: (N, M) ndarray
            segmented images
            
        centers: int or list of int
            Thresholds or centers for the kmeans  
     """
    
    if enhance:
        clahe = cv2.createCLAHE() # tileGridSize=(8,8)
        cl1 = clahe.apply(img)
    else:
        cl1 = img.copy()
    
    if blur:
        gaussian_blur_cl1 = cv2.GaussianBlur(cl1,(kernel,kernel),0, 0)        
    else:
        gaussian_blur_cl1 = cl1.copy()
    
    if n_intensities  == 2:
        if(gaussian_blur_cl1.dtype == 'uint8'):
            centers, segmented = cv2.threshold(gaussian_blur_cl1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        else:
            centers = filters.threshold_otsu(gaussian_blur_cl1)
            segmented = gaussian_blur_cl1 > centers
    elif n_intensities > 2:
        segmented, centers =  kmeans_img(gaussian_blur_cl1, n_intensities)
    else:
        print 'intensity colors neeed to be 2 or greater'
        
    return (cl1, gaussian_blur_cl1, segmented, centers)


def draw_blob_log(intensity_image,color_image,with_labels = False, max_sigma=8,  min_sigma=4,num_sigma=30,threshold=.02,overlap=0.7,color_blobs = (0,0,255),width =1 ):
    """ Identifies blobs in intensity image using the scikit image blob_log function, and then draws blobs in color image
        -------
        intensity_image: (N, M) ndarray
            Image where you want to detect the blobs
        
        color_image: (N, M, 3) ndarray
            RGB image were you want to draw the blobs
        
        min_sigma: float, optional
            The minimum standard deviation for Gaussian Kernel. Keep this low to detect smaller blobs.
        
        max_sigma: float, optional
            The maximum standard deviation for Gaussian Kernel. Keep this high to detect larger blobs.
        
        num_sigma: int, optional
            The number of intermediate values of standard deviations to consider between min_sigma and max_sigma.
        
        color_blobs: tuple, optional
            Color of the blobs. Example: (0,255,0)
        
        width: int, optional
            Width of line of blobs
        
        Returns
        -------
        color_image: (N, M,3) ndarray
            RGB image with blobs drawn                                                   
      
    """
    blobs = blob_log(intensity_image, max_sigma,  min_sigma,num_sigma,threshold,overlap)
    try:
        blobs[:, 2] = blobs[:, 2] * sqrt(2)
        n_label = 0
        for blob in blobs:
            y, x, r = blob
            cv2.circle(color_image, (int(x),int(y)), int(r),color_blobs , width)
            if with_labels:
                cv2.putText(color_image,str(n_label),(int(x),int(y)),font,1,color_blobs,width,cv2.LINE_AA)
            n_label += 1
    except:
        print(len(blobs))
    return color_image

def blob_log_measure(track_image,intensity_image, max_sigma=8,  min_sigma=4,num_sigma=30,threshold=.02,overlap=0.7):
    """ Identifies blobs in intensity image using the scikit image blob_log function, and then outputs a dataframe table with the positions of the blobs
        -------
        track_image: (N, M) ndarray
            Image where you want to detect the blobs
        intensity_image: (N, M) ndarray
            Image where you want to measure the intesnity of the blobs    
        min_sigma: float, optional
            The minimum standard deviation for Gaussian Kernel. Keep this low to detect smaller blobs.
        
        max_sigma: float, optional
            The maximum standard deviation for Gaussian Kernel. Keep this high to detect larger blobs.
        
        num_sigma: int, optional
            The number of intermediate values of standard deviations to consider between min_sigma and max_sigma.
        
        Returns
        -------
        positions: pandas Dataframe
            Table with the center coordinates and labels for the blobs. The columns are the following: ['x_col','y_row','r','track_window']                                             
      
    """
    positions = []
    blobs = blob_log(track_image, max_sigma,  min_sigma,num_sigma,threshold,overlap)
    if len(blobs) > 0 :
        blobs[:, 2] = blobs[:, 2] * sqrt(2)
        for blob in blobs:
            y_row, x_col, r = blob
            position = []
            position.append(x_col)
            position.append(y_row)
            position.append(r)
            x0 = x_col - r
            y0 = y_row - r
            if x0 < 0:
                x0 = 0
            if y0 < 0:
                y0 = 0
    
            track_window = (int(x0), int(y0), int(r*2), int(r*2))
            position.append(track_window)
            mask = np.zeros_like(intensity_image)
                      
            cv2.circle(mask, (int(x_col),int(y_row)), int(r),255 , -1)
            mean_intensity = intensity_image[mask > 0].mean()
            position.append(mean_intensity)
            positions.append(position)
           
        positions  = DataFrame(positions, columns = ['x_col','y_row','r','track_window','mean_intensity'])
            
        
        return positions
    else:
        return []

def draw_blob_dog(intensity_image,color_image,with_labels = False, max_sigma=8,  min_sigma=4,num_sigma=30,threshold=.02,overlap=0.7,color_blobs = (0,0,255),width =1 ):
    """ Identifies blobs in intensity image using the scikit image blob_dog function, and then draws blobs in color image
        -------
        intensity_image: (N, M) ndarray
            Image where you want to detect the blobs
        
        color_image: (N, M, 3) ndarray
            RGB image were you want to draw the blobs
        
        min_sigma: float, optional
            The minimum standard deviation for Gaussian Kernel. Keep this low to detect smaller blobs.
        
        max_sigma: float, optional
            The maximum standard deviation for Gaussian Kernel. Keep this high to detect larger blobs.
        
        num_sigma: int, optional
            The number of intermediate values of standard deviations to consider between min_sigma and max_sigma.
        
        color_blobs: tuple
            Color of the blobs. Example: (0,255,0)
        
        width: int
            Width of line of blobs
        
        Returns
        -------
        color_image: (N, M,3) ndarray
            RGB image with blobs drawn
    """
    blobs = blob_dog(intensity_image, max_sigma,  min_sigma,num_sigma,threshold,overlap)
    blobs[:, 2] = blobs[:, 2] * sqrt(2)
    n_label = 0
    for blob in blobs:
        y, x, r = blob
        cv2.circle(color_image, (int(x),int(y)), int(r),color_blobs , width)
        if with_labels:
            cv2.putText(color_image,str(n_label),(int(x),int(y)),font,1,color_blobs,width,cv2.LINE_AA)
        n_label += 1
    
    return color_image

def draw_contours(labeled,color_image,with_labels= False, color = (255,0,0),width = 1 ):
    """Draws contours based on labeled image
       -------
       
        color_blobs: tuple
            Color of the blobs. Example: (0,255,0)
        
        width: int, optional
            Width of line of blobs
       Returns
       -------
       color_image: (N, M,3) ndarray
           RGB image with blobs drawn
    """
    
    im2, contours, hierarchy = cv2.findContours(np.int32(labeled),cv2.RETR_FLOODFILL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(color_image, contours, -1, color, width)
    
    if with_labels:
        for region in regionprops(labeled):
            (min_row, min_col, max_row, max_col) = region.bbox
            label = int(region.label)-1
            cv2.putText(color_image,str(region.label),(int(min_col),int(min_row)),font,1,255,width,cv2.LINE_AA)

    return color_image

def regions_measure(labeled,intensity_image):
    
    """Creates a DataFrame of the positions and measurements of the regions based on a labeled image"""
    
    positions =[]
    for region in regionprops(labeled,intensity_image):
        position = []
        y0, x0, y1, x1 = region.bbox #(min_row, min_col, max_row, max_col)
        r = (x1-x0)/2.0
        y_row = region.centroid[0]
        x_col = region.centroid[1]
        position.append(x_col)
        position.append(y_row)
        position.append(r)
        
        track_window = (x0, y0, x1-x0, y1-y0)
    
        position.append(track_window)
        position.append(region.mean_intensity)

        positions.append(position)
    
    positions  = DataFrame(positions, columns = ['x_col','y_row','r','track_window','mean_intensity'])
    
    return positions

def create_video (path, name, zstack_color,fps = 2):
    
    """Creates an mp4v video from zstack and saves in the path"""
    
    fourcc2 = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') # note the lower case
    vout2 = cv2.VideoWriter(path+name+'.mov',fourcc2,fps,zstack_color.shape[1:3],True)
  
    
    zstack_color_bgr = np.zeros_like(zstack_color) 
    
    zstack_color_bgr[:,:,:,0] = zstack_color[:,:,:,2].copy()
    zstack_color_bgr[:,:,:,1] = zstack_color[:,:,:,1].copy()
    zstack_color_bgr[:,:,:,2] = zstack_color[:,:,:,0].copy()
    
    success = vout2.open(path+name+'.mov',fourcc2,fps,zstack_color_bgr.shape[1:3],True)
    #vout2 = cv2.VideoWriter(path+name+'.mov',fourcc2,fps,zstack_color_bgr.shape[1:3],True)
    #print success
    for frame in list(range(zstack_color_bgr.shape[0])):
        new_frame = zstack_color_bgr[frame]
        vout2.write(new_frame)
            
    vout2.release() 
    
def jupyter_video(fname):
    
    video = io.open(fname, 'r+b').read()
    encoded = base64.b64encode(video)
    HTML(data='''<video alt="test" controls>
                    <source src="data:video/mp4;base64,{0}" type="video/mp4" />
                 </video>'''.format(encoded.decode('ascii'))) 

def video_to_tif(path,index):
    """Reads video and retuns tif
    """
    cap = cv2.VideoCapture(path)
    rows = int(cap.get(4))
    columns = int(cap.get(3))
    frames = int(cap.get(7))
    
    tif_image = np.zeros((frames,rows, columns), dtype = np.uint8)
    
    i = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            tif_image[i] = frame[:,:,index].copy()
            i += 1
        if i == frames:
            break
    
    cap.release()
    return tif_image
    
def find_track_window_object(bool_image, window_mask, orig_image, window_area, prev_object_area, ws_smooth, ws_kernal, ws_dist):
    #segment and watershed to get objects in thresholded image
    #identify object that overlaps largest area of the tracking window
    #calculate mean intensity of object *limited to tracking window*
    #returns mean intensity, bbox of object
    
    #fill holes
    bool_image = ndimage.morphology.binary_fill_holes(bool_image)
    
    #label objects
    labeled, number = mh.label(bool_image,np.ones((3,3), bool))
            
    ### for debugging ###
    l1 = color.label2rgb(labeled,bg_label=0)
    l1_=img_as_ubyte(l1)
    
    #select region that covers the most % of tracking window
    regions =  regionprops(labeled,orig_image)
    max_perc = 0
    for i, region in enumerate(regions):
        count = 0
        for (row,col) in region.coords:
            if(window_mask[row][col]):
                count+= 1
        if(max_perc < count/float(window_area)):
            max_perc = count/float(window_area)
            mean_intensity = region.mean_intensity
            bbox = region.bbox
            region_label = region.label
            region_area = region.area
            
    #check regions is not too big
    if((prev_object_area == 0 and region_area > window_area) or #first case is the initial frame
       (prev_object_area > 0 and region_area > prev_object_area*1.5)):
        #print "*Watershed*"
        #might have 2 cells touching as the region, try watershed
        labeled_2 = watershedsegment(bool_image, ws_smooth, ws_kernal, ws_dist)
      
        ### for debugging ###
        l2 = color.label2rgb(labeled_2,bg_label=0)
        l2_=img_as_ubyte(l2)
        
        #combine segmentations
        labeled = segmentation.join_segmentations(labeled, labeled_2)
        
        ### for debugging ###
        l3 = color.label2rgb(labeled,bg_label=0)
        l3_=img_as_ubyte(l3)
     
        #select region that covers the most % of tracking window
        regions =  regionprops(labeled,orig_image)
        max_perc = 0
        for i, region in enumerate(regions):
            count = 0
            for (row,col) in region.coords:
                if(window_mask[row][col]):
                    count+= 1
            if(max_perc < count/float(window_area)):
                max_perc = count/float(window_area)
                mean_intensity = region.mean_intensity
                bbox = region.bbox
                region_label = region.label
                region_area = region.area    
        
                
    elif(prev_object_area > 0 and region_area*1.5 < prev_object_area):
    #check region is not too small
    #might have a low intensity object and need to try kmeans instead of otsu thresh
        pass

                   
    #create mask with only the identified object
    obj_mask = np.zeros_like(orig_image)
    obj_mask[labeled == region_label] = np.iinfo(orig_image.dtype).max
    
    return (obj_mask, mean_intensity, bbox)

class cell_tracking:
     #initiate
     """Class for cell tracking """
     def __init__(self,zstack, zstack_to_segment, zstack_color):
        """ 
        zstack: (Z, N, M) ndarray 
            Intensity zstack where Z is the number of slices or frames
        
        zstack_to_segment: (Z, N, M) ndarray where
            Z is the number of slices or frames
        
        zstack_color: An RGB Stack (Z, N, M, 3) ndarray where S is the number of slices or frames
        """
        self.zstack = zstack.copy()
        self.zstack_to_segment = zstack_to_segment.copy()
        self.zstack_color = zstack_color.copy()
        self.zstack_color_orig = zstack_color.copy()
        #Parameters
        self._enhance = None
        self._blur = None
        self._kernel = None
        self._n_intensities = None
        self._smooth_distance = None
        self._distance_kernel = None
        self._min_distance = None
        
        #Parameters
        self._max_sigma = None
        self._min_sigma = None
        self._num_sigma = None
        self._threshold = None
        self._overlap = None
      
        #Parameters
        self.segment_param = False
        self.watershed_param = False
        self.blob_param = False
        self.positions_table = []
        
     def set_segment_param(self,enhance = True, blur = True, kernel = 61, n_intensities = 2):
         
         """Segmentation parameters. Add them before using the function stack_enhance_blur_segment

         enhance: bool
            If the image will be enhanced using the CLAHE (Contrast Limited Adaptive Histogram Equalization) from OpenCV
        
         blur: bool 
            If the image will be blurred using the gaussian blur
        
         kernel: int 
           Kernel size of the gaussian blur
           
         n_intensities: int
            Number of intensities you want to segment. If you want to segment into two imatensities Otsu is used
            If the intensities are greater than 2, then kmeans_img function is used to segement
         
         """
        
         self._enhance = enhance
         self._blur = blur
         self._kernel = kernel
         self._n_intensities = n_intensities
         self.segment_param = True

     def set_watershed_param(self,smooth_distance = True, kernel = 3, min_distance=10):
         """Watershed segmentation parameters. Add them before using the function stack_watershedsegment 
         
         smooth_distance: bool
            If distances will be smoothed
        
         kernel: int 
            Kernel size for smoothing the distances map
         
         """
         
         self._smooth_distance = smooth_distance
         self._distance_kernel = kernel
         self._min_distance = min_distance
         self.watershed_param = True
         
     def set_blob_param(self,max_sigma,min_sigma,num_sigma, threshold, overlap):
         """Blob detection  parameters. Add them before using any blob detection function """
         
         self._max_sigma = max_sigma
         self._min_sigma = min_sigma
         self._num_sigma = num_sigma
         self._threshold = threshold
         self._overlap = overlap
         self.blob_param = True
         
     def stack_enhance_blur_segment(self):
        """Created stacks of enhanced, and segmented stacks"""
        
        cl1_stack = np.zeros_like(self.zstack_to_segment)
        gaussian_blur_cl1_stack = np.zeros_like(self.zstack_to_segment)
        segmented_stack = np.zeros_like(self.zstack_to_segment)
        
        for z_level in range(0,self.zstack_to_segment.shape[0]):
            img = self.zstack_to_segment[z_level].copy()
            
            cl1, gaussian_blur_cl1, segmented, centers = enhance_blur_segment(img,self._enhance,self._blur,self._kernel,self._n_intensities)
            
            cl1_stack[z_level] = cl1
            gaussian_blur_cl1_stack[z_level] = gaussian_blur_cl1
            segmented_stack[z_level] = segmented
        
        
        self.cl1_stack = cl1_stack   
        self.gaussian_blur_cl1_stack = gaussian_blur_cl1_stack
        self.segmented_stack = segmented_stack
        self.stack_blob = gaussian_blur_cl1_stack
        
        #return (cl1_stack,gaussian_blur_cl1_stack,segmented_stack)
    
     def stack_watershedsegment(self):
        """Creates labeled stack after watershed segmentation """
        
        labeled_stack = np.zeros_like(self.segmented_stack)
        for z_level in range(0,self.segmented_stack.shape[0]):
            segmented  = self.segmented_stack[z_level].copy()
            labeled = watershedsegment(segmented,self._smooth_distance,self._distance_kernel)
            labeled_stack[z_level]=labeled.copy()
        
        self.labeled_stack = labeled_stack
        #return labeled_stack
    
     def create_table_blobs(self): 
        """Creates DataFrame table of measurements of blobs. The columns are 'x_col','y_row','z','r','label','track_window','mean_intensity','median_intensity' """
        
        n_label = 1 
        positions = []
        
        for n in range(0,self.stack_blob.shape[0]):
            img = self.stack_blob[n] 
            intensity_image = self.zstack[n]

            blobs = blob_log(img, self._max_sigma,  self._min_sigma,self._num_sigma,self._threshold,self._overlap)
            
            if len(blobs) > 0:
                
                blobs[:, 2] = blobs[:, 2] * sqrt(2)
                for blob in blobs:
                    y_row, x_col, r = blob
            
                    position = []
                    position.append(x_col)
                    position.append(y_row)
                    position.append(n)
                    position.append(r)
                    position.append(0)
                    x0 = x_col - r
                    y0 = y_row - r
                    
                    if x0 < 0:
                        x0 = 0
                    if y0 < 0:
                        y0 = 0
                   
                    
                    track_window = (int(x0), int(y0), int(r*2), int(r*2))
                    position.append(track_window)
                    mask = np.zeros_like(intensity_image)
                  
                    cv2.circle(mask, (int(x_col),int(y_row)), int(r),255 , -1)
                    mean_intensity = intensity_image[mask > 0].mean()
                    median_intensity = np.median(intensity_image[mask > 0])
                    position.append(mean_intensity)
                    position.append(median_intensity)
                    n_label += 1
                    positions.append(position)
       
        self.positions_table  = DataFrame(positions, columns = ['x_col','y_row','z','r','label','track_window','mean_intensity','median_intensity'])
        #self.positions = np.array(positions)
        #return  (self.positions, self.positions_table)
    
     def create_table_regions(self): 
        
        """Creates DataFrame table of measurements of regions. The columns are 'x_col','y_row','z','r','label','track_window','mean_intensity'"""
        
        n_label = 1 
        positions = []
           
        for n in range(0,self.labeled_stack.shape[0]):  
            labeled_slice = self.labeled_stack[n]
            intensity_image = self.zstack[n]

            for region in regionprops(labeled_slice, intensity_image):
                position = []
                
                y_row = region.centroid[0]
                x_col = region.centroid[1]
                y0, x0, y1, x1 = region.bbox #(min_row, min_col, max_row, max_col)

                r = (x1-x0)/2
                
                position.append(x_col)
                position.append(y_row)
                position.append(n)
                position.append(r)
                position.append(0)
                track_window = (x0, y0, x1-x0, y1-y0)

                position.append(track_window)
                mean_intensity = region.mean_intensity
                position.append(mean_intensity)

                n_label += 1
                positions.append(position)
        self.positions_table  = DataFrame(positions, columns = ['x_col','y_row','z','r','label','track_window','mean_intensity'])
        #self.positions = np.array(positions)
        #return  (self.positions, self.positions_table)        
                    
       
     def add_labels_table(self):
        """ Adds the labels fo the cells based on the distances of the center        
        """
        
        positions = self.positions_table.ix[:,'x_col':'z'].as_matrix()
        max_label = self.positions_table['label'].max()
        
        # Loops through each zlevel 
        for z_level in  range(0,self.positions_table.z.max()+1):
            #create a table of only the posiitons in the zlevel
            z_level_positions =  self.positions_table[self.positions_table.z == z_level]
            
            
            if z_level <  self.positions_table.z.max() :
                #create a table of the positions of the next zlevel
                z_level_positions_other =  self.positions_table[self.positions_table.z == z_level+1] #maybe change this
                #fit the positions
                positions = z_level_positions_other.ix[:,'x_col':'z'].as_matrix()
                nbrs = NearestNeighbors().fit(positions)
                
                # loop for each region in zlevel positions
            
                for index in z_level_positions.index: # for each of the regions in the slice
                    region = z_level_positions.ix[index,'x_col':'z'].as_matrix()
                    region_label = z_level_positions.ix[index,'label']
                    
                    if region_label == 0:  # if region label is 0 then add 1
                        max_label+=1
                        region_label = max_label
                        self.positions_table.ix[index,'label'] = region_label # change the label
                    
                    region_radius = z_level_positions.ix[index,'r'] # get radius
                    
                    dist,indexes = nbrs.kneighbors(region.reshape(1,-1),1) #find in the next slice what is the closest center                    
                    
                    if dist[0][0] > region_radius: #if the distance is greater than the radius continue
                        continue
                    elif len(dist) == 0:
                        continue
                    else:
                        location = indexes[0][0] #if not then add the label to that region as well
                        positions_table_index = z_level_positions_other.index[location]
                        self.positions_table.ix[positions_table_index,'label']= region_label
            else: #last zlevel
                for index in z_level_positions.index: # for each of the regions in the zlevel
                    region = z_level_positions.ix[index,'x_col':'z'].as_matrix()
                    region_label = z_level_positions.ix[index,'label']
                    
                    if region_label == 0:  # if region label is 0 then add 1
                        max_label+=1
                        self.positions_table.ix[index,'label'] = max_label
    
     def filter_table(self,min_slices):
        """Function to remove labeled blobs or regions that are present in less than a minimum number of slices or frames
        """
        table_positions  = self.positions_table.copy()

        for unique_label in np.unique(table_positions.label):
            table_label = table_positions[table_positions.label == unique_label ]
           
            if len(table_label) < min_slices:
                table_positions = table_positions[table_positions.label != unique_label ]
    
        
        self.positions_table = table_positions
        
     def draw_labels(self, color = (0,0,255), with_blobs =True ,width = 2):
        """Draws labels and blobs/circles in the cells in the zstack_color """
        for z_level in  range(0,self.positions_table.z.max()+1):
            
            z_level_positions =  self.positions_table[self.positions_table.z == z_level]
            zstack_slice = self.zstack_color[z_level]
            
            for index in z_level_positions.index:
                y_row = self.positions_table.ix[index,'y_row']
                x_col = self.positions_table.ix[index,'x_col']
                r = self.positions_table.ix[index,'r']
                label = self.positions_table.ix[index,'label']
                
                cv2.putText(zstack_slice,str(label),(int(x_col),int(y_row)),font,1,color,width,cv2.LINE_AA)
                if with_blobs:
                    cv2.circle(zstack_slice, (int(x_col),int(y_row)), int(r),color , width)
                
     def draw_contours(self, color_contours = (0,0,255),  width = 2):
        """Draws the contours in the cells in the zstack_color """

        for z_level in range(self.zstack_color.shape[0]):
            labeled = self.labeled_stack[z_level]
            color = self.zstack_color[z_level]
            draw_contours(labeled,color,with_labels=False, color = (255,0,0) ,width = 1 )
     
     def track_with_blob(self,min_slices = 2, color_blobs = (0,0,255), reset_drawing = False):
         """Function for tracking with blob detection. You need to set the segmentation parameters and the blob parameters before running.
         It will created a table of the positions and measurments and will draw the labels in the zstack color image
         
         """
         
             
         if len(self.positions_table) == 0:
             if self.segment_param == True and self.blob_param == True:
                 self.stack_enhance_blur_segment()
                 self.create_table_blobs()
                 self.add_labels_table()
             else:
                 print "Set blob and segmentation parameters"
         self.filter_table(min_slices)
         if reset_drawing:
              self.reset_drawing()
         self.draw_labels(color= color_blobs)
        
    
     def track_with_regions(self,color_contours = (255,0,0),color_labels = (255,0,0),min_slices = 10,  reset_drawing = False):
         """Function for tracking with regions . You need to set the segmentation and watershed segmentation parameters before running.
         It will created a table of the positions and measurments and will draw the labels in the zstack color image
         
         """
         if self.segment_param == True and self.watershed_param == True:
             self.stack_enhance_blur_segment()
             self.stack_watershedsegment()
             if len(self.positions_table) == 0:
                 self.create_table_regions()
                 self.add_labels_table()
             self.filter_table(min_slices)
             if reset_drawing:
                  self.reset_drawing()
             self.draw_labels(color=color_labels, with_blobs= False)
             self.draw_contours(color_contours=color_contours)
         else:
             print "Set segmentation parameters"
             
     def find_trajectories(self):
        """
        Creates a table of the trajectories for each cell
        """
        trajectories ={}

        for label in self.positions_table['label'].unique():
            table_label = self.positions_table[self.positions_table.label == label]
            table_label = table_label.sort_values(by = 'z')

            trajectory_matrix = table_label.ix[:, 'x_col':'y_row'].as_matrix()
            #dist = distance.pdist(trajectory_matrix.T)
            dist = 0
            for index in range(0, len(trajectory_matrix)):
                if index+1 < len(trajectory_matrix):
                    dist += distance.euclidean(trajectory_matrix[index],trajectory_matrix[index+1])
                
            trajectories[label] = {'trajectory':trajectory_matrix, 'distance': dist ,'dist_per_frame' :dist/len(trajectory_matrix)}
      
        table_trajectories =DataFrame(trajectories)
        self.trajectories = table_trajectories
        
     def draw_trajectories(self,color_trajectory = (255,0,0) , width = 2):
       """Draws the trajectories of each cell in the zstack_color. Make sure to run find_trajectories function""" 
       
       zstack_color = self.zstack_color
       
       for label in self.positions_table['label'].unique():
            table_label = self.positions_table[self.positions_table.label == label]
            table_label = table_label.sort_values(by = 'z')
            z_min = table_label.z.min()
            zslice = zstack_color[z_min]
            zslice_row = table_label[table_label['z'] == z_min]
            center = zslice_row.ix[:,'x_col':'y_row'].as_matrix()
            trajectory = center[0].astype('int64')
            x = int(zslice_row.ix[:,'x_col'].values[0])
            y = int(zslice_row.ix[:,'y_row'].values[0])
            cv2.circle(zslice, (x,y), width, color_trajectory , -1)
            
            for z in table_label.z.unique():
                if z > z_min:
                    zslice = zstack_color[z]
                    zslice_row = table_label[table_label['z'] == z]
                    new_center = zslice_row.ix[:,'x_col':'y_row'].as_matrix()
                    new_center = new_center[0].astype('int64')
                    trajectory =  np.vstack((trajectory,new_center))
                    cv2.polylines(zslice,[trajectory],False, color_trajectory,width)
                else:
                    continue
                
     def create_video(self, path, name, fps = 2):
        """Creates and saves the zstack_color as a  mp4v video in the path """
        
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') # note the lower case
        vout = cv2.VideoWriter()
    
        zstack_color = self.zstack_color
            
        zstack_color_bgr = np.zeros_like(zstack_color) 
        success = vout.open(path+name+'.mov',fourcc,fps,zstack_color_bgr.shape[1:3],True)
        zstack_color_bgr[:,:,:,0] = zstack_color[:,:,:,2].copy()
        zstack_color_bgr[:,:,:,1] = zstack_color[:,:,:,1].copy()
        zstack_color_bgr[:,:,:,2] = zstack_color[:,:,:,0].copy()
        
        for frame in list(range(self.zstack_color.shape[0])):
            new_frame = zstack_color_bgr[frame]
            vout.write(new_frame)
                
        vout.release() 
        
     def reset_drawing(self):
        """Resets the drawing of the zstack_color"""
         
        self.zstack_color = self.zstack_color_orig.copy()
        
     def track_window_object(self, z, track_window, enhance_bool = True, blur_bool = True, kernel_size = 11):
        """ 
        Function to track an object centered in the given track_window
        Uses thresholding over the track_window pixels to identify the object
        (The object that overlaps the largest % of pixels in the track_window will be identified)
        The bbox of the identified object will be used in the next iteration of tracking
        Mean intensity and position is saved for each frame
        In addition, object contours and tracking window are marked on each frame
        (in order to re-create a video of the tracking)
        CamShift/MeanShift are not used, works with 16-bit images
        """
        positions = []
        
        x0 = track_window[0]
        y0 = track_window[1]
        w = track_window[2]
        h = track_window[3]
        x1 = x0+w
        y1 = y0+h
        x_col = x0+(w/2)
        y_row = y0+(h/2)
        
        zstack_color = self.zstack_color_orig.copy()
        zstack = self.zstack_to_segment.copy()
        zstack_intens = self.zstack.copy()

        first_frame = zstack[z].copy()
        first_frame_color = zstack_color[z]
        first_frame_intens = zstack_intens[z]
        
        max_of_dtype = np.iinfo(first_frame.dtype).max #to work with 8 or 16 bit
        
        tw_mask = np.zeros_like(first_frame)
        tw_mask[y0:y0+h,x0:x0+w] = max_of_dtype 
        
        #frame_img_adjusted = exposure.adjust_gamma(first_frame, gamma=0.5)
        #frame_img_adjusted = filters.gaussian(frame_img_adjusted,sigma=0.8)
        frame_img_adjusted = enhance_blur(first_frame, enhance=enhance_bool, blur=blur_bool, kernel=kernel_size)

        
        #find th over tracking window only
        if len(np.unique(frame_img_adjusted[tw_mask > 0]))> 2:
            th = threshold_otsu(frame_img_adjusted[tw_mask > 0])
        else: 
            th = 0
            
        #apply th to entire image
        thresh_img = frame_img_adjusted > th
        
        #segment and watershed to get objects in thresholded image
        #identify object that overlaps largest area of the tracking window
        #calculate mean intensity of object *limited to tracking window*
        object_image,mean_intensity,bbox = find_track_window_object(thresh_img, tw_mask, first_frame_intens, w*h, 0,
                                                       self._smooth_distance, self._distance_kernel, self._min_distance) #watersh params
        
        #add to positions list for output
        position =[]
        position.append(z)
        position.append(x_col)
        position.append(y_row)
        position.append(track_window)
        position.append(mean_intensity)
        positions.append(position) 
        
        #draw current tracking window on image frame
        cv2.rectangle(first_frame_color, (x0,y0), (x0+w,y0+h), (max_of_dtype,max_of_dtype,0),2)
        
        #draw contours around region used for intensity measurement 
        object_image = img_as_ubyte(object_image)  
        im2, contours, hierarchy = cv2.findContours(object_image,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(first_frame_color, contours, -1, (max_of_dtype,0,max_of_dtype), 2)
        
        #track for rest of frames
        num_failed_qc = 0
        for frame_n in range(z+1,zstack_color.shape[0]):
            #print 'frame='+str(frame_n)
            #if(frame_n == 37):
            #    print frame_n
#            if(frame_n == 68):
#                print frame_n
            
            #get current frame
            frame_img = zstack[frame_n]
            frame_img_color = zstack_color[frame_n]
            frame_img_intens = zstack_intens[frame_n]
            
            tw_mask = np.zeros_like(frame_img)
            tw_mask[y0:y0+h,x0:x0+w] = max_of_dtype 
        
            #frame_img_adjusted = exposure.adjust_gamma(frame_img, gamma=0.5)
            #frame_img_adjusted = filters.gaussian(frame_img_adjusted,sigma=0.8)
            frame_img_adjusted = enhance_blur(frame_img, enhance=enhance_bool, blur=blur_bool, kernel=kernel_size)

            
            #find th over tracking window only
            if len(np.unique(frame_img_adjusted[tw_mask > 0]))> 2:
                th = threshold_otsu(frame_img_adjusted[tw_mask > 0])
            else: 
                th = 0
            
            #apply th to entire image
            thresh_img = frame_img_adjusted > th
            
            #segment and watershed to get objects in thresholded image
            #identify object that overlaps largest area of the tracking window
            #calculate mean intensity of object *limited to tracking window*
            failed_qc = False
            
            prev_object_image = object_image.copy()
            object_image[object_image>0] = 1
            prev_object_area = object_image.flatten().sum()  
            
            object_image,mean_intensity,bbox = find_track_window_object(thresh_img, tw_mask, frame_img_intens, w*h, prev_object_area,
                                                           self._smooth_distance, self._distance_kernel, self._min_distance) #watersh params
            
            y0_, x0_, y1_, x1_ = bbox
            x_col_ = x0_+((x1_-x0_)/2.)
            y_row_ = y0_+((y1_-y0_)/2.)
            
            #draw current tracking window on image frame
            #cv2.rectangle(frame_img_color, (x0,y0), (x0+w,y0+h), (max_of_dtype,max_of_dtype,0),2)
            
            #quality check: if NEW tw is out of range of OLD tw, keep using OLD tw
            #checks of size out of range or distance from prev tw out of range
            if((x1_-x0_) < 1.25*w and (y1_-y0_) < 1.25*h and (x1_-x0_) > .75*w and (y1_-y0_) > .75*h
               and (distance.euclidean((x_col_,y_row_),(x_col,y_row)) < (0.5)*max(x1-x0,y1-y0)) ):
               #passed quality check
                
                #set up NEW track window based on found object (movement of nuclei)
                y0, x0, y1, x1 = bbox
                track_window = (x0, y0, x1-x0, y1-y0) 
            
                #set up vars for new track window
                w = track_window[2]
                h = track_window[3]
                x_col = x0+(w/2)
                y_row = y0+(h/2)
            else: 
                failed_qc = True
                #print '*failed QC*'
                #sio.imsave("/Users/sarahkeegan/fenyolab/data_and_results/pagano/rona/temp/"+str(frame_n)+"_th2.tiff",
                #           thresh_img.astype('uint8')*255)
                
                #do not update to new tw
                #do not update object
                #obtain mean intensity over old object mask in the new frame
                labeled, number = mh.label(prev_object_image>0,np.ones((3,3), bool))
                regions = regionprops(labeled,frame_img_intens)
                
                mean_intensity = regions[0].mean_intensity
                
                #draw contours and rect around region that did not pass QC (for debugging only)
                #cv2.rectangle(frame_img_color, (x0_,y0_), (x0_+(x1_-x0_),y0_+(y1_-y0_)), (0,0,max_of_dtype),1)
                object_image = img_as_ubyte(object_image) 
#                im2, contours, hierarchy = cv2.findContours(object_image,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
#                cv2.drawContours(frame_img_color, contours, -1, (0,0,max_of_dtype), 1)
#            
                object_image = prev_object_image
    
            #draw tracking window on image frame
            cv2.rectangle(frame_img_color, (x0,y0), (x0+w,y0+h), (max_of_dtype,max_of_dtype,0),2)
            
            #draw contours around region used for intensity measurement 
            object_image = img_as_ubyte(object_image) 
            im2, contours, hierarchy = cv2.findContours(object_image,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(frame_img_color, contours, -1, (max_of_dtype,0,max_of_dtype), 2)
            
            #record data about this frame
            position = []
            position.append(frame_n)
            position.append(x_col)
            position.append(y_row)
            position.append(track_window)
            position.append(mean_intensity)
            positions.append(position)
            
            if(failed_qc):
                num_failed_qc = num_failed_qc + 1
            else: 
                num_failed_qc = 0
                
            if(num_failed_qc >= 10):
                positions = positions[:-10]
                print "Failed QC 10 times, stopping at frame " + str(frame_n) + " and removing final 10 track windows."
                break
            #if(frame_n >= 100): break
        
        positions_table  = DataFrame(positions, columns = ['z','x_col','y_row','track_window','mean_intensity'])    
        return zstack_color, positions_table
        
     def track_window(self, z, track_window,enhance_bool = False ,blur_bool = True, kernel_size = 61):
         
        """Function to track one cell. You will give the tracking window and zlevel and it will return a marked 
        zstack and table of intensity measurements""" 
        
        positions = []
        x0 = track_window[0]
        y0 = track_window[1]
        w = track_window[2]
        h = track_window[3]
        center = np.array([x0+(w/2),y0+(h/2)])
        x_col = center[0]
        y_row = center[1]
              
        zstack_color = self.zstack_color_orig.copy()
        zstack = self.zstack_to_segment.copy()
        
        
        first_frame = zstack[z].copy()
        first_frame_color = zstack_color[z]
                
        # thresh is a mask with a square or tracking window
        thresh = np.zeros_like(first_frame)
        thresh[y0:y0+h,x0:x0+w] = 255
        
        cl1, gaussian_blur_cl1, segmented_zlevel, centers = enhance_blur_segment(first_frame,enhance = enhance_bool, blur = blur_bool, kernel = kernel_size, n_intensities = 2)
        
        # getting the threshold inside the tracking window or square
        if len(np.unique(gaussian_blur_cl1[thresh > 0]))> 2:
            th = threshold_otsu(gaussian_blur_cl1[thresh > 0])
        else:
            th = 0
        # getting and roi mask where you  threshold the whole image
        mask_new_roi =gaussian_blur_cl1 > th
        
        # make a mask of the intersection between the mask with the tracking window and the one wiht the region
        combined_thresh =np.logical_and(mask_new_roi, thresh > 0)
        combined_thresh =combined_thresh.astype(np.uint8)
        combined_thresh[combined_thresh==1]= 255
        
        # calculate the histogram of the region
        roi_hist = cv2.calcHist([gaussian_blur_cl1],[0],combined_thresh,[256],[0,256]) # create histogram of roi
        
        term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1) #10, 1 

        
        # draw rectangle in the first color frame
        cv2.rectangle(first_frame_color, (x0,y0), (x0+w,y0+h), (255,0,0),2)
       
        # getting measurements
        
        # make labeled image
        labeled,number = mh.label(combined_thresh>0,np.ones((3,3), bool))#np.ones((3,3), bool)
        regions =  regionprops(labeled,first_frame)
    
        # remove small regions
        new_area = 0
        for region in regions:
            area = region.area
            if area > new_area:
                new_area = area
        # combined_thresh will have the small regions removed
        combined_thresh = remove_regions(combined_thresh, new_area, size='smaller')        
        roi_hist = cv2.calcHist([gaussian_blur_cl1],[0],combined_thresh,[256],[0,256]) # create histogram of roi gaussian_blur_cl1
        intensities = self.zstack[z][combined_thresh > 0].ravel()
        # getting the mean intensity
        labeled,number = mh.label(combined_thresh>0,np.ones((3,3), bool))#np.ones((3,3), bool)
        regions =  regionprops(labeled,self.zstack[z])
        index = 0
        mean_intensity = regions[index].mean_intensity        
       
        # drawing the contours
        im2, contours, hierarchy = cv2.findContours(combined_thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(first_frame_color, contours, -1, (0,0,255), 2)
               
        # adding all measurements
        position =[]
        position.append(z)
        position.append(x_col)
        position.append(y_row)
        position.append(track_window)
        position.append(mean_intensity)
        position.append(list(intensities))
		       
        #Start loop
        
        for frame_n in range(z+1,zstack_color.shape[0]):
            
            frame_img = zstack[frame_n]
            frame_img_color = zstack_color[frame_n]
            # create enhanced, blur segmentes and centers
            cl1, gaussian_blur_cl1, segmented_zlevel, centers = enhance_blur_segment(frame_img,enhance = enhance_bool, blur = blur_bool, kernel = kernel_size, n_intensities = 2)
        
            # create map of probabilities
            dst = cv2.calcBackProject([gaussian_blur_cl1],[0],roi_hist,[0,256],1)
            
            # get new tracking window and center
            retcam, track_window = cv2.CamShift(dst, track_window, term_crit)
            new_center = np.array(retcam[0])
            # calc distance of the new center and old center
            distance_val = distance.euclidean(center,new_center)
            
            # if the distance is greater than the width of the box or greater than the height, then break
            if (distance_val > w) or (distance_val > h):
                print('Center Far away'+str(frame_n)+' Possible_Division')
                break
                
            else:
                center = new_center# create histogram of roi
            # get the new width and the new height
            new_w = track_window[2]
            new_h = track_window[3]
            
            # if the height or width is less than 1/4 then break
            if (new_w < w/4) or (new_h < h/4):
                print('W or H small'+str(frame_n))
                break
                    
            else:
                w = new_w
                h = new_h
            
            # draw the tracking window
            pts = cv2.boxPoints(retcam)
            pts = np.int0(pts)
            cv2.polylines(frame_img_color,[pts],True, (255,0,0), 2)
        
            # create a mask with the tracking window
            
            thresh = np.zeros_like(first_frame)
            cv2.fillPoly(thresh,[pts], (255,0,0))
            
            # get a threshold of inside the window
            if len(np.unique(gaussian_blur_cl1[thresh > 0]))> 2:
                th = threshold_otsu(gaussian_blur_cl1[thresh >0])
            else:
                th = 0
            # apply the threshold to the blurred image
            dst_thresh = gaussian_blur_cl1 > th
            dst_thresh = ndimage.morphology.binary_fill_holes(dst_thresh)
            dst_thresh = dst_thresh.astype(np.uint8)
            dst_thresh[dst_thresh==1]= 255
            
            # get labeled image
            labeled,number = mh.label(dst_thresh>0,np.ones((3,3), bool))
            
            # get a new labeled image with the segmented image obtained in 1260
            labeled_2 = watershedsegment(segmented_zlevel,self._smooth_distance,self._distance_kernel)
            
            # if the labeled image has more than one labeled object then join semgentetion
            if len(np.unique(labeled_2[thresh > 0])) > 2:
                labeled = segmentation.join_segmentations(labeled, labeled_2)
            
            # make everything outside the square as zero
            labeled[thresh != 255] =0
        
            # get the regions
            regions =  regionprops(labeled,frame_img)
            
            # eliminate small regions
            new_area = 0
            for region in regions:
                area = region.area
                if area > new_area:
                    new_area = area
                                
            combined_thresh = remove_regions(labeled, new_area, size='smaller')        
            labeled,number = mh.label(combined_thresh>0,np.ones((3,3), bool))
           
            # get the mean intensity
            
            regions =  regionprops(labeled,self.zstack[frame_n])
            print(len(regions))
            if len(regions) != 0:
            
                index =0            
                mean_intensity = regions[index].mean_intensity        
                
                # get the tracking window
                y0, x0, y1, x1 = regions[index].bbox #(min_row, min_col, max_row, max_col)
                track_window = (x0, y0, x1-x0, y1-y0)
                
                combined_thresh[combined_thresh > 0] = 255
                combined_thresh = combined_thresh.astype(np.uint8)
                
                # get and draw contours
                im2, contours, hierarchy = cv2.findContours(combined_thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(frame_img_color, contours, -1, (0,0,255), 2)
                
                # get new histogram
                roi_hist = cv2.calcHist([gaussian_blur_cl1],[0],combined_thresh,[256],[0,256])
                intensities = self.zstack[z][combined_thresh > 0].ravel()
                
                x_col = center[0]
                y_row = center[1]
                position =[]
                position.append(frame_n)
                position.append(x_col)
                position.append(y_row)
                position.append(track_window)
                position.append(mean_intensity)
                position.append(list(intensities))

                positions.append(position)
                
        positions_table  = DataFrame(positions, columns = ['z','x_col','y_row','track_window','mean_intensity','intensities'])    
        return zstack_color , positions_table                 
                        
     def track_window_graph(self,label,z,track_window, use_camshift=True, enhance_bool = False, blur_bool = True, kernel_size = 61, size = 2.5,vline_x =None ):
        plt.ioff()

        """Function to track one cell. You will give the tracking window and zlevel and it will return a marked zstack with a and table with intensity measurments""" 
        mpl.rcParams.update(mpl.rcParamsDefault)
        mpl.rcParams['axes.linewidth'] = 1
        mpl.rcParams['font.sans-serif'] = 'Arial'
        mpl.rcParams['font.size'] = 10
        
        pixels= int(size *100 + 25)

        #print "Begin tracking for label = " + str(label)
        if(use_camshift):
            tracked_label, measurements = self.track_window(z,track_window,enhance_bool,blur_bool, kernel_size)
        else:
            tracked_label,measurements = self.track_window_object(z,track_window,enhance_bool,blur_bool, kernel_size)
        
        stack_graph = np.zeros((tracked_label.shape[0],
                                tracked_label.shape[1]+pixels,
                                tracked_label.shape[2]+pixels,
                                tracked_label.shape[3]), dtype = 'uint8')
        
        #Change this
        measurementsSeries = Series(measurements.mean_intensity.values, index = measurements.z.values)
        last_zindex_measurements = measurements.iloc[len(measurements)-1].z 
        first_zindex_measurements = measurements.iloc[0].z
        max_measurements = max(measurements.mean_intensity.values)
        
        for z_index in range(0,tracked_label.shape[0]):
            zslice= tracked_label[z_index].copy() 
            slice_graph = np.zeros((zslice.shape[0]+pixels,zslice.shape[1]+pixels,3), dtype = 'uint8')
        
            if z_index < z or z_index > last_zindex_measurements:
                slice_graph[0:zslice.shape[0],pixels-1:pixels-1+zslice.shape[1]] = zslice
            else:
        
                _fig, _ax1  = plt.subplots(nrows=1,ncols=1, figsize=(size,size), dpi=100)
                _ax1.set_xlabel('Time')
                _ax1.set_ylabel('Mean Intensity')
                _ax1.plot(measurementsSeries,color ='w')
                if vline_x != None:
                     _ax1.axvline(x=vline_x, color='k',ls ='--')
                _ax1.set_title('Mean Intensity Cell '+ str(label))
                _ax1.set_xlim([0,tracked_label.shape[0]]) #len(measurements)]) 
                _ax1.set_ylim([0,max_measurements+.05*max_measurements])
                zslice= tracked_label[z_index].copy() 
                
                slice_graph[0:zslice.shape[0],pixels-1:pixels-1+zslice.shape[1]] = zslice
                
                series_index = z_index-first_zindex_measurements
                #_ax1.plot(measurementsSeries[0:series_index+1],color ='b')
                _ax1.plot(measurements.loc[0:series_index,'z'].values,measurements.loc[0:series_index,'mean_intensity'].values,color='b')
                
                _fig.tight_layout()
                _fig.canvas.draw()
                
                data = np.fromstring(_fig.canvas.tostring_rgb(), dtype='uint8', sep='')
                data = data.reshape(_fig.canvas.get_width_height()[::-1] + (3,))
                slice_graph[10:int(size*100+10),10:int(size*100+10)] = data.copy()
                
                plt.clf()
                plt.close()
                
            stack_graph[z_index] = slice_graph.copy()
        plt.ion()
        return (stack_graph,measurements)
    
     def track_blob_labeled(self,label,levels_after =4,size = 2.5):
        
        plt.ioff()
         
        """This function will track one cell and plot the mean intensity measurements. It will return the zstack with the graph"""
        
        mpl.rcParams['axes.linewidth'] = 1
        mpl.rcParams['font.sans-serif'] = 'Arial'
        mpl.rcParams['font.size'] = 10
        pixels = size * 100+ 25
        zstack_color = self.zstack_color
       
        table_label = self.positions_table[self.positions_table.label == label]
        table_label = table_label.sort_values(by = 'z')
                
        stack_graph = np.zeros((len(table_label)+levels_after,zstack_color.shape[1]+pixels,zstack_color.shape[2]+pixels,zstack_color.shape[3]), dtype = np.uint8)
        measurements_z = table_label.z.values.astype('int')
        measurements_mean = table_label.mean_intensity.values
        measurementsSeries = Series(measurements_mean, index = measurements_z)
        min_z = measurementsSeries.index.min()
        max_z = measurementsSeries.index.max()
        for i, z in enumerate(range(min_z,max_z+levels_after+1)): 
            if z < zstack_color.shape[0]:
                
                zslice= zstack_color[int(z)].copy()
                slice_graph = np.zeros((zslice.shape[0]+pixels,zslice.shape[1]+pixels,3), dtype = np.uint8)
                if z <= max_z:
                
                    zslice_row = table_label[table_label['z'] == z]
                    
                    x = int(zslice_row.ix[:,'x_col'].values[0])
                    y = int(zslice_row.ix[:,'y_row'].values[0])
                    r = int(zslice_row.ix[:,'r'].values[0])
                    
                    cv2.circle(zslice, (int(x),int(y)), int(r),(255,255,255) , 3)
                    
                    _fig, _ax1  = plt.subplots(nrows=1,ncols=1, figsize=(size,size), dpi=100)
                    _ax1.set_xlabel('Time')
                    _ax1.set_ylabel('Mean Intensity')
                    _ax1.plot(measurementsSeries,color ='w')
                    _ax1.set_title('Mean Intensity Cell '+ str(label))
                    _ax1.plot(measurementsSeries.ix[min_z:z],color ='b')
                    _fig.tight_layout()
                    _fig.canvas.draw()
                    
                    # Now we can save it to a numpy array.
                    data = np.fromstring(_fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                    data = data.reshape(_fig.canvas.get_width_height()[::-1] + (3,))
                    slice_graph[10:(size*100+10),10:(size*100+10)] = data.copy()
                     
                slice_graph[0:zslice.shape[0],(pixels-1):(pixels-1)+zslice.shape[1]] = zslice   
                stack_graph[i] = slice_graph.copy()
        plt.ion()
        
        return stack_graph
        
    
    
           
       

        
    
   
     