#!/usr/bin/env python

import cv2
import random
import numpy as np

# speed-up using multithreads
#cv2.setUseOptimized(True);
#cv2.setNumThreads(4);


def np_make_grid(ims, col = 3):
    """ims: a list or numpy"""
    # base unit size is h x w x c.
    c, h, w = ims[0].shape
    # display layout
    row = len(ims) // col
    # row * (h+1) x col * (w+1) x 3
    grid = np.zeros(row * (h + 2) * col * (w + 2) * c).astype(np.uint8).reshape(row * (h + 2), col * (w + 2), c)
    for i in range(row):
        for j in range(col):
            print(i*col+j)
            if c != 1:
                grid[i*(h+2)+1:(i+1)*(h+2)-1, j*(w+2)+1:(j+1)*(w+2)-1, :] = np.transpose(ims[col*i+j], (1, 2, 0))
            else:
                grid[i*(h+2)+1:(i+1)*(h+2)-1, j*(w+2)+1:(j+1)*(w+2)-1, :] = np.transpose(ims[col*i+j], (1, 2, 0))
    return grid


def salt_and_pepper(im, n, mode = 'random'):
    # set image to writeable
    im.flags.writeable = True
    
    for k in range(n):
        i = random.randint(0, im.shape[0] - 1)
        j = random.randint(0, im.shape[1] - 1)
        
        if mode == 'random':
            im[i, j, :] = [255, 255, 255] if random.random() <  0.5 else [0, 0, 0]
        elif mode == 'salt':
            im[i, j, :] = [255, 255, 255]
        elif mode == 'pepper':
            im[i, j, :] = [0, 0, 0]

    # set image to unwriteable
    im.flags.writeable = False
            
    return im


# References: https://www.learnopencv.com/selective-search-for-object-detection-cpp-python/
def selective_search(im, k, inc, sigma = 0.8, mode = None):
    '''return Rect(x, y, w, h)'''
    # create Selective Search Segmentation Object using default parameters
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
 
    # set input image on which we will run segmentation
    ss.setBaseImage(im)
 
    # Switch to fast but low recall Selective Search method
    if (mode == 'f'):
        ss.switchToSelectiveSearchFast(base_k = k, inc_k = inc, sigma = 0.8)
    # Switch to high recall but slow Selective Search method
    elif (mode == 'q'):
        ss.switchToSelectiveSearchQuality(base_k = k, inc_k = inc, sigma = 0.8)
        
    # run selective search segmentation on input image
    rects = ss.process()
    return rects

# References: https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
# 2 Refs: https://bitbucket.org/tomhoag/nms
def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
 
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
 
    # initialize the list of picked indexes
    pick = []
 
    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
 
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
 
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
 
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
 
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
 
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
 
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))
 
    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")
