"""
ACR Uniformity

Calculates uniformity for slice 7 of the ACR phantom.

This script calculates the integral uniformity in accordance with the ACR Guidance.
This is done by first defining a large 200cm2 ROI before placing 1cm2 ROIs at every pixel within
the large ROI. At each point, the mean of the 1cm2 ROI is calculated. The ROIs with the maximum and
minimum mean value are used to calculate the integral uniformity. The results are also visualised.

Created by Yassine Azma
yassine.azma@rmh.nhs.uk

13/01/2022
"""

import sys
import traceback
import numpy as np
import hazenlib.tools
import hazenlib.exceptions as exc

res = metadata.PixelSpacing # In-plane resolution from metadata
r_largeROI = np.ceil(80/res[0]) # Required pixel radius to produce ~200cm2 ROI
r_smallROI = np.ceil(np.sqrt(100/np.pi)/res[0]) # Required pixel radius to produce ~1cm2 ROI
dims = img_unif.shape # Dimensions of image

def centroid_com(img):
    # Calculate centroid of object using a centre-of-mass calculation
    mask = img > 0.25 * np.max(img)
    coords = np.nonzero(mask)  # row major - first array is columns

    sum_x = np.sum(coords[0])
    sum_y = np.sum(coords[1])
    cxy = sum_x / coords[0].shape, sum_y / coords[1].shape

    cxy = [cxy[0].astype(int), cxy[1].astype(int)]
    return cxy

def circular_mask(centre,radius,dims):
    # Define a circular logical mask
    nx = np.linspace(1, dims[1], dims[1])
    ny = np.linspace(1, dims[0], dims[0])

    x, y = np.meshgrid(nx, ny)
    mask = np.square(x - centre[1]) + np.square(y - centre[0]) <= np.square(radius)
    return mask

def integral_uniformity(img,report_path):
    cxy = centroid_com(img)
    base_mask = circular_mask(cxy, r_smallROI)  # Dummy circular mask at centroid
    coords = np.nonzero(base_mask)  # Coordinates of mask

    lROI = circular_mask([cxy[0], cxy[1] - np.divide(5 / res[1])], r_largeROI)
    img_masked = lROI*img
    lROI_extent = np.nonzero(lROI)

    for ii in range(0, len(lROI_extent[0])):
        centre = [lROI_extent[0][ii], lROI_extent[1][ii]]  # Extract coordinates of new mask centre within large ROI
        trans_mask = [coords[0] + centre[0] - cxy[0],
                      coords[1] + centre[1] - np.round(cxy[1])]  # Translate mask within the limits of the large ROI
        sROI_val = img_masked[trans_mask[0], trans_mask[1]]  # Extract values within translated mask
        if np.count_nonzero(sROI_val) < np.count_nonzero(base_mask):
            mean_val[ii] = 0
        else:
            mean_val[ii] = np.mean(sROI_val[np.nonzero(sROI_val)])
        mean_array[lROI_extent[0][ii], lROI_extent[1][ii]] = mean_val[ii]

    sig_max = np.max(mean_val)
    sig_min = np.min(mean_val[np.nonzero(mean_val)])

    max_loc = np.where(mean_array == sig_max)
    min_loc = np.where(mean_array == sig_min)

    PIU = 100 * (1 - (sig_max - sig_min) / (sig_max + sig_min))

    return {'Percentage Integral Uniformity = ' + PIU}

