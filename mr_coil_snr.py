# mr_coil_snr
#
# Reads in a DICOM image of a phantom, places an ROI in the brightest region and
# calculates the SNR
#
# Neil Heraghty
# neil.heraghty@nhs.net
#
# 11/07/2018

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy import signal
from pydicom.filereader import dcmread
import math


def tri_thresh(a):

    """
        Calculates a threshold for image binarisation using the "Triangle-method"

        This method was first described in the following work:

        Zack GW, Rogers WE, Latt SA (1977), "Automatic measurement of sister chromatid exchange frequency",
        J. Histochem. Cytochem. 25 (7): 74153

        The script aims to threshold an inhomogeneous object on a dark background, as is
        common in MRI.

        parameters:
        ---------------
        image        :   dicom image read using dcmread

        returns:
        ---------------
        tri_thresh    :   calculated threshold level
    """

    # Generate image histogram
    maxima = np.amax(a)
    ihist = np.bincount(np.ravel(a.astype(int)))
    histmax = np.amax(ihist)
    histpeak = np.argmax(ihist)

    # Calcalate gradient of straight line
    gradient = -histmax / (maxima-histpeak)

    # Calculate angle between histogram peak and data maximum (in radians)
    alpha = math.atan((maxima-histpeak)/histmax)

    # Generate tangent array, len
    len = [None] * (maxima.astype(int)-histpeak)

    for x in range(histpeak, maxima.astype(int)):
        len[x-histpeak] = (gradient*(x-histpeak) + histmax - ihist[x])*math.sin(alpha)

    # Determine threshold: the value at which len is maximised
    tri_thresh = np.argmax(len) + histpeak

    return tri_thresh


def image_noise(a):
    """
    Separates the image noise by smoothing the image and subtracing the smoothed image
    from the original.

    parameters:
    ---------------
    a: image array from dcmread and .pixelarray

    returns:
    ---------------
    Inoise: image representing the image noise
    """

    # Create 3x3 boxcar kernel (recommended size - adjustments will affect results)
    size = (3, 3)
    kernel = np.ones(size) / 9

    # Convolve image with boxcar kernel
    imsmoothed = conv2d(a, kernel)
    # Pad the array (to counteract the pixels lost from the convolution)
    imsmoothed = np.pad(imsmoothed, 1, 'minimum')
    # Subtract smoothed array from original
    imnoise = a - imsmoothed

    return imnoise, imsmoothed


def conv2d(a, f):
    """
    Performs a 2D convolution (for filtering images)

    parameters:
    ---------------
    a: array to be filtered
    f: filter kernel

    returns:
    ---------------
    filtered numpy array
    """

    s = f.shape + tuple(np.subtract(a.shape, f.shape) + 1)
    strd = np.lib.stride_tricks.as_strided
    subM = strd(a, shape=s, strides=a.strides * 2)
    return np.einsum('ij,ijkl->kl', f, subM)


# Read image
image = dcmread('test_thresh.dcm')

# Prepare image for processing
idata = image.pixel_array  # Read the pixel values into an array
idata = np.array(idata)  # Make it a numpy array
idown = ((idata / idata.max()) * 256).astype('uint8')  # Downscale to uint8 for openCV techniques
idata = idata.astype(float)


# Create noise image
[imnoise, imsmoothed] = image_noise(idata)

# Find brightest region of image
[ymax, xmax] = np.unravel_index(imsmoothed.argmax(), imsmoothed.shape)

# Apply median filter
ifilt = signal.medfilt2d(idata, (5, 5))

# Threshold image
tri_thresh = tri_thresh(ifilt) # calculate image threshold
ithresh = np.zeros(ifilt.shape)
mask = ifilt > tri_thresh
ithresh[mask] = 1

# Find centre of mass of thresholded image
count = (ithresh == 1).sum()
y_center, x_center = np.argwhere(ithresh==1).sum(0)/count

# Calculate midpoint between phantom centre and brightest point
y_mid = (np.round((y_center + ymax)/2)).astype(int)
x_mid = (np.round((x_center + xmax)/2)).astype(int)

# Measure mean signal and stdev noise in a 20x20 ROI at this location
sig = np.mean(idata[(y_mid - 10):(y_mid + 10), (x_mid - 10):(x_mid + 10)])
noise = np.std(imnoise[(y_mid - 10):(y_mid + 10), (x_mid - 10):(x_mid + 10)])

# Calculate SNR
SNR = sig/noise

# Draw ROI for QA purposes
cv.rectangle(idown, ((x_mid - 10), (y_mid - 10)), ((x_mid + 10), (y_mid + 10)), 32, 2)

# Display the image
fig = plt.figure(1)
plt.imshow(idown,cmap='gray')
plt.show()