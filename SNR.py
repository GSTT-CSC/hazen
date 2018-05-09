# SNR(Im)
#
# Calculates the SNR for a single-slice image of a uniform MRI phantom
#
# This script utilises the smoothed subtraction method described in McCann 2013:
# A quick and robust method for measurement of signal-to-noise ratio in MRI, Phys. Med. Biol. 58 (2013) 3775:3790
#
#
# Created by Neil Heraghty
#
# 04/05/2018

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from pydicom.filereader import dcmread


def find_circle(a):
    """
    Finds a circle that fits the outline of the phantom using the Hough Transform
    
    parameters:
    ---------------
    a: image array (should be uint8)
    
    returns: 
    ---------------
    cenx, ceny: xy pixel coordinates of the centre of the phantom
    cradius : radius of the phantom in pixels
    errCircle: error message if good circle isn't found    
    """

    # Perform Hough transform to find circle
    circles = cv.HoughCircles(idown, cv.HOUGH_GRADIENT, 1, 100, param1=50,
                              param2=30, minRadius=0, maxRadius=0)

    # Check that a single phantom was found
    if len(circles) == 1:
        errCircle = "1 circle found."
    else:
        errCircle = "Wrong number of circles detected, check image."
    print(errCircle)

    # Draw circle onto original image
    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        # draw the outer circle
        cv.circle(idown,(i[0],i[1]),i[2],30,2)
        # draw the center of the circle
        cv.circle(idown,(i[0],i[1]),2,30,2)

    cenx = i[0]
    ceny = i[1]
    cradius = i[2]
    return (cenx, ceny, cradius)


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
    subM = strd(a, shape = s, strides = a.strides * 2)
    return np.einsum('ij,ijkl->kl', f, subM)


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
    size = (3,3)
    kernel = np.ones(size)/9

    # Convolve image with boxcar kernel
    imsmoothed = conv2d(a,kernel)
    # Pad the array (to counteract the pixels lost from the convolution)
    imsmoothed = np.pad(imsmoothed, 1, 'minimum')
    # Subtract smoothed array from original
    imnoise = a - imsmoothed

    return imnoise


# Read DICOM image
image = dcmread('uniformCRFA.dcm')     # Read the DICOM file

# Prepare image for processing
idata = image.pixel_array              # Read the pixel values into an array
idata = np.array(idata)                # Make it a numpy array
idown = (idata/256).astype('uint8')    # Downscale to uint8 for openCV techniques

# Find phantom centre and radius
(cenx, ceny, cradius) = find_circle(idown)

# Create noise image
imnoise = image_noise(idata)

# Measure signal and noise within 5 20x20 pixel ROIs (central and +-40pixels in x & y)
sig = [None]*5
noise = [None]*5

sig[0] = np.mean(idata[(cenx-10):(cenx+10),(ceny-10):(ceny+10)])
sig[1] = np.mean(idata[(cenx-50):(cenx-30),(ceny-50):(ceny-30)])
sig[2] = np.mean(idata[(cenx+30):(cenx+50),(ceny-50):(ceny-30)])
sig[3] = np.mean(idata[(cenx-50):(cenx-10),(ceny+30):(ceny+50)])
sig[4] = np.mean(idata[(cenx+30):(cenx+50),(ceny+30):(ceny+50)])

noise[0] = np.std(imnoise[(cenx-10):(cenx+10),(ceny-10):(ceny+10)])
noise[1] = np.std(imnoise[(cenx-50):(cenx-30),(ceny-50):(ceny-30)])
noise[2] = np.std(imnoise[(cenx+30):(cenx+50),(ceny-50):(ceny-30)])
noise[3] = np.std(imnoise[(cenx-50):(cenx-10),(ceny+30):(ceny+50)])
noise[4] = np.std(imnoise[(cenx+30):(cenx+50),(ceny+30):(ceny+50)])

# Draw regions for testing
cv.rectangle(idown,((cenx-10),(ceny-10)),((cenx+10),(ceny+10)),30,2)
cv.rectangle(idown,((cenx-50),(ceny-50)),((cenx-30),(ceny-30)),30,2)
cv.rectangle(idown,((cenx+30),(ceny-50)),((cenx+50),(ceny-30)),30,2)
cv.rectangle(idown,((cenx-50),(ceny+30)),((cenx-30),(ceny+50)),30,2)
cv.rectangle(idown,((cenx+30),(ceny+30)),((cenx+50),(ceny+50)),30,2)

# Plot annotated image for user
fig = plt.figure(1)
plt.imshow(idown, cmap='gray')
plt.show()

# Calculate SNR for each ROI and average
snr=np.divide(sig,noise)
mean_snr = np.mean(snr)

print("Measured SNR: ",int(round(mean_snr)))

