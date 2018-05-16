# Spatial Resolution
#
# This script determines the spatial resolution of image by measuring the FWHM across the edges of
# a uniform phantom
#
# Created by Neil Heraghty
# neil.heraghty@nhs.net
#
# 16/05/2018

import cv2 as cv
import numpy as np
import numpy.polynomial.polynomial as poly
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
    for i in circles[0, :]:
        # draw the outer circle
        cv.circle(idown, (i[0], i[1]), i[2], 30, 2)
        # draw the center of the circle
        cv.circle(idown, (i[0], i[1]), 2, 30, 2)

    cenx = i[0]
    ceny = i[1]
    cradius = i[2]
    return (cenx, ceny, cradius)


def calc_fwhm(lsf):
    """
    Determines the FWHM of an LSF

    parameters:
    ---------------
    lsf: image array (1D)

    returns:
    ---------------
    fwhm: the measured fwhm in pixels
    """

    # Find the peak pixel
    ind1 = np.argmax(lsf)

    # Create array of values around the peak
    lsfp = np.asarray([lsf[ind1-1], lsf[ind1], lsf[ind1+1]])
    # Create index array
    xloc = np.asarray([ind1-1, ind1, ind1+1])

    # Fit a 2nd order polynomial to the points
    fit = poly.polyfit(xloc,lsfp,2)

    # Find the peak value of this polynomial
    x = np.linspace(int(ind1-1), int(ind1+1), 1000)
    y = fit[0] + fit[1]*x + fit[2]*x**2
    peakval = np.max(y)
    halfpeak = peakval/2

    # Find indices where lsf > halfpeak
    gthalf = np.where(lsf > halfpeak)
    gthalf = gthalf[0]

    # Find left & right edges of profile and corresponding values
    leftx = np.asarray([gthalf[0]-1,gthalf[0]])
    lefty = lsf[leftx]
    rightx = np.asarray([gthalf[len(gthalf)-1],gthalf[len(gthalf)-1]+1])
    righty = lsf[rightx]

    # Use linear interpolation to find half maximum locations and calculate fwhm (pixels)
    lefthm = np.interp(halfpeak,lefty,leftx)
    righthm = np.interp(halfpeak,np.flip(righty,0),np.flip(rightx,0))
    fwhm = righthm-lefthm

    return fwhm


# Read DICOM image
image = dcmread('uniformCNSA.dcm')     # Read the DICOM file

# Read pixel size
pixelsize = image[0x28,0x30].value

# Prepare image for processing
idata = image.pixel_array              # Read the pixel values into an array
idata = np.array(idata)                # Make it a numpy array
idown = (idata/256).astype('uint8')    # Downscale to uint8 for openCV techniques

# Find phantom centre and radius
(cenx, ceny, cradius) = find_circle(idown)

# Create profile through edges
lprof = idata[(cenx-cradius-20):(cenx-cradius+20),ceny]
bprof = idata[cenx,(ceny+cradius-20):(ceny+cradius+20)]
bprof = np.flipud(bprof)

# Differentiate profiles to obtain LSF
llsf = np.gradient(lprof)
blsf = np.gradient(bprof)

# Calculate FWHM of LSFs
hor_fwhm = calc_fwhm(llsf)*pixelsize[0]
ver_fwhm = calc_fwhm(blsf)*pixelsize[0]

print("Horizontal FWHM: ", np.round(hor_fwhm,2),"mm")
print("Vertical FWHM: ", np.round(ver_fwhm,2),"mm")

