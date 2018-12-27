"""
Uniformity + Ghosting & Distortion

Calculates uniformity for a single-slice image of a uniform MRI phantom

This script implements the IPEM/MAGNET method of measuring fractional uniformity.
It also calculates integral uniformity using a 75% area FOV ROI and CoV for the same ROI.

This script also measures Ghosting within a single image of a uniform phantom.
This follows the guidance from ACR for testing their large phantom.

A simple measurement of distortion is also made by comparing the height and width of the circular phantom.

Created by Neil Heraghty
neil.heraghty@nhs.net

14/05/2018

"""
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
    circles = cv.HoughCircles(a, cv.HOUGH_GRADIENT, 1, 200, param1=30,
                              param2=45, minRadius=0, maxRadius=0)

    # Check that a single phantom was found
    if len(circles) == 1:
        pass
        # errCircle = "1 circle found."
    else:
        errCircle = "Wrong number of circles detected, check image."
        print(errCircle)

    # Draw circle onto original image
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # draw the outer circle
        cv.circle(a, (i[0], i[1]), i[2], 128, 2)
        # draw the center of the circle
        cv.circle(a, (i[0], i[1]), 2, 128, 2)

    cenx = i[0]
    ceny = i[1]
    cradius = i[2]
    return (cenx, ceny, cradius)


def mode(a, axis=0):
    """
        Finds the modal value of an array. From scipy.stats.mode

        parameters:
        ---------------
        a: array

        returns:
        ---------------
        mostfrequent: the modal value
        oldcounts: the number of times this value was counted (check this)
        """

    scores = np.unique(np.ravel(a))       # get ALL unique values
    testshape = list(a.shape)
    testshape[axis] = 1
    oldmostfreq = np.zeros(testshape)
    oldcounts = np.zeros(testshape)

    for score in scores:
        template = (a == score)
        counts = np.expand_dims(np.sum(template, axis),axis)
        mostfrequent = np.where(counts > oldcounts, score, oldmostfreq)
        oldcounts = np.maximum(counts, oldcounts)
        oldmostfreq = mostfrequent

    return mostfrequent, oldcounts


def main(image):
    # Read DICOM image
    #image = dcmread('uniformCNSA.dcm')     # Read the DICOM file

    # Prepare image for processing
    idata = image.pixel_array              # Read the pixel values into an array
    idata = np.array(idata)                # Make it a numpy array
    idown = ((idata / idata.max()) * 256).astype('uint8')  # Downscale to uint8 for openCV techniques

    # Find phantom centre and radius
    (cenx, ceny, cradius) = find_circle(idown)

    # Create central 10x10 ROI and measure modal value
    cenroi = idata[(ceny-5):(ceny+5),(cenx-5):(cenx+5)]
    cenroi_flat = cenroi.flatten()
    (cenmode, modenum) = mode(cenroi_flat)

    # Create 160-pixel profiles (horizontal and vertical, centred at cenx,ceny)
    horroi = idata[(ceny-5):(ceny+5),(cenx-80):(cenx+80)]
    horprof = np.mean(horroi, axis=0)
    verroi = idata[(ceny-80):(ceny+80),(cenx-5):(cenx+5)]
    verprof = np.mean(verroi, axis=1)

    # Count how many elements are within 0.9-1.1 times the modal value
    hornum = np.where(np.logical_and((horprof > (0.9*cenmode)),(horprof < (1.1*cenmode))))
    hornum = len(hornum[0])
    vernum = np.where(np.logical_and((verprof > (0.9*cenmode)),(verprof < (1.1*cenmode))))
    vernum = len(vernum[0])

    # Calculate fractional uniformity
    fract_uniformity_hor = hornum/160
    fract_uniformity_ver = vernum/160

    #print("Horizontal Fractional Uniformity: ",np.round(fract_uniformity_hor,2))
    #print("Vertical Fractional Uniformity: ",np.round(fract_uniformity_ver,2))

    # Define 75% area mask for alternative uniformity measures
    r = cradius*0.865
    n = len(idata)
    y,x = np.ogrid[:n,:n]
    dist_from_center = np.sqrt((x - cenx)**2 + (y-ceny)**2)

    mask = dist_from_center <= r

    # Calculate stats for masked region
    roimean = np.mean(idata[mask])
    roistd = np.std(idata[mask])
    roimax = np.amax(idata[mask])
    roimin = np.amin(idata[mask])

    # Calculate CoV and integral uniformity
    cov = 100*roistd/roimean
    intuniform = (1-(roimax-roimin)/(roimax+roimin))*100

    #print("CoV: ",np.round(cov,1),"%")
    #print("Integral Uniformity (ACR): ",np.round(intuniform,1),"%")

    # ------ Ghosting ------

    # Measure mean pixel value within rectangular ROIs slightly outside the top/bottom/left/right of phantom
    meantop = np.mean(idata[(cenx-50):(cenx+50),(ceny-cradius-20):(ceny-cradius-10)])
    meanbot = np.mean(idata[(cenx-50):(cenx+50),(ceny+cradius+10):(ceny+cradius+20)])
    meanl = np.mean(idata[(cenx-cradius-20):(cenx-cradius-10),(ceny-50):(ceny+50)])
    meanr = np.mean(idata[(cenx+cradius+10):(cenx+cradius+20),(ceny-50):(ceny+50)])

    # Calculate percentage ghosting
    ghosting = np.abs(((meantop+meanbot)-(meanl+meanr))/(2*roimean))*100

    #print("Ghosting: ",np.round(ghosting,1), "%")

    # ------ Distortion ------

    # Threshold using half of roimin as a cutoff
    thresh = idata < roimin/2
    ithresh = idata
    ithresh[thresh] = 0

    # Find the indices of thresholded pixels
    bbox = np.argwhere(ithresh)
    (bbystart, bbxstart), (bbystop, bbxstop) = bbox.min(0), bbox.max(0) + 1

    idistort = (bbxstop-bbxstart)/(bbystop-bbystart)
    idistort =np.abs(idistort-1)

    #print("Distortion: ", np.round(idistort,2),"%")

    return fract_uniformity_hor, fract_uniformity_ver, cov, intuniform, ghosting, idistort