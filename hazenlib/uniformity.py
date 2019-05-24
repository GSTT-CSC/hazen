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
import json
import sys

import numpy as np
import matplotlib.pyplot as plt
import pydicom

import hazenlib as hazen


def mode(a, axis=0):
    """
    Finds the modal value of an array. From scipy.stats.mode

    Parameters:
    ---------------
    a: array

    Returns:
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


def get_fractional_uniformity():
    pass


def get_ghosting():
    pass


def get_distortion():
    pass


def main(data: list) -> dict:

    if len(data) != 1:
        raise Exception('Only single DICOM input.')

    image = pydicom.read_file(data[0])

    arr = image.pixel_array

    # Find phantom centre and radius
    cenx, ceny, cradius = hazen.find_circle(image)

    # Create central 10x10 ROI and measure modal value
    cenroi = arr[(ceny-5):(ceny+5), (cenx-5):(cenx+5)]
    cenroi_flat = cenroi.flatten()
    (cenmode, modenum) = mode(cenroi_flat)

    # Create 160-pixel profiles (horizontal and vertical, centred at cenx,ceny)
    horroi = arr[(ceny-5):(ceny+5), (cenx-80):(cenx+80)]
    horprof = np.mean(horroi, axis=0)
    verroi = arr[(ceny-80):(ceny+80), (cenx-5):(cenx+5)]
    verprof = np.mean(verroi, axis=1)

    # Count how many elements are within 0.9-1.1 times the modal value
    hornum = np.where(np.logical_and((horprof > (0.9*cenmode)),(horprof < (1.1*cenmode))))
    hornum = len(hornum[0])
    vernum = np.where(np.logical_and((verprof > (0.9*cenmode)),(verprof < (1.1*cenmode))))
    vernum = len(vernum[0])

    # Calculate fractional uniformity
    fract_uniformity_hor = hornum/160
    fract_uniformity_ver = vernum/160

    # Define 75% area mask for alternative uniformity measures
    r = cradius*0.865
    n = len(arr)
    y,x = np.ogrid[:n,:n]
    dist_from_center = np.sqrt((x - cenx)**2 + (y-ceny)**2)

    mask = dist_from_center <= r

    # Calculate stats for masked region
    roimean = np.mean(arr[mask])
    roistd = np.std(arr[mask])
    roimax = np.amax(arr[mask])
    roimin = np.amin(arr[mask])

    # Calculate CoV and integral uniformity
    cov = 100*roistd/roimean
    intuniform = (1-(roimax-roimin)/(roimax+roimin))*100

    # ------ Ghosting ------

    # Measure mean pixel value within rectangular ROIs slightly outside the top/bottom/left/right of phantom
    meantop = np.mean(arr[(cenx-50):(cenx+50),(ceny-cradius-20):(ceny-cradius-10)])
    meanbot = np.mean(arr[(cenx-50):(cenx+50),(ceny+cradius+10):(ceny+cradius+20)])
    meanl = np.mean(arr[(cenx-cradius-20):(cenx-cradius-10),(ceny-50):(ceny+50)])
    meanr = np.mean(arr[(cenx+cradius+10):(cenx+cradius+20),(ceny-50):(ceny+50)])

    # Calculate percentage ghosting
    ghosting = np.abs(((meantop+meanbot)-(meanl+meanr))/(2*roimean))*100

    # ------ Distortion ------

    # Threshold using half of roimin as a cutoff
    thresh = arr < roimin/2
    ithresh = arr
    ithresh[thresh] = 0

    # Find the indices of thresholded pixels
    bbox = np.argwhere(ithresh)
    (bbystart, bbxstart), (bbystop, bbxstop) = bbox.min(0), bbox.max(0) + 1

    idistort = (bbxstop-bbxstart)/(bbystop-bbystart)
    idistort =np.abs(idistort-1)

    results = {

        'uniformity':
            {
                'horizontal':
                    {
                        'IPEM': fract_uniformity_hor
                    },
                'vertical':
                    {
                        'IPEM': fract_uniformity_ver
                    }

            }
    }

    return json.dumps(results, indent=4)


if __name__ == "__main__":
    main([sys.argv[1]])
