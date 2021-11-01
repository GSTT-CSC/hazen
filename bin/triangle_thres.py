import numpy as np
from scipy import signal
from pydicom.filereader import dcmread
import math

def tri_thresh(image):

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

    # Read image
    image = dcmread('test_thresh.dcm')

    # Prepare image for processing
    idata = image.pixel_array  # Read the pixel values into an array
    idata = np.array(idata)  # Make it a numpy array
    idata = idata.astype(float)

    # Apply median filter (this should probably be done in the parent script)
    ifilt = signal.medfilt2d(idata, (5, 5))

    # Generate image histogram
    maxima = np.amax(ifilt)
    ihist = np.bincount(np.ravel(ifilt.astype(int)))
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

# # Threshold image
# ithresh = np.zeros(ifilt.shape)
# mask = ifilt > tri_thresh
# ithresh[mask] = 1
