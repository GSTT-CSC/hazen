# mr_coil_qa
#
# This script carries out QA tests on images from MR coil tests.
#
# The script expects to find DICOM images within the target folder. The images
# should contain a uniform test phantom. An ROI is positioned within the phantom (halfway
# between the centre and the brightest region), which is used to determine SNR using the
# smoothed subtraction method.
#
# This SNR value, as well as metadata for the image being analysed, are then written into
# an SQL database.
#
#
# Neil Heraghty
# neil.heraghty@nhs.net
#
# First version 12/07/2018

import numpy as np
import os
from scipy import signal
from pydicom.filereader import dcmread
import math
import csv


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


def coilsnr(a):
    """
    Places an ROI in the brightest region and calculates the SNR

    parameters:
    ---------------
    a: image from dcmread

    returns:
    ---------------
    snr: the signal to noise ratio found within the selected ROI
    """

    # Prepare image for processing
    idata = a.pixel_array  # Read the pixel values into an array
    idata = np.array(idata)  # Make it a numpy array
    idata = idata.astype(float)

    # Create noise image
    [imnoise, imsmoothed] = image_noise(idata)

    # Find brightest region of image
    [ymax, xmax] = np.unravel_index(imsmoothed.argmax(), imsmoothed.shape)

    # Apply median filter
    ifilt = signal.medfilt2d(idata, (5, 5))

    # Threshold image
    thresh = tri_thresh(ifilt)  # calculate image threshold
    ithresh = np.zeros(ifilt.shape)
    mask = ifilt > thresh
    ithresh[mask] = 1

    # Find centre of mass of thresholded image
    count = (ithresh == 1).sum()
    y_center, x_center = np.argwhere(ithresh == 1).sum(0) / count

    # Calculate midpoint between phantom centre and brightest point
    y_mid = (np.round((y_center + ymax) / 2)).astype(int)
    x_mid = (np.round((x_center + xmax) / 2)).astype(int)

    # Measure mean signal and stdev noise in a 20x20 ROI at this location
    sig = np.mean(idata[(y_mid - 10):(y_mid + 10), (x_mid - 10):(x_mid + 10)])
    noise = np.std(imnoise[(y_mid - 10):(y_mid + 10), (x_mid - 10):(x_mid + 10)])

    # Calculate SNR
    sig2noise = sig / noise

    return round(sig2noise, 3)


def find_nth(haystack, needle, n):
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start+len(needle))
        n -= 1
    return start


# Select directory (currently hard-coded - needs results in a fixed location)
os.chdir("E:/Neil's Data/Neil's Work/Medical Physics/Elective/Routine QC Data/sn3")

# Read in DICOM filenames within this directory
ext = ('.dcm','.IMA')
imagelist = [i for i in os.listdir('.') if i.endswith(ext)]

# Define results lists
scanner = [None]*len(imagelist)
manufacturer = [None]*len(imagelist)
date = [None]*len(imagelist)
time = [None]*len(imagelist)
site = [None]*len(imagelist)
freq = [None]*len(imagelist)
sequence = [None]*len(imagelist)
tr = [None]*len(imagelist)
te = [None]*len(imagelist)
averages = [None]*len(imagelist)
slice_thickness = [None]*len(imagelist)
matrix_length = [None]*len(imagelist)
flip_angle = [None]*len(imagelist)
slice_measurement_duration = [None]*len(imagelist)
slice_location = [None]*len(imagelist)
coil_name = [None]*len(imagelist)
element_ID = [None]*len(imagelist)
snr = [None]*len(imagelist)

# Read out data for each image in turn
for entry in range(len(imagelist)):

    # Read image
    image = dcmread(imagelist[entry])

    # Record some metadata
    scanner[entry] = getattr(image, 'StationName', 'n/a')
    manufacturer[entry] = getattr(image, 'Manufacturer', 'n/a')
    date[entry] = getattr(image, 'SeriesDate', 'n/a')
    time[entry] = getattr(image, 'SeriesTime', 'n/a')
    site[entry] = getattr(image, 'InstitutionName', 'n/a')
    freq[entry] = getattr(image, 'ImagingFrequency', 'n/a')
    sequence[entry] = getattr(image, 'SequenceName', 'n/a')
    tr[entry] = getattr(image, 'RepetitionTime', 'n/a')
    te[entry] = getattr(image, 'EchoTime', 'n/a')
    averages[entry] = getattr(image, 'NumberOfAverages', 'n/a')
    slice_thickness[entry] = getattr(image, 'SliceThickness', 'n/a')
    matrix_length[entry] = getattr(image, 'Columns', 'n/a')
    flip_angle[entry] = getattr(image, 'FlipAngle', 'n/a')
    slice_location[entry] = getattr(image, 'SliceLocation', 'n/a')
    if (0x0019, 0x100b) in image:
        slice_measurement_duration[entry] = image[0x0019, 0x100b].value
    else:
        slice_measurement_duration[entry] = 'n/a'

    # Read some manufacturer-specific metadata
    if manufacturer[entry] == 'SIEMENS':
        if (0x0051, 0x100f) in image:
            element_ID[entry] = image[0x0051, 0x100f].value
        else:
            element_ID[entry] = 'n/a'

        with open(imagelist[entry], 'rb') as file:
            text_dump = file.read()
        coil_loc = find_nth(text_dump, b'\x74\x43\x6f\x69\x6c\x49\x44', 4)  # Find the 4th entry of "tCoilID"
        coil_name[entry] = text_dump[(coil_loc+13):(coil_loc+29)]

    # Analyse the image and calculate the coil element SNR
    snr[entry] = coilsnr(image)

# Write the recorded data into a csv file

# First, check if results file exists and, if not, create one
if os.path.isfile('coil_snr_results.csv'):
    pass
else:
    with open('coil_snr_results.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Filename', 'Scanner ID', 'Manufacturer', 'Site Name', 'Date', 'Time',
                        'Frequency/MHz', 'Sequence', 'TR/ms', 'TE/ms', 'Averages',
                         'Slice Thickness/mm', 'Matrix Length', 'Flip Angle', 'Slice Location',
                         'Slice Measurement Duration/ms', 'Coil Name', 'Element ID',
                         'Coil Element SNR'])

# Now, write data into CSV file
with open('coil_snr_results.csv', 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for i in range(len(imagelist)):
        writer.writerow([imagelist[i], scanner[i], manufacturer[i], site[i], date[i], time[i],
                         freq[i], sequence[i], tr[i], te[i], averages[i], slice_thickness[i],
                         matrix_length[i], flip_angle[i], slice_location[i], slice_measurement_duration[i],
                         coil_name[i], element_ID[i], snr[i]])
