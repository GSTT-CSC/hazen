"""
SNR(Im)

Calculates the SNR for a single-slice image of a uniform MRI phantom

This script utilises the smoothed subtraction method described in McCann 2013:
A quick and robust method for measurement of signal-to-noise ratio in MRI, Phys. Med. Biol. 58 (2013) 3775:3790


Created by Neil Heraghty

04/05/2018
"""
import sys

import cv2 as cv
import numpy as np
import pydicom

import hazenlib
import hazenlib.tools
import hazenlib.exceptions as exc


def two_inputs_match(dcm1: pydicom.Dataset, dcm2: pydicom.Dataset) -> bool:
    """
    Checks if two DICOMs are sufficiently similar

    Parameters
    ----------
    dcm1
    dcm2

    Returns
    -------

    """
    fields_to_match = ['StudyInstanceUID', 'RepetitionTime', 'EchoTime', 'FlipAngle']

    for field in fields_to_match:
        if dcm1.get(field) != dcm2.get(field):
            return False
    else:
        return True


def get_normalised_snr_factor(dcm: pydicom.Dataset, measured_slice_width=None) -> float:
    dx, dy = hazenlib.get_pixel_size(dcm)
    bandwidth = hazenlib.get_bandwidth(dcm)

    if measured_slice_width:
        slice_thickness = measured_slice_width
    else:
        slice_thickness = hazenlib.get_slice_thickness(dcm)
    averages = hazenlib.get_average(dcm)
    bandwidth_factor = np.sqrt((bandwidth * 256 / 2) / 1000) / np.sqrt(30)
    voxel_factor = (1 / (0.001 * dx * dy * slice_thickness))

    normalised_snr_factor = bandwidth_factor * voxel_factor * (1 / (np.sqrt(averages) * np.sqrt(256)))

    return normalised_snr_factor


def conv2d(dcm: pydicom.Dataset, f) -> np.array:
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
    a = dcm.pixel_array.astype('int')
    s = f.shape + tuple(np.subtract(a.shape, f.shape) + 1)
    strd = np.lib.stride_tricks.as_strided
    subM = strd(a, shape=s, strides=a.strides * 2)
    return np.einsum('ij,ijkl->kl', f, subM)


def smoothed_subtracted_image(dcm: pydicom.Dataset) -> np.array:
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
    a = dcm.pixel_array.astype('int')
    # Create 3x3 boxcar kernel (recommended size - adjustments will affect results)
    size = (3, 3)
    kernel = np.ones(size) / 9

    # Convolve image with boxcar kernel
    imsmoothed = conv2d(dcm, kernel)
    # Pad the array (to counteract the pixels lost from the convolution)
    imsmoothed = np.pad(imsmoothed, 1, 'minimum')
    # Subtract smoothed array from original
    imnoise = a - imsmoothed

    return imnoise


def get_roi_samples(dcm: pydicom.Dataset or np.ndarray, cx: int, cy: int) -> list:

    if type(dcm) == np.ndarray:
        data = dcm
    else:
        data = dcm.pixel_array

    sample = [None] * 5

    sample[0] = data[(cx - 10):(cx + 10), (cy - 10):(cy + 10)]
    sample[1] = data[(cx - 50):(cx - 30), (cy - 50):(cy - 30)]
    sample[2] = data[(cx + 30):(cx + 50), (cy - 50):(cy - 30)]
    sample[3] = data[(cx - 50):(cx - 10), (cy + 30):(cy + 50)]
    sample[4] = data[(cx + 30):(cx + 50), (cy + 30):(cy + 50)]

    return sample


def snr_by_smoothing(dcm: pydicom.Dataset, measured_slice_width=None) -> float:
    """

    Parameters
    ----------
    dcm
    measured_slice_width

    Returns
    -------
    normalised_snr: float

    """
    shape_detector = hazenlib.tools.ShapeDetector(arr=dcm.pixel_array)

    orientation = hazenlib.tools.get_image_orientation(dcm.ImageOrientationPatient)

    if orientation in ['Sagittal', 'Coronal']:
        # orientation is sagittal to patient
        try:
            (x, y), size, angle = shape_detector.get_shape('rectangle')
        except exc.ShapeError:
            # shape_detector.find_contours()
            # shape_detector.detect()
            # im = cv.drawContours(arr.copy(), [shape_detector.contours[0]], -1, (0, 0, 255), 2)
            # plt.imshow(im)
            # plt.show()
            # print(shape_detector.shapes.keys())
            raise
    elif orientation == 'Transverse':
        try:
            x, y, r = shape_detector.get_shape('circle')
        except exc.MultipleShapesError:
            print('Warning! Found multiple circles in image, will assume largest circle is phantom.')
            x, y, r = get_largest_circle(shape_detector.shapes['circle'])
    else:
        raise Exception("Direction must be Transverse, Sagittal or Coronal.")

    x, y = int(x), int(y)
    noise_img = smoothed_subtracted_image(dcm=dcm)

    signal = [np.mean(roi) for roi in get_roi_samples(dcm=dcm, cx=x, cy=y)]
    noise = np.divide([np.std(roi, ddof=1) for roi in get_roi_samples(dcm=noise_img, cx=x, cy=y)], np.sqrt(2))
    snr = np.mean(np.divide(signal, noise))

    normalised_snr = snr * get_normalised_snr_factor(dcm, measured_slice_width)

    return snr, normalised_snr


def get_largest_circle(circles):
    largest_r = 0
    largest_x, largest_y = 0, 0
    for circle in circles:
        (x, y), r = cv.minEnclosingCircle(circle)
        if r > largest_r:
            largest_r = r
            largest_x, largest_y = x, y

    return largest_x, largest_y, largest_r


def snr_by_subtraction(dcm1: pydicom.Dataset, dcm2: pydicom.Dataset, measured_slice_width=None) -> float:
    """

    Parameters
    ----------
    dcm1
    dcm2
    measured_slice_width

    Returns
    -------

    """
    shape_detector = hazenlib.tools.ShapeDetector(arr=dcm1.pixel_array)

    orientation = hazenlib.tools.get_image_orientation(dcm1.ImageOrientationPatient)

    if orientation in ['Sagittal', 'Coronal']:
        # orientation is sagittal to patient
        try:
            (x, y), size, angle = shape_detector.get_shape('rectangle')
        except exc.ShapeError:
            # shape_detector.find_contours()
            # shape_detector.detect()
            # im = cv.drawContours(arr.copy(), [shape_detector.contours[0]], -1, (0, 0, 255), 2)
            # plt.imshow(im)
            # plt.show()
            # print(shape_detector.shapes.keys())
            raise
    elif orientation == 'Transverse':
        try:
            x, y, r = shape_detector.get_shape('circle')
        except exc.MultipleShapesError:
            print('Warning! Found multiple circles in image, will assume largest circle is phantom.')
            x, y, r = get_largest_circle(shape_detector.shapes['circle'])
    else:
        raise Exception("Direction must be Transverse, Sagittal or Coronal.")

    x, y = int(x), int(y)
    difference = np.subtract(dcm1.pixel_array.astype('int'), dcm2.pixel_array.astype('int'))

    signal = [np.mean(roi) for roi in get_roi_samples(dcm=dcm1, cx=x, cy=y)]
    noise = np.divide([np.std(roi, ddof=1) for roi in get_roi_samples(dcm=difference, cx=x, cy=y)], np.sqrt(2))

    snr = np.mean(np.divide(signal, noise))

    normalised_snr = snr * get_normalised_snr_factor(dcm1, measured_slice_width)

    return snr, normalised_snr


def main(data: list, measured_slice_width=None) -> dict:
    """

    Parameters
    ----------
    data
    measured_slice_width

    Returns
    -------
    results: list
    """
    results = {}
    if len(data) == 2:
        snr, normalised_snr = snr_by_subtraction(data[0], data[1], measured_slice_width)
        results["measured_snr_subtraction_method"] = snr
        results["normalised_snr_subtraction_method"] = normalised_snr

    for idx, dcm in enumerate(data):
        snr, normalised_snr = snr_by_smoothing(dcm, measured_slice_width)
        results[f"measured_snr_smoothing_method_{idx}"] = snr
        results[f"normalised_snr_smoothing_method_{idx}"] = normalised_snr

    return results
    # # Draw regions for testing
    # cv.rectangle(idown, ((cenx-10), (ceny-10)), ((cenx+10), (ceny+10)), 128, 2)
    # cv.rectangle(idown, ((cenx-50), (ceny-50)), ((cenx-30), (ceny-30)), 128, 2)
    # cv.rectangle(idown, ((cenx+30), (ceny-50)), ((cenx+50), (ceny-30)), 128, 2)
    # cv.rectangle(idown, ((cenx-50), (ceny+30)), ((cenx-30), (ceny+50)), 128, 2)
    # cv.rectangle(idown, ((cenx+30), (ceny+30)), ((cenx+50), (ceny+50)), 128, 2)

    # Plot annotated image for user
    # fig = plt.figure(1)
    # plt.imshow(idown, cmap='gray')
    # plt.show()

