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
    return True


def get_normalised_snr_factor(dcm: pydicom.Dataset, measured_slice_width=None) -> float:

    """
    Calculates SNR normalisation factor. Method matches MATLAB script.
    Utilises user provided slice_width if provided. Else finds from dcm.
    Finds dx, dy and bandwidth from dcm.
    Seeks to find TR, image columns and rows from dcm. Else uses default values.

    Parameters
    ----------
    dcm, measured_slice_width

    Returns
    -------
    normalised snr factor: float

    """

    dx, dy = hazenlib.get_pixel_size(dcm)
    bandwidth = hazenlib.get_bandwidth(dcm)
    TR=hazenlib.get_TR(dcm)
    rows = hazenlib.get_rows(dcm)
    columns = hazenlib.get_columns(dcm)

    if measured_slice_width:
        slice_thickness = measured_slice_width
    else:
        slice_thickness = hazenlib.get_slice_thickness(dcm)

    averages = hazenlib.get_average(dcm)
    bandwidth_factor = np.sqrt((bandwidth * columns / 2) / 1000) / np.sqrt(30)
    voxel_factor = (1 / (0.001 * dx * dy * slice_thickness))

    normalised_snr_factor = bandwidth_factor * voxel_factor * (1 / (np.sqrt(averages*rows*(TR/1000))))

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
    # Create 3x3 boxcar kernel
    # size = (3, 3)
    # kernel = np.ones(size) / 9

    # implementing 9 x 9 kernel to match matlab (and McCann 2013 for Head Coil, although note McCann 2013 recommends 25 x 25 for Body Coil)
    size = (9, 9)
    kernel = np.ones(size) / 81

    # Convolve image with boxcar kernel
    imsmoothed = conv2d(dcm, kernel)
    # Pad the array (to counteract the pixels lost from the convolution)
    # imsmoothed = np.pad(imsmoothed, 1, 'minimum')

    # changed padding to align with new kernel size - more padding required

    imsmoothed = np.pad(imsmoothed, 4, 'minimum')

    # Subtract smoothed array from original
    imnoise = a - imsmoothed

    return imnoise


def get_roi_samples(ax, dcm: pydicom.Dataset or np.ndarray, centre_col: int, centre_row: int) -> list:

    if type(dcm) == np.ndarray:
        data = dcm
    else:
        data = dcm.pixel_array

    sample = [None] * 5
    #for array indexing: [row, column] format
    sample[0] = data[(centre_row - 10):(centre_row + 10), (centre_col - 10):(centre_col + 10)]
    sample[1] = data[(centre_row - 50):(centre_row - 30), (centre_col - 50):(centre_col - 30)]
    sample[2] = data[(centre_row + 30):(centre_row + 50), (centre_col - 50):(centre_col - 30)]
    sample[3] = data[(centre_row - 50):(centre_row - 30), (centre_col + 30):(centre_col + 50)]
    sample[4] = data[(centre_row + 30):(centre_row + 50), (centre_col + 30):(centre_col + 50)]

    if ax:

        from matplotlib.patches import Rectangle
        from matplotlib.collections import PatchCollection
        #for patches: [column/x, row/y] format

        rects = [Rectangle((centre_col - 10, centre_row - 10), 20, 20),
                 Rectangle((centre_col - 50, centre_row - 50), 20, 20),
                 Rectangle((centre_col + 30, centre_row - 50), 20, 20),
                 Rectangle((centre_col - 50, centre_row + 30), 20, 20),
                 Rectangle((centre_col + 30, centre_row + 30), 20, 20)]
        pc = PatchCollection(rects, edgecolors='red', facecolors="None", label='ROIs')
        ax.add_collection(pc)

    return sample


def get_object_centre(dcm) -> (int, int):
    """
    Find the phantom object within the image and returns its centre col and row value. Note first element in output = col, second = row.

    Args:
        dcm:

    Returns:
        centre: (col:int, row:int)

    """

    shape_detector = hazenlib.tools.ShapeDetector(arr=dcm.pixel_array)
    orientation = hazenlib.tools.get_image_orientation(dcm.ImageOrientationPatient)

    if orientation in ['Sagittal', 'Coronal']:
        # orientation is sagittal to patient
        try:
            (col, row), size, angle = shape_detector.get_shape('rectangle')
        except exc.ShapeError:
            # shape_detector.find_contours()
            # shape_detector.detect()
            # contour = shape_detector.shapes['rectangle'][1]
            # angle, centre, size = cv.minAreaRect(contour)
            # print((angle, centre, size))
            # im = cv.drawContours(dcm.pixel_array.copy(), [shape_detector.contours[0]], -1, (0, 255, 255), 10)
            # plt.imshow(im)
            # plt.savefig("rectangles.png")
            # print(shape_detector.shapes.keys())
            raise
    elif orientation == 'Transverse':
        try:
            col, row, r = shape_detector.get_shape('circle')
        except exc.MultipleShapesError:
            print('Warning! Found multiple circles in image, will assume largest circle is phantom.')
            col, row, r = get_largest_circle(shape_detector.shapes['circle'])
    else:
        raise Exception("Direction must be Transverse, Sagittal or Coronal.")

    return int(col), int(row)


def snr_by_smoothing(dcm: pydicom.Dataset, measured_slice_width=None, report_path=False) -> float:
    """

    Parameters
    ----------
    dcm
    measured_slice_width
    report_path

    Returns
    -------
    normalised_snr: float

    """
    col, row = get_object_centre(dcm=dcm)
    noise_img = smoothed_subtracted_image(dcm=dcm)

    signal = [np.mean(roi) for roi in get_roi_samples(ax=None, dcm=dcm, centre_col=col, centre_row=row)]

    noise = [np.std(roi, ddof=1) for roi in get_roi_samples(ax=None, dcm=noise_img, centre_col=col, centre_row=row)]
    # note no root_2 factor in noise for smoothed subtraction (one image) method, replicating Matlab approach and McCann 2013

    snr = np.mean(np.divide(signal, noise))

    normalised_snr = snr * get_normalised_snr_factor(dcm, measured_slice_width)

    if report_path:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 1)
        fig.set_size_inches(5, 5)
        fig.tight_layout(pad=1)

        axes.set_title('smoothed noise image')
        axes.imshow(noise_img, cmap='gray', label='smoothed noise image')
        axes.scatter(col, row, 10, marker="+", label='centre')
        get_roi_samples(axes, dcm, col, row)
        axes.legend()

        fig.savefig(report_path + ".png")

    return snr, normalised_snr

def get_largest_circle(circles):
    largest_r = 0
    largest_col, largest_row = 0, 0
    for circle in circles:
        (col, row), r = cv.minEnclosingCircle(circle)
        if r > largest_r:
            largest_r = r
            largest_col, largest_row = col, row

    return largest_col, largest_row, largest_r


def snr_by_subtraction(dcm1: pydicom.Dataset, dcm2: pydicom.Dataset, measured_slice_width=None, report_path=False) -> float:
    """

    Parameters
    ----------
    dcm1
    dcm2
    measured_slice_width
    report_path

    Returns
    -------

    """
    col, row = get_object_centre(dcm=dcm1)

    difference = np.subtract(dcm1.pixel_array.astype('int'), dcm2.pixel_array.astype('int'))

    signal = [np.mean(roi) for roi in get_roi_samples(ax=None, dcm=dcm1, centre_col=col, centre_row=row)]
    noise = np.divide([np.std(roi, ddof=1) for roi in get_roi_samples(ax=None, dcm=difference, centre_col=col, centre_row=row)], np.sqrt(2))
    snr = np.mean(np.divide(signal, noise))

    normalised_snr = snr * get_normalised_snr_factor(dcm1, measured_slice_width)

    if report_path:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 1)
        fig.set_size_inches(5, 5)
        fig.tight_layout(pad=1)

        axes.set_title('difference image')
        axes.imshow(difference, cmap='gray', label='difference image')
        axes.scatter(col, row, 10, marker="+", label='centre')
        get_roi_samples(axes, dcm1, col, row)
        axes.legend()

        fig.savefig(report_path + ".png")

    return snr, normalised_snr


def main(data: list, measured_slice_width=None, report_path=False) -> dict:
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
        key = f"{data[0].SeriesDescription}_{data[0].SeriesNumber}_{data[0].InstanceNumber}"
        if report_path:
            report_path = key
        snr, normalised_snr = snr_by_subtraction(data[0], data[1], measured_slice_width, report_path)
        results[f"snr_subtraction_measured_{key}"] = round(snr, 2)
        results[f"snr_subtraction_normalised_{key}"] = round(normalised_snr, 2)

    for idx, dcm in enumerate(data):
        try:
            key = f"{dcm.SeriesDescription}_{dcm.SeriesNumber}_{dcm.InstanceNumber}"
        except AttributeError as e:
            print(e)
            key = f"{dcm.SeriesDescription}_{dcm.SeriesNumber}"

        if report_path:
            report_path = key

        snr, normalised_snr = snr_by_smoothing(dcm, measured_slice_width, report_path)
        results[f"snr_smoothing_measured_{key}"] = round(snr, 2)
        results[f"snr_smoothing_normalised_{key}"] = round(normalised_snr, 2)

    return results


