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
import skimage.filters
from scipy import ndimage
from skimage import filters

import hazenlib
import hazenlib.tools
import hazenlib.exceptions as exc
from hazenlib.logger import logger


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


def filtered_image(dcm: pydicom.Dataset) -> np.array:
    """
    Performs a 2D convolution (for filtering images)
    uses uniform_filter SciPy function

    parameters:
    ---------------
    a: array to be filtered

    returns:
    ---------------
    filtered numpy array
    """
    a = dcm.pixel_array.astype('int')

    # filter size = 9, following MATLAB code and McCann 2013 paper for head coil, although note McCann 2013 recommends 25x25 for body coil.
    filtered_array=ndimage.uniform_filter(a,9,mode='constant')
    return filtered_array


def get_noise_image(dcm: pydicom.Dataset) -> np.array:
    """
    Separates the image noise by smoothing the image and subtracting the smoothed image
    from the original.

    parameters:
    ---------------
    a: image array from dcmread and .pixelarray

    returns:
    ---------------
    Imnoise: image representing the image noise
    """
    a = dcm.pixel_array.astype('int')

    # Convolve image with boxcar/uniform kernel
    imsmoothed = filtered_image(dcm)


    # Subtract smoothed array from original
    imnoise = a - imsmoothed

    return imnoise


def threshold_image(dcm: pydicom.Dataset):
    """
    Threshold images

    parameters:
    ---------------
    a: image array from dcmread and .pixelarray

    returns:
    ---------------
    imthresholded: thresholded image
    mask: threshold mask
    """
    a = dcm.pixel_array.astype('int')

    threshold_value = skimage.filters.threshold_li(a)  # threshold_li: Pixels > this value are assumed foreground
    # print('threshold_value =', threshold_value)
    mask = a > threshold_value
    imthresholded = np.zeros_like(a)
    imthresholded[mask] = a[mask]

    # # For debugging: Threshold figures:
    # from matplotlib import pyplot as plt
    # plt.figure()
    # fig, ax = plt.subplots(2, 2)
    # ax[0, 0].imshow(a)
    # ax[0, 1].imshow(mask)
    # ax[1, 0].imshow(imthresholded)
    # ax[1, 1].imshow(a-imthresholded)
    # fig.savefig("../THRESHOLD.png")

    return imthresholded, mask


def get_binary_mask_centre(binary_mask) -> (int, int):
    """
    Return centroid coordinates of binary polygonal shape

    parameters:
    ---------------
    binary_mask: mask of a shape

    returns:
    ---------------
    centroid_coords: (col:int, row:int)
    """

    from skimage import util
    from skimage.measure import label, regionprops
    img = util.img_as_ubyte(binary_mask) > 0
    label_img = label(img, connectivity=img.ndim)
    props = regionprops(label_img)
    col = int(props[0].centroid[0])
    row = int(props[0].centroid[1])
    # print('Centroid coords [x,y] =', col, row)

    return int(col), int(row)


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

    # Shape Detection
    try:
        logger.debug('Performing phantom shape detection.')
        shape_detector = hazenlib.tools.ShapeDetector(arr=dcm.pixel_array)
        orientation = hazenlib.tools.get_image_orientation(dcm.ImageOrientationPatient)

        if orientation in ['Sagittal', 'Coronal']:
            logger.debug('Orientation = sagittal or coronal.')
            # orientation is sagittal to patient
            try:
                (col, row), size, angle = shape_detector.get_shape('rectangle')
            except exc.ShapeError as e:
                # shape_detector.find_contours()
                # shape_detector.detect()
                # contour = shape_detector.shapes['rectangle'][1]
                # angle, centre, size = cv.minAreaRect(contour)
                # print((angle, centre, size))
                # im = cv.drawContours(dcm.pixel_array.copy(), [shape_detector.contours[0]], -1, (0, 255, 255), 10)
                # plt.imshow(im)
                # plt.savefig("rectangles.png")
                # print(shape_detector.shapes.keys())
                raise e
        elif orientation == 'Transverse':
            logger.debug('Orientation = transverse.')
            try:
                col, row, r = shape_detector.get_shape('circle')
            except exc.MultipleShapesError:
                logger.info('Warning! Found multiple circles in image, will assume largest circle is phantom.')
                col, row, r = get_largest_circle(shape_detector.shapes['circle'])
        else:
            raise exc.ShapeError("Unable to identify phantom shape.")

    # Threshold Detection
    except exc.ShapeError:
        logger.info('Shape detection failed. Performing object centre measurement by thresholding.')
        _, mask = threshold_image(dcm)
        row, col = get_binary_mask_centre(mask)

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
    noise_img = get_noise_image(dcm=dcm)

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
    report_path

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



