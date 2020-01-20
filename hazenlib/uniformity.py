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
import sys

import numpy as np
import pydicom
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2 as cv

import hazenlib.tools
import hazenlib.exceptions as exc


def mode(a, axis=0):
    """
    Finds the modal value of an array. From scipy.stats.mode

    Parameters:
    ---------------
    a: array

    Returns:
    ---------------
    most_frequent: the modal value
    old_counts: the number of times this value was counted (check this)
    """
    scores = np.unique(np.ravel(a))  # get ALL unique values
    test_shape = list(a.shape)
    test_shape[axis] = 1
    old_most_frequent = np.zeros(test_shape)
    old_counts = np.zeros(test_shape)
    most_frequent = None

    for score in scores:
        template = (a == score)
        counts = np.expand_dims(np.sum(template, axis), axis)
        most_frequent = np.where(counts > old_counts, score, old_most_frequent)
        old_counts = np.maximum(counts, old_counts)
        old_most_frequent = most_frequent

    return most_frequent, old_counts


def get_fractional_uniformity(dcm):

    arr = dcm.pixel_array
    shape_detector = hazenlib.tools.ShapeDetector(arr=arr)
    orientation = hazenlib.tools.get_image_orientation(dcm.ImageOrientationPatient)

    if orientation in ['Sagittal', 'Coronal']:
        # orientation is sagittal to patient
        try:
            (x, y), size, angle = shape_detector.get_shape('rectangle')
        except exc.ShapeError as e:
            shape_detector.find_contours()
            shape_detector.detect()
            im = cv.drawContours(arr.copy(), [shape_detector.contours[0]], -1, (0, 0, 255), 2)
            plt.imshow(im)
            plt.show()
            print(shape_detector.shapes.keys())
            raise

        x, y = int(x), int(y)
        central_roi = arr[(y - 5):(y + 5), (x - 5):(x + 5)].flatten()

    elif orientation == 'Transverse':
        # orientation is axial
        x, y, r = shape_detector.get_shape('circle')
        x, y = int(x), int(y)
        central_roi = arr[(y - 5):(y + 5), (x - 5):(x + 5)].flatten()
    else:
        raise Exception("Direction must be Transverse, Sagittal or Coronal.")

    # Create central 10x10 ROI and measure modal value

    central_roi_mode, mode_popularity = mode(central_roi)

    # Create 160-pixel profiles (horizontal and vertical, centred at x,y)
    horizontal_roi = arr[(y - 5):(y + 5), (x - 80):(x + 80)]
    horizontal_profile = np.mean(horizontal_roi, axis=0)
    vertical_roi = arr[(y - 80):(y + 80), (x - 5):(x + 5)]
    vertical_profile = np.mean(vertical_roi, axis=1)

    # Count how many elements are within 0.9-1.1 times the modal value
    horizontal_count = np.where(np.logical_and((horizontal_profile > (0.9 * central_roi_mode)), (horizontal_profile < (
            1.1 * central_roi_mode))))
    horizontal_count = len(horizontal_count[0])
    vertical_count = np.where(np.logical_and((vertical_profile > (0.9 * central_roi_mode)), (vertical_profile < (
            1.1 * central_roi_mode))))
    vertical_count = len(vertical_count[0])

    # Calculate fractional uniformity
    fractional_uniformity_horizontal = horizontal_count / 160
    fractional_uniformity_vertical = vertical_count / 160

    # Define 75% area mask for alternative uniformity measures
    # r = r * 0.865
    # n = len(arr)
    # y, x = np.ogrid[:n, :n]
    # dist_from_center = np.sqrt((x - x) ** 2 + (y - y) ** 2)
    #
    # mask = dist_from_center <= r
    #
    # # Calculate stats for masked region
    # roi_mean = np.mean(arr[mask])
    # roi_std = np.std(arr[mask])
    # roi_max = np.amax(arr[mask])
    # roi_min = np.amin(arr[mask])
    #
    # # Calculate CoV and integral uniformity
    # cov = 100 * roi_std / roi_mean
    # int_uniform = (1 - (roi_max - roi_min) / (roi_max + roi_min)) * 100

    return {'uniformity': {'horizontal': {'IPEM': fractional_uniformity_horizontal},
                           'vertical': {'IPEM': fractional_uniformity_vertical}}}


# def get_ghosting(arr, c, roi_mean):
#     x, y, r = c
#     # Measure mean pixel value within rectangular ROIs slightly outside the top/bottom/left/right of phantom
#     mean_top = np.mean(arr[(x - 50):(x + 50), (y - r - 20):(y - r - 10)])
#     mean_bottom = np.mean(arr[(x - 50):(x + 50), (y + r + 10):(y + r + 20)])
#     mean_left = np.mean(arr[(x - r - 20):(x - r - 10), (y - 50):(y + 50)])
#     mean_right = np.mean(arr[(x + r + 10):(x + r + 20), (y - 50):(y + 50)])
#
#     # Calculate percentage ghosting
#     ghosting = np.abs(((mean_top + mean_bottom) - (mean_left + mean_right)) / (2 * roi_mean)) * 100
#     return ghosting
#
#
# def get_distortion(arr, roi_min):
#     # ------ Distortion ------
#
#     # Threshold using half of roimin as a cutoff
#     thresh = arr < roi_min / 2
#     i_thresh = arr
#     i_thresh[thresh] = 0
#
#     # Find the indices of thresholded pixels
#     bbox = np.argwhere(i_thresh)
#     (bby_start, bbx_start), (bby_stop, bbx_stop) = bbox.min(0), bbox.max(0) + 1
#
#     i_distort = (bbx_stop - bbx_start) / (bby_stop - bby_start)
#     i_distort = np.abs(i_distort - 1)
#     return i_distort


def main(data: list) -> dict:
    if len(data) != 1:
        raise Exception('Only single DICOM input.')

    dcm = pydicom.read_file(data[0])

    results = get_fractional_uniformity(dcm)
    return results


if __name__ == "__main__":
    main([sys.argv[1]])
