import os
import sys

import pydicom
import numpy as np
import matplotlib
# matplotlib.use("agg")
import matplotlib.pyplot as plt


def calculate_ghost_intensity(ghost, phantom, noise) -> float:
    """
    Calculates the ghost intensity using the formula from IPEM Report 112
    Ghosting = (Sg-Sn)/(Sp-Sn) x 100%

    Returns:    :float

    References: IPEM Report 112 - Small Bottle Method
                MagNET

    """

    if ghost is None or phantom is None or noise is None:
        raise Exception(f"At least one of ghost, phantom and noise ROIs is empty or null")

    if type(ghost) is not np.ndarray:
        raise Exception(f"Ghost, phantom and noise ROIs must be of type numpy.ndarray")

    ghost_mean = np.mean(ghost)
    phantom_mean = np.mean(phantom)
    noise_mean = np.mean(noise)

    if phantom_mean < ghost_mean or phantom_mean < noise_mean:
        raise Exception(f"The mean phantom signal is lower than the ghost or the noise signal. This can't be the case ")

    return 100 * (ghost_mean - noise_mean) / phantom_mean


def get_signal_bounding_box(array: np.ndarray):
    max_signal = np.max(array)
    signal_limit = max_signal * 0.5  # assumes phantom signal is at least 50% of the max signal inside the phantom
    signal = []
    for idx, voxel in np.ndenumerate(array):
        if voxel > signal_limit:
            signal.append(idx)

    signal_row = sorted([voxel[0] for voxel in signal])
    signal_column = sorted([voxel[1] for voxel in signal])

    upper_row = min(signal_row) - 1  # minus 1 to get the box that CONTAINS the signal
    lower_row = max(signal_row) + 1  # ditto for add one
    left_row = min(signal_column) - 1  # ditto
    right_row = max(signal_column) + 1  # ditto

    return upper_row, lower_row, left_row, right_row


def get_signal_slice(bounding_box, slice_size=10):
    slice_radius = round(slice_size / 2)
    upper_row, lower_row, left_column, right_column = bounding_box
    centre_row = upper_row + round((lower_row - upper_row) / 2)
    centre_column = left_column + round((right_column - left_column) / 2)

    idxs = (np.array(range(centre_row - slice_radius, centre_row + slice_radius), dtype=np.intp)[:, np.newaxis],
            np.array(range(centre_column - slice_radius, centre_column + slice_radius), dtype=np.intp))
    return idxs


def get_pe_direction(dcm):
    return dcm.InPlanePhaseEncodingDirection


def get_background_rois(dcm, signal_centre):
    background_rois = []

    if get_pe_direction(dcm) == 'ROW':
        # phase encoding is left -right i.e. increases with columns
        if signal_centre[0] < dcm.Rows * 0.5:
            # phantom is in top half of image
            background_rois_row = round(dcm.Rows * 0.75)  # in the bottom quadrant
        else:
            # phantom is bottom half of image
            background_rois_row = round(dcm.Rows * 0.25)  # in the top quadrant
        background_rois.append((background_rois_row, signal_centre[1]))

        if signal_centre[1] >= round(dcm.Columns/2):
            # phantom is right half of image need 3 ROIs evenly spaced from 0->background_roi[0]
            gap = round(background_rois[0][1]/ 4)
            background_rois = [(background_rois_row, background_rois[0][1] - i * gap) for i in range(4)]
        else:
            # phantom is left half of image need 3 ROIs evenly spaced from background_roi[0]->end
            gap = round((dcm.Columns - background_rois[0][1]) / 4)
            background_rois = [(background_rois_row, background_rois[0][1] - i * gap) for i in range(4)]

    else:
        if signal_centre[1] < dcm.Columns * 0.5:
            # phantom is in left half of image
            background_rois_column = round(dcm.Columns * 0.75)  # in the right quadrant
        else:
            # phantom is right half of image
            background_rois_column = round(dcm.Columns * 0.25)  # in the top quadrant
        background_rois.append((signal_centre[0], background_rois_column))

    return background_rois


def get_background_slices(background_rois, slice_size=10):
    slice_radius = round(slice_size / 2)
    slices = [(np.array(range(roi[0]-slice_radius, roi[0]+slice_radius), dtype=np.intp)[:, np.newaxis], np.array(
        range(roi[1]-slice_radius, roi[1]+slice_radius), dtype=np.intp))for roi in background_rois]
    return slices


def get_ghosting(dicom_data: list) -> dict:
    return {}


def main(data: list) -> dict:
    data = [pydicom.read_file(dcm) for dcm in data]  # load dicom objects into memory

    results = get_ghosting(data)

    return results


if __name__ == "__main__":
    main([os.path.join(sys.argv[1], i) for i in os.listdir(sys.argv[1])])
