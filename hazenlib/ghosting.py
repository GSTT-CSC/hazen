import os
import sys

import pydicom
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches


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
            background_rois = [(background_rois_row, background_rois[0][1] + i * gap) for i in range(4)]

    else:
        if signal_centre[1] < dcm.Columns * 0.5:
            # phantom is in left half of image
            background_rois_column = round(dcm.Columns * 0.75)  # in the right quadrant
        else:
            # phantom is right half of image
            background_rois_column = round(dcm.Columns * 0.25)  # in the top quadrant
        background_rois.append((signal_centre[0], background_rois_column))

        if signal_centre[0] >= round(dcm.Rows/2):
            # phantom is bottom half of image need 3 ROIs evenly spaced from 0->background_roi[0]
            gap = round(background_rois[0][0] / 4)
            background_rois = [(background_rois[0][0] - i * gap, background_rois_column) for i in range(4)]
        else:
            # phantom is top half of image need 3 ROIs evenly spaced from background_roi[0]->end
            gap = round((dcm.Columns - background_rois[0][0]) / 4)
            background_rois = [(background_rois[0][0] + i * gap, background_rois_column) for i in range(4)]

    return background_rois


def get_background_slices(background_rois, slice_size=10):
    slice_radius = round(slice_size / 2)
    slices = [(np.array(range(roi[0]-slice_radius, roi[0]+slice_radius), dtype=np.intp)[:, np.newaxis], np.array(
        range(roi[1]-slice_radius, roi[1]+slice_radius), dtype=np.intp))for roi in background_rois]

    return slices


def get_eligible_area(signal_bounding_box, dcm, slice_radius=5):
    upper_row, lower_row, left_column, right_column = signal_bounding_box
    padding_from_box = 30  # pixels
    if get_pe_direction(dcm) == 'ROW':
        if left_column < dcm.Columns / 2:
            # signal is in left half
            eligible_columns = range(right_column + padding_from_box, dcm.Columns - slice_radius)
            eligible_rows = range(upper_row, lower_row)
        else:
            # signal is in right half
            eligible_columns = range(slice_radius, left_column - padding_from_box)
            eligible_rows = range(upper_row, lower_row)

    else:
        if upper_row < dcm.Rows / 2:
            # signal is in top half
            eligible_rows = range(lower_row + padding_from_box, dcm.Rows - slice_radius)
            eligible_columns = range(left_column, right_column)
        else:
            # signal is in bottom half
            eligible_rows = range(slice_radius, upper_row - padding_from_box)
            eligible_columns = range(left_column, right_column)

    return eligible_rows, eligible_columns


def get_ghost_slice(signal_bounding_box, dcm, slice_size=10):
    max_mean = 0
    max_index = (0, 0)
    slice_radius = round(slice_size/2)
    windows = {}
    arr = dcm.pixel_array

    eligible_rows, eligible_columns = get_eligible_area(signal_bounding_box, dcm, slice_radius)

    for idx, centre_voxel in np.ndenumerate(arr):
        if idx[0] not in eligible_rows or idx[1] not in eligible_columns:
            continue
        else:
            windows[idx] = arr[idx[0]-slice_radius:idx[0]+slice_radius, idx[1]-slice_radius:idx[1]+slice_radius]

    for idx, window in windows.items():
        if np.mean(window) > max_mean:
            max_mean = np.mean(window)
            max_index = idx

    return np.array(
        range(max_index[0] - slice_radius, max_index[0] + slice_radius), dtype=np.intp)[:, np.newaxis], np.array(
        range(max_index[1] - slice_radius, max_index[1] + slice_radius)
    )


def get_ghosting(dcm) -> dict:
    bbox = get_signal_bounding_box(dcm.pixel_array)
    signal_centre = [bbox[0]+(bbox[1]-bbox[0])//2, bbox[2]+(bbox[3]-bbox[2])//2]
    background_rois = get_background_rois(dcm, signal_centre)
    ghost = dcm.pixel_array[get_ghost_slice(bbox, dcm)]
    phantom = dcm.pixel_array[get_signal_slice(bbox)]

    noise = np.concatenate([dcm.pixel_array[roi] for roi in get_background_slices(background_rois)])

    ghosting = calculate_ghost_intensity(ghost, phantom, noise)

    return {'ghosting_percentage': ghosting}


def main(data: list) -> dict:
    results = {f"{dcm.SeriesDescription}_{dcm.EchoTime}ms": dcm for dcm in data}  # load dicom objects into memory

    for path, dcm in results.items():
        results[path] = get_ghosting(dcm)

    return results


if __name__ == "__main__":
    main([os.path.join(sys.argv[1], i) for i in os.listdir(sys.argv[1])])
