"""

Local Otsu thresholding
http://scikit-image.org/docs/0.11.x/auto_examples/plot_local_otsu.html

"""
from hazenlib.logger import logger
import sys
import os
import traceback

import pydicom
from skimage import measure, filters
import numpy as np
import cv2 as cv

import hazenlib
import hazenlib.tools
import hazenlib.exceptions


def get_rod_rotation(x_pos: list, y_pos: list) -> float:
    """
    Determine the in-plane rotation i.e. the x-position of the rods should not vary with y-position. If they do it's
    because the phantom is rotated in-plane.

    We can determine the angle of in-plane rotation by plotting the x-position against the y-position. We then fit a
    straight line through the points.
    arctan (gradient) is the angle of rotation.

    If y=c+mx, we can formulate the straight line fit matrix problem, y=X*beta where y is the x-position (because if I
    set y to be the y-position the fit isn't very good because the x-position hardly varies ), X is the two column
    design matrix, the first  column is a constant and the second column are the y-positions.

    Parameters
    ----------
    x_pos: int
        x co-ordinate of a rod
    y_pos: int
        y co-ordinate of a rod

    Returns
    -------
    theta: float
        angle of rotation in degrees

    """
    X = np.array([[i, 1] for i in y_pos])

    m, c = np.linalg.lstsq(X, np.array(x_pos), rcond=None)[0]

    theta = np.arctan(m)
    return theta


def get_rods_coords(dcm: pydicom.Dataset):
    shape_detector = hazenlib.tools.ShapeDetector(arr=dcm.pixel_array)
    try:
        x, y, r = shape_detector.get_shape('circle')

    except hazenlib.exceptions.MultipleShapesError as e:
        # logger.info(f'Warning: found multiple shapes: {list(shape_detector.shapes.keys())}')
        shape_detector.find_contours()
        shape_detector.detect()
        x, y, r = 0, 0, 0
        for contour in shape_detector.shapes['circle']:
            (new_x, new_y), new_r = cv.minEnclosingCircle(contour)
            if new_r > r:
                # logger.info(f"Found bigger circle: {new_x}, {new_y}, {new_r}")
                x, y, r = new_x, new_y, new_r
            # logger.info(f"Found circle with x={x},y={y},r={r}")

    except hazenlib.exceptions.ShapeError:
        raise

    x, y = int(x), int(y)

    # clip image in xy plane to only include regions where rods could be
    x_window = int(r/4)
    y_window = int(r*0.95)

    arr = dcm.pixel_array
    clipped = np.zeros_like(arr)
    clipped[y-y_window:y+y_window, x-x_window:x+x_window] = arr[y-y_window:y+y_window, x-x_window:x+x_window]

    threshold = filters.threshold_otsu(clipped, 2)

    clipped_thresholded = clipped <= threshold # binarise using otsu threshold

    labels, num = measure.label(clipped_thresholded, return_num=True)
    measured_objects = measure.regionprops(label_image=labels)

    rods = []
    for obj in measured_objects:
        if 5 < obj.bbox_area < 25:
            rods.append(obj)

    if len(rods) != 2:
        raise Exception(f'Found {len(rods)} rods instead of 2.')

    rods.sort(key=lambda x: x.centroid[1])  # sort into Left and Right by using second coordinate

    ly, lx = rods[0].centroid
    ry, rx = rods[1].centroid

    # fig = plt.figure(1)
    # plt.imshow(labels)
    # plt.show()

    return lx, ly, rx, ry


def get_rods(data: list):

    left_rod, right_rod = {'x_pos': [], 'y_pos': []}, {'x_pos': [], 'y_pos': []}
    nominal_positions = []
    for i, dcm in enumerate(data):

        nominal_positions.append((i + 10) * dcm.SpacingBetweenSlices)

        lx, ly, rx, ry = get_rods_coords(dcm)

        left_rod['x_pos'].append(lx)
        left_rod['y_pos'].append(ly)
        right_rod['x_pos'].append(rx)
        right_rod['y_pos'].append(ry)
        # img = dcm.pixel_array
        # cv2.circle(img, (lx, ly), 5, color=(0, 255, 0))
        # cv2.circle(img, (rx, ry), 5, color=(0, 255, 0))
        # fig = plt.figure(1)
        # plt.imshow(img, cmap='gray')
        # plt.show()

    return left_rod, right_rod, nominal_positions


def correct_rods_for_rotation(left_rod: dict, right_rod: dict) -> (dict, dict):
    """

    Parameters
    ----------
    left_rod
    right_rod

    Returns
    -------

    """
    r_theta = get_rod_rotation(x_pos=right_rod['x_pos'], y_pos=right_rod['y_pos'])
    l_theta = get_rod_rotation(x_pos=left_rod['x_pos'], y_pos=left_rod['y_pos'])
    theta = np.mean([r_theta, l_theta])

    left_rod['x_pos'] = np.subtract(np.multiply(np.cos(theta), left_rod['x_pos']),
                                    np.multiply(np.sin(theta), left_rod['y_pos']))

    left_rod['y_pos'] = np.add(np.multiply(np.sin(theta), left_rod['x_pos']),
                               np.multiply(np.cos(theta), left_rod['y_pos']))

    right_rod['x_pos'] = np.subtract(np.multiply(np.cos(theta), right_rod['x_pos']),
                                     np.multiply(np.sin(theta), right_rod['y_pos']))

    right_rod['y_pos'] = np.add(np.multiply(np.sin(theta), right_rod['x_pos']),
                                np.multiply(np.cos(theta), right_rod['y_pos']))

    return left_rod, right_rod


def slice_position_error(data: list, report_path=False):

    # get rod positions and nominal positions
    left_rod, right_rod, nominal_positions = get_rods(data)
    # Correct for phantom rotation
    left_rod, right_rod = correct_rods_for_rotation(left_rod, right_rod)

    fov = hazenlib.get_field_of_view(data[0])

    # x_length_mm = np.subtract(right_rod['x_pos'], left_rod['x_pos']) * fov/data[0].Columns
    y_length_mm = np.subtract(left_rod['y_pos'], right_rod['y_pos']) * fov / data[0].Columns

    z_length_mm = np.divide(y_length_mm, 2)

    if z_length_mm[0] > z_length_mm[-1]:
        nominal_positions = nominal_positions[::-1]

    # Correct for zero offset
    nominal_positions = [x - nominal_positions[18] + z_length_mm[18] for x in nominal_positions]
    results = np.subtract(z_length_mm, nominal_positions)

    if report_path:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1)
        fig.set_size_inches(10, 10)
        fig.tight_layout()
        ax[0].imshow(data[19].pixel_array, cmap='gray')

        for idx in range(40):
            rods_x = [left_rod["x_pos"][idx], right_rod['x_pos'][idx]]
            rods_y = [left_rod["y_pos"][idx], right_rod['y_pos'][idx]]
            ax[0].scatter(rods_x, rods_y, 20, c='green', marker='+')

        ax[1].scatter(range(10, 50), results, marker='x')
        ax[1].set_yticks(np.arange(-2.5, 2.5, 0.5))
        fig.savefig('slice_position.png')

    return results


def main(data: list, report_path=False) -> list:

    if len(data) != 60:
        raise Exception('Need 60 DICOM')

    data.sort(key=lambda x: x.SliceLocation)  # sort by slice location

    data = data[10:50]  # ignore first and last 10 dicom

    try:
        results = slice_position_error(data, report_path)
    except Exception as e:
        raise

    import decimal
    decimal.getcontext().prec = 3
    results = [str(abs(decimal.Decimal(i)*1)) for i in results]
    del decimal

    return results


if __name__ == "__main__":
    main([os.path.join(sys.argv[1], i) for i in os.listdir(sys.argv[1])])
