"""

Local Otsu thresholding
http://scikit-image.org/docs/0.11.x/auto_examples/plot_local_otsu.html

"""
import sys
import os
import pydicom
from skimage import measure, filters
from skimage.morphology import disk
from matplotlib import pyplot as plt
import numpy as np
import cv2

import hazenlib as hazen


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
    X = np.matrix([[i, 1] for i in y_pos])

    m, c = np.linalg.lstsq(X, np.array(x_pos))[0]

    theta = np.arctan(m)
    return theta


def get_field_of_view(dcm: pydicom.Dataset):
    # assumes square pixels
    manufacturer = hazen.get_manufacturer(dcm)

    if manufacturer == 'GE MEDICAL SYSTEMS':
        fov = dcm['0x19, 101e']
    elif manufacturer == 'SIEMENS':
        fov = dcm.Columns * dcm.PixelSpacing[0]
    elif manufacturer == 'Philips Medical Systems':
        if hazen.is_enhanced_dicom(dcm):
            fov = dcm.Columns * dcm.PerFrameFunctionalGroupsSequence[0].PixelMeasuresSequence[0].PixelSpacing[0]
        else:
            fov = dcm.Columns * dcm.PixelSpacing[0]

    return fov


def get_rods_coords(dcm: pydicom.Dataset):
    cx, cy, cradius = hazen.find_circle(dcm=dcm)

    # clip image in xy plane to only include regions where rods could be
    x_window = int(cradius/4)
    y_window = int(cradius*0.95)

    arr = dcm.pixel_array
    clipped = np.zeros_like(arr)
    clipped[cy-y_window:cy+y_window, cx-x_window:cx+x_window] = arr[cy-y_window:cy+y_window, cx-x_window:cx+x_window]

    threshold = filters.threshold_otsu(clipped, 2)

    clipped_thresholded = clipped <= threshold  # binarise using otsu threshold

    labels, num = measure.label(clipped_thresholded, return_num=True)
    measured_objects = measure.regionprops(label_image=labels)

    rods = []
    for obj in measured_objects:
        if 5 < obj.bbox_area < 25:
            rods.append(obj)

    if len(rods) != 2:
        raise Exception('Did not find two rods.')

    rods.sort(key=lambda x: x.centroid[1])  # sort into Left and Right by using second coordinate

    ly, lx = rods[0].centroid
    ry, rx = rods[1].centroid

    # fig = plt.figure(1)
    # plt.imshow(labels)
    # plt.show()

    return int(round(lx)), int(round(ly)), int(round(rx)), int(round(ry))


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

    left_rod['x_pos'] = [np.subtract(np.multiply(np.cos(theta), left_rod['x_pos']),
                                     np.multiply(np.sin(theta), left_rod['y_pos'])
                                     )
                         ]

    left_rod['y_pos'] = [np.add(np.multiply(np.cos(theta), left_rod['x_pos']),
                                np.multiply(np.cos(theta), left_rod['y_pos'])
                                )
                         ]

    right_rod['x_pos'] = [np.subtract(np.multiply(np.cos(theta), right_rod['x_pos']),
                                      np.multiply(np.sin(theta), right_rod['y_pos'])
                                      )
                          ]
    right_rod['y_pos'] = [np.add(np.multiply(np.cos(theta), right_rod['x_pos']),
                                 np.multiply(np.cos(theta), right_rod['y_pos'])
                                 )
                          ]
    return left_rod, right_rod


def slice_position_error(data: list):

    # get rod positions and nominal positions
    left_rod, right_rod, nominal_positions = get_rods(data)

    # Correct for phantom rotation
    left_rod, right_rod = correct_rods_for_rotation(left_rod, right_rod)

    fov = get_field_of_view(data[0])

    # x_length_mm = np.subtract(right_rod['x_pos'], left_rod['x_pos']) * fov/dcm.Columns
    y_length_mm = np.subtract(left_rod['y_pos'], right_rod['y_pos']) * fov / data[0].Columns

    z_length_mm = np.divide(y_length_mm, 2)

    # Correct for zero offset
    nominal_positions = [x - nominal_positions[19] + z_length_mm[0][0][19] for x in nominal_positions]

    results = [round(i, 3) for i in np.subtract(z_length_mm[0][0], nominal_positions)]

    return results


def main(data: list)-> list:

    if len(data) != 60:
        raise Exception('Need 60 DICOM')

    data = [pydicom.read_file(dcm) for dcm in data]  # load dicom objects into memory

    data.sort(key=lambda x: x.SliceLocation)  # sort by slice location

    data = data[10:50]  # ignore first and last dicom

    results = slice_position_error(data)

    import decimal
    decimal.getcontext().prec = 3
    results = [str(abs(decimal.Decimal(i)*1)) for i in results]
    del decimal

    return results


if __name__ == "__main__":
    main([os.path.join(sys.argv[1], i) for i in os.listdir(sys.argv[1])])
