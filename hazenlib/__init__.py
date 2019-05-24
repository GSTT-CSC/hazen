import pydicom
import cv2 as cv
import numpy as np

__version__ = 'dev-0.1.0'
__author__ = "mohammad_haris.shuaib@kcl.ac.uk"
__all__ = ['snr', 'slice_position', 'slice_width', 'spatial_resolution', 'uniformity']


def is_enhanced_dicom(dcm: pydicom.Dataset) -> bool:
    """

    Parameters
    ----------
    dcm

    Returns
    -------
    bool

    Raises
    ------
    Exception
     Unrecognised SOPClassUID

    """

    if dcm.SOPClassUID == '1.2.840.10008.5.1.4.1.1.4.1':
        return True
    elif dcm.SOPClassUID == '1.2.840.10008.5.1.4.1.1.4':
        return False
    else:
        raise Exception('Unrecognised SOPClassUID')


def get_manufacturer(dcm: pydicom.Dataset) -> str:
    supported = ['GE', 'GE MEDICAL SYSTEMS', 'SIEMENS', 'Philips']  # Siemens has to be upper-case

    if dcm.Manufacturer in supported:
        return dcm.Manufacturer
    else:
        raise Exception('Manufacturer not recognised')


def find_circle(dcm: pydicom.Dataset):
    """
    Finds a circle that fits the outline of the phantom using the Hough Transform

    Parameters:
    ---------------
    a: image array (should be uint8)

    Returns:
    ---------------
    cenx, ceny, cradius: int, int, int
        xy pixel coordinates of the centre of the phantom
        radius of the phantom in pixels

    Raises
    ------
    Exception:
        Wrong number of circles detected, check image.

    """

    a = dcm.pixel_array
    a = ((a / a.max()) * 256).astype('uint8')

    # Perform Hough transform to find circle
    circles = cv.HoughCircles(a, cv.HOUGH_GRADIENT, 1, 200, param1=30,
                              param2=45, minRadius=0, maxRadius=0)

    # Check that a single phantom was found
    if len(circles) == 1:
        pass

    else:
        raise Exception("Wrong number of circles detected, check image.")

    # Draw circle onto original image
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # draw the outer circle
        cv.circle(a, (i[0], i[1]), i[2], 128, 2)
        # draw the center of the circle
        cv.circle(a, (i[0], i[1]), 2, 128, 2)

    cenx = i[0]
    ceny = i[1]
    cradius = i[2]
    return cenx, ceny, cradius