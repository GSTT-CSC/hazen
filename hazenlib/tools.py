import os
from collections import defaultdict
from skimage import filters
import cv2 as cv
import imutils
import matplotlib
import numpy as np
from pydicom import dcmread
matplotlib.use("Agg")

import hazenlib.exceptions as exc


def get_image_orientation(iop):
    """
    From http://dicomiseasy.blogspot.com/2013/06/getting-oriented-using-image-plane.html
    Args:
        iop:

    Returns:

    """
    iop_round = [round(x) for x in iop]
    plane = np.cross(iop_round[0:3], iop_round[3:6])
    plane = [abs(x) for x in plane]
    if plane[0] == 1:
        return "Sagittal"
    elif plane[1] == 1:
        return "Coronal"
    elif plane[2] == 1:
        return "Transverse"


def rescale_to_byte(array):
    """
    WARNING: This function normalises/equalises the histogram. This might have unintended consequences.
    Args:
        array:

    Returns:

    """
    image_histogram, bins = np.histogram(array.flatten(), 255)
    cdf = image_histogram.cumsum()  # cumulative distribution function
    cdf = 255 * cdf / cdf[-1]  # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(array.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(array.shape).astype('uint8')


class ShapeDetector:
    """
    This class is largely adapted from https://www.pyimagesearch.com/2016/02/08/opencv-shape-detection/

    """
    def __init__(self, arr):
        self.arr = arr
        self.contours = None
        self.shapes = defaultdict(list)
        self.blurred = None
        self.thresh = None

    def find_contours(self):
        # convert the resized image to grayscale, blur it slightly, and threshold it
        self.blurred = cv.GaussianBlur(self.arr.copy(), (5, 5), 0)    # magic numbers

        optimal_threshold = filters.threshold_li(self.blurred, initial_guess=np.quantile(self.blurred, 0.25))
        self.thresh = np.where(self.blurred > optimal_threshold, 255, 0).astype(np.uint8)

        # have to convert type for find contours
        contours = cv.findContours(self.thresh, cv.RETR_TREE, 1)
        self.contours = imutils.grab_contours(contours)
        # rep = cv.drawContours(self.arr.copy(), [self.contours[0]], -1, color=(0, 255, 0), thickness=5)
        # plt.imshow(rep)
        # plt.title("rep")
        # plt.colorbar()
        # plt.show()

    def detect(self):
        for c in self.contours:
            # initialize the shape name and approximate the contour
            peri = cv.arcLength(c, True)
            if peri < 100:
                # ignore small shapes, magic number is complete guess
                continue
            approx = cv.approxPolyDP(c, 0.04 * peri, True)

            # if the shape is a triangle, it will have 3 vertices
            if len(approx) == 3:
                shape = "triangle"

            # if the shape has 4 vertices, it is either a square or
            # a rectangle
            elif len(approx) == 4:
                shape = "rectangle"

            # if the shape is a pentagon, it will have 5 vertices
            elif len(approx) == 5:
                shape = "pentagon"

            # otherwise, we assume the shape is a circle
            else:
                shape = "circle"

            # return the name of the shape
            self.shapes[shape].append(c)

    def get_shape(self, shape):

        self.find_contours()
        self.detect()

        if shape not in self.shapes.keys():
            # print(self.shapes.keys())
            raise exc.ShapeDetectionError(shape)

        if len(self.shapes[shape]) > 1:
            shapes = [{shape: len(contours)}for shape, contours in self.shapes.items()]
            raise exc.MultipleShapesError(shapes)

        contour = self.shapes[shape][0]
        if shape == 'circle':
            # (x,y) is centre of circle, in x, y coordinates. x=column, y=row.
            (x, y), r = cv.minEnclosingCircle(contour)
            return x, y, r

        #have changed name of outputs in below code to match cv.minAreaRect output
        #(https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html#b-rotated-rectangle)
        # (x,y) is top-left of rectangle, in x, y coordinates. x=column, y=row.

        if shape == 'rectangle' or shape == 'square':
            (x,y), size, angle = cv.minAreaRect(contour)
            # OpenCV v4.5 adjustment
            # - cv.minAreaRect() output tuple order changed since v3.4
            # - swap size order & rotate angle by -90
            size = (size[1], size[0])
            angle = angle-90
            return (x,y), size, angle


def get_dicom_files(folder: str, sort=False) -> list:
    if sort:
        file_list = [os.path.join(folder, x) for x in os.listdir(folder) if is_dicom_file(os.path.join(folder, x))]
        file_list.sort(key=lambda x: dcmread(x).InstanceNumber)
    else:
        file_list = [os.path.join(folder, x) for x in os.listdir(folder) if is_dicom_file(os.path.join(folder, x))]
    return file_list


def is_dicom_file(filename):
        """
        Util function to check if file is a dicom file
        the first 128 bytes are preamble
        the next 4 bytes should contain DICM otherwise it is not a dicom

        :param filename: file to check for the DICM header block
        :type filename: str
        :returns: True if it is a dicom file
        """
        file_stream = open(filename, 'rb')
        file_stream.seek(128)
        data = file_stream.read(4)
        file_stream.close()
        if data == b'DICM':
            return True
        else:
            return False


