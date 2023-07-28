import os
from collections import defaultdict
from skimage import filters
import cv2 as cv
import imutils
import matplotlib
import numpy as np
import pydicom
import hazenlib.exceptions as exc
matplotlib.use("Agg")


def get_dicom_files(folder: str, sort=False) -> list:
    if sort:
        file_list = [os.path.join(folder, x) for x in os.listdir(folder) if is_dicom_file(os.path.join(folder, x))]
        file_list.sort(key=lambda x: pydicom.dcmread(x).InstanceNumber)
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
    supported = ['ge', 'siemens', 'philips', 'toshiba', 'canon']
    manufacturer = dcm.Manufacturer.lower()
    for item in supported:
        if item in manufacturer:
            return item

    raise Exception(f'{manufacturer} not recognised manufacturer')


def get_average(dcm: pydicom.Dataset) -> float:
    if is_enhanced_dicom(dcm):
        averages = dcm.SharedFunctionalGroupsSequence[0].MRAveragesSequence[0].NumberOfAverages
    else:
        averages = dcm.NumberOfAverages

    return averages


def get_bandwidth(dcm: pydicom.Dataset) -> float:
    """
    Returns PixelBandwidth

    Parameters
    ----------
    dcm: pydicom.Dataset

    Returns
    -------
    bandwidth: float
    """
    bandwidth = dcm.PixelBandwidth
    return bandwidth


def get_num_of_frames(dcm: pydicom.Dataset) -> int:
    """
    Returns number of frames of dicom object

    Parameters
    ----------
    dcm: pydicom.Dataset
        DICOM object

    Returns
    -------

    """
    if len(dcm.pixel_array.shape) > 2:
        return dcm.pixel_array.shape[0]
    elif len(dcm.pixel_array.shape) == 2:
        return 1


def get_slice_thickness(dcm: pydicom.Dataset) -> float:
    if is_enhanced_dicom(dcm):
        try:
            slice_thickness = dcm.PerFrameFunctionalGroupsSequence[0].PixelMeasuresSequence[0].SliceThickness
        except AttributeError:
            slice_thickness = dcm.PerFrameFunctionalGroupsSequence[0].Private_2005_140f[0].SliceThickness
        except Exception:
            raise Exception('Unrecognised metadata Field for Slice Thickness')
    else:
        slice_thickness = dcm.SliceThickness

    return slice_thickness


def get_pixel_size(dcm: pydicom.Dataset) -> (float, float):
    manufacturer = get_manufacturer(dcm)
    try:
        if is_enhanced_dicom(dcm):
            dx, dy = dcm.PerFrameFunctionalGroupsSequence[0].PixelMeasuresSequence[0].PixelSpacing
        else:
            dx, dy = dcm.PixelSpacing
    except:
        print('Warning: Could not find PixelSpacing..')
        if 'ge' in manufacturer:
            fov = get_field_of_view(dcm)
            dx = fov / dcm.Columns
            dy = fov / dcm.Rows
        else:
            raise Exception('Manufacturer not recognised')

    return dx, dy


def get_TR(dcm: pydicom.Dataset) -> (float):
    """
    Returns Repetition Time (TR)

    Parameters
    ----------
    dcm: pydicom.Dataset

    Returns
    -------
    TR: float
    """

    try:
        TR = dcm.RepetitionTime
    except:
        print('Warning: Could not find Repetition Time. Using default value of 1000 ms')
        TR = 1000
    return TR


def get_rows(dcm: pydicom.Dataset) -> (float):
    """
    Returns number of image rows (rows)

    Parameters
    ----------
    dcm: pydicom.Dataset

    Returns
    -------
    rows: float
    """
    try:
        rows = dcm.Rows
    except:
        print('Warning: Could not find Number of matrix rows. Using default value of 256')
        rows = 256

    return rows


def get_columns(dcm: pydicom.Dataset) -> (float):
    """
    Returns number of image columns (columns)

    Parameters
    ----------
    dcm: pydicom.Dataset

    Returns
    -------
    columns: float
    """
    try:
        columns = dcm.Columns
    except:
        print('Warning: Could not find matrix size (columns). Using default value of 256.')
        columns = 256
    return columns


def get_field_of_view(dcm: pydicom.Dataset):
    # assumes square pixels
    manufacturer = get_manufacturer(dcm)

    if 'ge' in manufacturer:
        fov = dcm[0x19, 0x101e].value
    elif 'siemens' in manufacturer:
        fov = dcm.Columns * dcm.PixelSpacing[0]
    elif 'philips' in manufacturer:
        if is_enhanced_dicom(dcm):
            fov = dcm.Columns * dcm.PerFrameFunctionalGroupsSequence[0].PixelMeasuresSequence[0].PixelSpacing[0]
        else:
            fov = dcm.Columns * dcm.PixelSpacing[0]
    elif 'toshiba' in manufacturer:
        fov = dcm.Columns * dcm.PixelSpacing[0]
    else:
        raise NotImplementedError('Manufacturer not ge,siemens, toshiba or philips so FOV cannot be calculated.')

    return fov


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


class Rod:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f'Rod: {self.x}, {self.y}'

    def __str__(self):
        return f'Rod: {self.x}, {self.y}'

    @property
    def centroid(self):
        return self.x, self.y

    def __lt__(self, other):
        """Using "reading order" in a coordinate system where 0,0 is bottom left"""
        try:
            x0, y0 = self.centroid
            x1, y1 = other.centroid
            return (-y0, x0) < (-y1, x1)
        except AttributeError:
            return NotImplemented

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y


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
        self.blurred = cv.GaussianBlur(self.arr.copy(), (5, 5), 0)  # magic numbers

        optimal_threshold = filters.threshold_li(self.blurred, initial_guess=np.quantile(self.blurred, 0.50))
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
            shapes = [{shape: len(contours)} for shape, contours in self.shapes.items()]
            raise exc.MultipleShapesError(shapes)

        contour = self.shapes[shape][0]
        if shape == 'circle':
            # (x,y) is centre of circle, in x, y coordinates. x=column, y=row.
            (x, y), r = cv.minEnclosingCircle(contour)
            return x, y, r

        # Outputs in below code chosen to match cv.minAreaRect output
        # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html#b-rotated-rectangle
        # (x,y) is top-left of rectangle, in x, y coordinates. x=column, y=row.

        if shape == 'rectangle' or shape == 'square':
            (x, y), size, angle = cv.minAreaRect(contour)
            # OpenCV v4.5 adjustment
            # - cv.minAreaRect() output tuple order changed since v3.4
            # - swap size order & rotate angle by -90
            size = (size[1], size[0])
            angle = angle - 90
            return (x, y), size, angle
