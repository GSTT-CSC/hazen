import os
import cv2 as cv
import pydicom
import imutils
import matplotlib
import numpy as np

from collections import defaultdict
from skimage import filters

import hazenlib.exceptions as exc
from hazenlib.logger import logger

matplotlib.use("Agg")


def get_dicom_files(folder: str, sort=False) -> list:
    """Collect files with pixel_array into a list

    Args:
        folder (str): path to folder to check

    Returns:
        list: paths to DICOM image files (may be multi-framed)
    """
    file_list = []
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        if has_pixel_array(file_path):
            file_list.append(file_path)
    return file_list


def is_dicom_file(filename):
    """Check if file is a DICOM file, using the the first 128 bytes are preamble
    the next 4 bytes should contain DICM otherwise it is not a dicom

    Args:
        filename (str): path to file to be checked for the DICM header block

    Returns:
        bool: True or False whether file is a DICOM
    """
    # TODO: make it more robust, ensure that file contains a pixel_array
    file_stream = open(filename, "rb")
    file_stream.seek(128)
    data = file_stream.read(4)
    file_stream.close()
    if data == b"DICM":
        return True
    else:
        return False


def has_pixel_array(filename) -> bool:
    """Check whether DICOM object has pixel_array that can be used for calc

    Args:
        filename (str): path to file to be checked

    Returns:
        bool: True/False whether pixel_array is available
    """

    try:
        dcm = pydicom.dcmread(filename)
        # while enhanced DICOMs have a pixel_array, it's shape is in the format
        # (# frames, x_dim, y_dim)
        img = dcm.pixel_array
        return True
    except:
        logger.debug("%s does not contain image data", filename)
        return False


def is_enhanced_dicom(dcm: pydicom.Dataset) -> bool:
    """Check if file is an enhanced DICOM file

    Args:
        dcm (pydicom.Dataset): DICOM image object

    Raises:
        Exception: Unrecognised_SOPClassUID

    Returns:
        bool: True or False whether file is an enhanced DICOM
    """

    if dcm.SOPClassUID in ["1.2.840.10008.5.1.4.1.1.4.1", "EnhancedMRImageStorage"]:
        return True
    elif dcm.SOPClassUID == "1.2.840.10008.5.1.4.1.1.4":
        return False
    else:
        raise Exception("Unrecognised SOPClassUID")


def get_manufacturer(dcm: pydicom.Dataset) -> str:
    """Get the manufacturer field from the DICOM header

    Args:
        dcm (pydicom.Dataset): DICOM image object

    Raises:
        Exception: _description_

    Returns:
        str: manufacturer of the scanner used to obtain the DICOM image
    """
    supported = ["ge", "siemens", "philips", "toshiba", "canon"]
    manufacturer = dcm.Manufacturer.lower()
    for item in supported:
        if item in manufacturer:
            return item

    raise Exception(f"{manufacturer} not recognised manufacturer")


def get_average(dcm: pydicom.Dataset) -> float:
    """Get the NumberOfAverages field from the DICOM header

    Args:
        dcm (pydicom.Dataset): DICOM image object

    Returns:
        float: value of the NumberOfAverages field from the DICOM header
    """
    if is_enhanced_dicom(dcm):
        averages = (
            dcm.SharedFunctionalGroupsSequence[0].MRAveragesSequence[0].NumberOfAverages
        )
    else:
        averages = dcm.NumberOfAverages

    return averages


def get_bandwidth(dcm: pydicom.Dataset) -> float:
    """Get the PixelBandwidth field from the DICOM header

    Args:
        dcm (pydicom.Dataset): DICOM image object

    Returns:
        float: value of the PixelBandwidth field from the DICOM header
    """
    bandwidth = dcm.PixelBandwidth
    return bandwidth


def get_num_of_frames(dcm: pydicom.Dataset) -> int:
    """Get the number of frames from the DICOM pixel_array

    Args:
        dcm (pydicom.Dataset): DICOM image object

    Returns:
        float: value of the PixelBandwidth field from the DICOM header
    """
    # TODO: investigate what values could the dcm.pixel_array.shape be and what that means
    if len(dcm.pixel_array.shape) > 2:
        return dcm.pixel_array.shape[0]
    elif len(dcm.pixel_array.shape) == 2:
        return 1


def get_slice_thickness(dcm: pydicom.Dataset) -> float:
    """Get the SliceThickness field from the DICOM header

    Args:
        dcm (pydicom.Dataset): DICOM image object

    Returns:
        float: value of the SliceThickness field from the DICOM header
    """
    if is_enhanced_dicom(dcm):
        try:
            slice_thickness = (
                dcm.PerFrameFunctionalGroupsSequence[0]
                .PixelMeasuresSequence[0]
                .SliceThickness
            )
        except AttributeError:
            slice_thickness = (
                dcm.PerFrameFunctionalGroupsSequence[0]
                .Private_2005_140f[0]
                .SliceThickness
            )
        except Exception:
            raise Exception("Unrecognised metadata Field for Slice Thickness")
    else:
        slice_thickness = dcm.SliceThickness

    return slice_thickness


def get_pixel_size(dcm: pydicom.Dataset) -> (float, float):
    """Get the PixelSpacing field from the DICOM header

    Args:
        dcm (pydicom.Dataset): DICOM image object

    Returns:
        tuple of float: x and y values of the PixelSpacing field from the DICOM header
    """
    manufacturer = get_manufacturer(dcm)
    try:
        if is_enhanced_dicom(dcm):
            dx, dy = (
                dcm.PerFrameFunctionalGroupsSequence[0]
                .PixelMeasuresSequence[0]
                .PixelSpacing
            )
        else:
            dx, dy = dcm.PixelSpacing
    except:
        print("Warning: Could not find PixelSpacing.")
        if "ge" in manufacturer:
            fov = get_field_of_view(dcm)
            dx = fov / dcm.Columns
            dy = fov / dcm.Rows
        else:
            raise Exception("Manufacturer not recognised")

    return dx, dy


def get_TR(dcm: pydicom.Dataset) -> float:
    """Get the RepetitionTime field from the DICOM header

    Args:
        dcm (pydicom.Dataset): DICOM image object

    Returns:
        float: value of the RepetitionTime field from the DICOM header, or defaults to 1000
    """
    # TODO: explore what type of DICOM files do not have RepetitionTime in DICOM header
    try:
        if is_enhanced_dicom(dcm):
            TR = (
                dcm.SharedFunctionalGroupsSequence[0]
                .MRTimingAndRelatedParametersSequence[0]
                .RepetitionTime
            )
        else:
            TR = dcm.RepetitionTime
    except:
        logger.warning("Could not find Repetition Time. Using default value of 1000 ms")
        TR = 1000
    return TR


def get_rows(dcm: pydicom.Dataset) -> float:
    """Get the Rows field from the DICOM header

    Args:
        dcm (pydicom.Dataset): DICOM image object

    Returns:
        float: value of the Rows field from the DICOM header, or defaults to 256
    """
    try:
        rows = dcm.Rows
    except:
        logger.warning(
            "Could not find Number of matrix rows. Using default value of 256"
        )
        rows = 256

    return rows


def get_columns(dcm: pydicom.Dataset) -> float:
    """Get the Columns field from the DICOM header

    Args:
        dcm (pydicom.Dataset): DICOM image object

    Returns:
        float: value of the Columns field from the DICOM header, or defaults to 256
    """
    try:
        columns = dcm.Columns
    except:
        logger.warning(
            "Could not find matrix size (columns). Using default value of 256."
        )
        columns = 256
    return columns


def get_pe_direction(dcm: pydicom.Dataset):
    """Get the PhaseEncodingDirection field from the DICOM header

    Args:
        dcm (pydicom.Dataset): DICOM image object

    Returns:
        str: value of the InPlanePhaseEncodingDirection field from the DICOM header
    """
    if is_enhanced_dicom(dcm):
        return (
            dcm.SharedFunctionalGroupsSequence[0]
            .MRFOVGeometrySequence[0]
            .InPlanePhaseEncodingDirection
        )
    else:
        return dcm.InPlanePhaseEncodingDirection


def get_field_of_view(dcm: pydicom.Dataset):
    """Get Field of View value from DICOM header depending on manufacturer encoding

    Args:
        dcm (pydicom.Dataset): DICOM image object

    Raises:
        NotImplementedError: Manufacturer not GE, Siemens, Toshiba or Philips so FOV cannot be calculated.

    Returns:
        float: value of the Field of View (calculated as Columns * PixelSpacing[0])
    """
    # assumes square pixels
    manufacturer = get_manufacturer(dcm)

    if "ge" in manufacturer:
        fov = dcm[0x19, 0x101E].value
    elif "siemens" in manufacturer:
        fov = dcm.Columns * dcm.PixelSpacing[0]
    elif "philips" in manufacturer:
        if is_enhanced_dicom(dcm):
            fov = (
                dcm.Columns
                * dcm.PerFrameFunctionalGroupsSequence[0]
                .PixelMeasuresSequence[0]
                .PixelSpacing[0]
            )
        else:
            fov = dcm.Columns * dcm.PixelSpacing[0]
    elif "toshiba" in manufacturer:
        fov = dcm.Columns * dcm.PixelSpacing[0]
    else:
        raise NotImplementedError(
            "Manufacturer not GE, Siemens, Toshiba or Philips so FOV cannot be calculated."
        )

    return fov


def get_image_orientation(dcm):
    """
    From http://dicomiseasy.blogspot.com/2013/06/getting-oriented-using-image-plane.html

    Args:
        dcm (list): values of dcm.ImageOrientationPatient - list of float

    Returns:
        str: Sagittal, Coronal or Transverse
    """
    if is_enhanced_dicom(dcm):
        iop = (
            dcm.PerFrameFunctionalGroupsSequence[0]
            .PlaneOrientationSequence[0]
            .ImageOrientationPatient
        )
    else:
        iop = dcm.ImageOrientationPatient

    iop_round = [round(x) for x in iop]
    plane = np.cross(iop_round[0:3], iop_round[3:6])
    plane = [abs(x) for x in plane]
    if plane[0] == 1:
        return "Sagittal"
    elif plane[1] == 1:
        return "Coronal"
    elif plane[2] == 1:
        return "Transverse"


def determine_orientation(dcm_list):
    """Determine the phantom orientation based on DICOM metadata from a list of DICOM images.

    Note:
        The ImageOrientationPatient tag is a record of the orientation of the
        imaging volume which contains the phantom. The orientation of the
        imaging volume MAY NOT align with the true phantom orientation.

    Args:
        dcm_list (list): list of pyDICOM image objects.

    Returns:
        tuple (string, list):
            "saggital", "coronal", "axial", or "unexpected" orientation. \n
            list of the changing ImagePositionPatient values.
    """
    # for dcm in dcm_list:
    #     print(dcm.InstanceNumber) # unique
    #     print(dcm.ImagePositionPatient) # unique
    #     # The x, y, and z coordinates of the upper left hand corner (center of the first voxel transmitted) of the image, in mm
    #     # eg [28.364610671997, -88.268096923828, 141.94101905823]
    #     print(dcm.ImageOrientationPatient) # common
    #     # The direction cosines of the first row and the first column with respect to the patient.
    #     # eg
    #     # [1, 0, 0, 0, 1, 0]  transverse/axial
    #     # [1, 0, 0, 0, 0, -1] coronal
    #     # [0, 1, 0, 0, 0, -1] sagittal
    #     print(dcm.PixelSpacing) # common
    #     # Physical distance in the patient between the center of each pixel, specified by a numeric pair - adjacent row spacing (dx) (delimiter) adjacent column spacing (dy) in mm.
    #     print(dcm.SliceThickness) # common
    #     # Nominal slice thickness, in mm
    # Get the number of images in the list,
    # assuming each have a unique position in one of the 3 directions
    expected = len(dcm_list)
    iop = dcm_list[0].ImageOrientationPatient
    x = np.array([round(dcm.ImagePositionPatient[0]) for dcm in dcm_list])
    y = np.array([round(dcm.ImagePositionPatient[1]) for dcm in dcm_list])
    z = np.array([round(dcm.ImagePositionPatient[2]) for dcm in dcm_list])

    # Determine phantom orientation based on DICOM header metadata
    # Assume phantom orientation based on ImageOrientationPatient
    logger.debug("Checking phantom orientation based on ImageOrientationPatient")
    if iop == [0, 1, 0, 0, 0, -1] and len(set(x)) == expected:
        logger.debug("x %s", set(x))
        return "sagittal", x
    elif iop == [1, 0, 0, 0, 0, -1] and len(set(y)) == expected:
        logger.debug("y %s", set(y))
        return "coronal", y
    elif iop == [1, 0, 0, 0, 1, 0] and len(set(z)) == expected:
        logger.debug("z %s", set(z))
        return "axial", z
    else:
        logger.debug("Checking phantom orientation based on ImagePositionPatient")
        # Assume phantom orientation based on the changing value in ImagePositionPatient
        if (
            len(set(x)) == expected
            and len(set(y)) < expected
            and len(set(z)) < expected
        ):
            return "sagittal", x
        elif (
            len(set(x)) < expected
            and len(set(y)) == expected
            and len(set(z)) < expected
        ):
            return "coronal", y
        elif (
            len(set(x)) < expected
            and len(set(y)) < expected
            and len(set(z)) == expected
        ):
            return "axial", z
        else:
            logger.warning("Unable to determine orientation based on DICOM metadata")
            logger.info("x %s", set(x))
            logger.info("y %s", set(y))
            logger.info("z %s", set(z))
            return "unexpected", [x, y, z]


def rescale_to_byte(array):
    """
    WARNING: This function normalises/equalises the histogram. This might have unintended consequences.

    Args:
        array (np.array): dcm.pixel_array

    Returns:
        np.array: normalised pixel values as 8-bit (byte) integer
    """
    image_histogram, bins = np.histogram(array.flatten(), 255)
    cdf = image_histogram.cumsum()  # cumulative distribution function
    cdf = 255 * cdf / cdf[-1]  # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(array.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(array.shape).astype("uint8")


def detect_circle(img, dx):
    normalised_img = cv.normalize(
        src=img,
        dst=None,
        alpha=0,
        beta=255,
        norm_type=cv.NORM_MINMAX,
        dtype=cv.CV_8U,
    )
    detected_circles = cv.HoughCircles(
        normalised_img,
        cv.HOUGH_GRADIENT,
        1,
        param1=50,
        param2=30,
        minDist=int(10 / dx),  # used to be 180 / dx
        minRadius=int(5 / dx),
        maxRadius=int(16 / dx),
    )
    return detected_circles


class Rod:
    """Class for rods detected in the image"""

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Rod: {self.x}, {self.y}"

    def __str__(self):
        return f"Rod: {self.x}, {self.y}"

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
    """Class for the detection of shapes in pixel arrays
    This class is largely adapted from https://www.pyimagesearch.com/2016/02/08/opencv-shape-detection/
    """

    def __init__(self, arr):
        self.arr = arr
        self.contours = None
        self.shapes = defaultdict(list)
        self.blurred = None
        self.thresh = None

    def find_contours(self):
        """Find contours in pixel array"""
        # convert the resized image to grayscale, blur it slightly, and threshold it
        self.blurred = cv.GaussianBlur(self.arr.copy(), (5, 5), 0)  # magic numbers

        optimal_threshold = filters.threshold_li(
            self.blurred, initial_guess=np.quantile(self.blurred, 0.50)
        )
        self.thresh = np.where(self.blurred > optimal_threshold, 255, 0).astype(
            np.uint8
        )

        # have to convert type for find contours
        contours = cv.findContours(self.thresh, cv.RETR_TREE, 1)
        self.contours = imutils.grab_contours(contours)
        # rep = cv.drawContours(self.arr.copy(), [self.contours[0]], -1, color=(0, 255, 0), thickness=5)
        # plt.imshow(rep)
        # plt.title("rep")
        # plt.colorbar()
        # plt.show()

    def detect(self):
        """Detect specified shapes in pixel array

        Currently supported shapes:
            - circle
            - triangle
            - rectangle
            - pentagon
        """
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
        """Identify shapes in pixel array

        Args:
            shape (_type_): _description_

        Raises:
            exc.ShapeDetectionError: ensure that only expected shapes are detected
            exc.MultipleShapesError: ensure that only 1 shape is detected

        Returns:
            tuple: varies depending on shape detected
                - circle: x, y, r - corresponding to x,y coords of centre and radius
                - rectangle/square: (x, y), size, angle - corresponding to x,y coords of centre, size (tuple) and angle in degrees
        """
        self.find_contours()
        self.detect()

        if shape not in self.shapes.keys():
            # print(self.shapes.keys())
            raise exc.ShapeDetectionError(shape)

        if len(self.shapes[shape]) > 1:
            shapes = [{shape: len(contours)} for shape, contours in self.shapes.items()]
            raise exc.MultipleShapesError(shapes)

        contour = self.shapes[shape][0]
        if shape == "circle":
            # (x,y) is centre of circle, in x, y coordinates. x=column, y=row.
            (x, y), r = cv.minEnclosingCircle(contour)
            return x, y, r

        # Outputs in below code chosen to match cv.minAreaRect output
        # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html#b-rotated-rectangle
        # (x,y) is top-left of rectangle, in x, y coordinates. x=column, y=row.

        if shape == "rectangle" or shape == "square":
            (x, y), size, angle = cv.minAreaRect(contour)
            # OpenCV v4.5 adjustment
            # - cv.minAreaRect() output tuple order changed since v3.4
            # - swap size order & rotate angle by -90
            size = (size[1], size[0])
            angle = angle - 90
            return (x, y), size, angle
