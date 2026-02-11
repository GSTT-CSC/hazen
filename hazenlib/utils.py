"""A utility module for medical imaging analysis.

Providing:

- robust DICOM metadata extraction
- image processing
- ROI detection functions.

Supports multi-frame and vendor-specific DICOM formats
(GE, Siemens, Philips, Toshiba),
includes string scrubbing for security,
circular/shape detection via OpenCV,
parallel processing helpers, and debugging tools.

Designed for automated phantom QA in MRI workflows.

"""

from __future__ import annotations

# Python imports
import copy
import multiprocessing
import os
import re
from collections import defaultdict
from multiprocessing import Pool

# Module imports
import cv2 as cv
import imutils
import matplotlib
import numpy as np
import pydicom
from skimage import filters

# Local imports
import hazenlib.exceptions as exc
from hazenlib.logger import logger

matplotlib.use("Agg")

REGEX_SCRUBNAME = "\\^\\\\`\\{\\}\\[\\]\\(\\)\\!\\$'\\/\\ \\_\\:\\,\\-\\&\\=\\.\\*\\+\\;\\#"  #: Regex to match for these dirty characters.


def dcmread(*args, **kwargs) -> pydicom.dataset.FileDataset:
    """Thin wrapper around pydicom.dcmread.

    Provide default arguments and error handling.

    """
    try:
        return pydicom.dcmread(*args, **kwargs)

    except pydicom.errors.InvalidDicomError:
        logger.warning(
            "Couldn't read DICOM, trying with force=True"
            "\nargs:\n%s\n\nkwargs:\n%s",
            args,
            kwargs,
        )
        _ = kwargs.pop("force", True)
        return pydicom.dcmread(*args, force=True, **kwargs)


def scrub(dirtyString, matchCharacters, join_str="_"):
    """
    This function provides the core functionality for scrubbing strings for bad characters. This is the most useful
    function in the core library since it provides a security hardening benefit as well. Ideally, user input should get
    scrubbed with this function or derivatives. This functionality is mediated by `re.split
    <https://docs.python.org/3/library/re.html?highlight=re%20split#re.split>`_.

    :param dirtyString: Untrustworthy string that needs to be stripped of bad characters such as newlines.
    :param matchCharacters: Iterable of characters to scrub out of the string (typically a string of such characters)
    :param join_str: String used in-between clean fragments. Example, te\\nst => te_st if this parameter is _
    :return: Reconstructed clean string.
    """
    pattern = re.compile("[{}]".format(matchCharacters))
    target_string = re.split(pattern, str(dirtyString))
    return join_str.join(target_string)


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
        logger.debug("%s is not a DICOM file.", filename)
        return False


def has_pixel_array(filename) -> bool:
    """Check whether DICOM object has pixel_array that can be used for calc

    Args:
        filename (str): path to file to be checked

    Returns:
        bool: True/False whether pixel_array is available
    """

    try:
        dcm = dcmread(filename)
        # while enhanced DICOMs have a pixel_array, it's shape is in the format
        # (# frames, x_dim, y_dim)
        _ = dcm.pixel_array

    except AttributeError:
        logger.debug("%s does not contain image data", filename)
        return False

    else:
        return True


def is_enhanced_dicom(dcm: pydicom.Dataset) -> bool:
    """Check if file is an enhanced DICOM file

    Args:
        dcm (pydicom.Dataset): DICOM image object

    Raises:
        Exception: Unrecognised_SOPClassUID

    Returns:
        bool: True or False whether file is an enhanced DICOM
    """

    if dcm.SOPClassUID in [
        "1.2.840.10008.5.1.4.1.1.4.1",
        "EnhancedMRImageStorage",
    ]:
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

    msg = f"{manufacturer} not recognised manufacturer"
    logger.error(msg)
    raise Exception(msg)


def get_average(dcm: pydicom.Dataset) -> float:
    """Get the NumberOfAverages field from the DICOM header.

    Args:
        dcm (pydicom.Dataset): DICOM image object

    Returns:
        float: value of the NumberOfAverages field from the DICOM header

    """
    if is_enhanced_dicom(dcm):
        averages = dcm[
            (0x5200, 0x9230)  # Per-Frame Functional Groups Sequence
        ][0][
            (0x0018, 0x9119)  # MR Averages Sequence
        ][0][
            (0x0018, 0x0083)  # Number of Averages
        ].value

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
    try:
        bandwidth = dcm.PixelBandwidth

    except AttributeError:
        bandwidth = dcm[
            (0x5200, 0x9229)  # Shared Functional Groups Sequence
        ][0][
            (0x0018, 0x9006)  # MR Imaging Modifier Sequence
        ][0][
            (0x0018, 0x0095)  # Pixel Bandwidth
        ].value
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
            msg = "Unrecognised metadata Field for Slice Thickness"
            logger.exception(msg)
            raise Exception(msg)
    else:
        slice_thickness = dcm.SliceThickness

    return slice_thickness


def get_pixel_size(
    dcm: pydicom.Dataset,
    *,
    swap_indexes: bool = False,
) -> (float, float):
    """Get the PixelSpacing field from the DICOM header

    Args:
        dcm (pydicom.Dataset): DICOM image object
        swap_indexes : If True, will return (dy, dx) else (dx, dy)

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

    except AttributeError as err:
        logger.warning("Could not find PixelSpacing")
        if "ge" in manufacturer:
            fov = get_field_of_view(dcm)
            dx = fov / dcm.Columns
            dy = fov / dcm.Rows
        else:
            msg = "Manufacturer not recognised"
            logger.error(msg)
            raise ValueError(msg) from err

    if swap_indexes:
        return dy, dx
    return dx, dy


def get_TR(dcm: pydicom.Dataset, *, strict: bool = False) -> float:
    """Get the RepetitionTime field from the DICOM header.

    Args:
        dcm : DICOM image object

    Returns:
        return value : RepetitionTime field from the DICOM header.
            If not found and strict is False, defaults to 1000.

    Raises:
        AttributeError : If strict is True and no Reptition is found.

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

    except AttributeError as err:
        msg = "Could not find Repetition Time"
        if strict:
            raise AttributeError(msg) from err

        logger.warning("%s. Using default value of 1000 ms", msg)
        TR = 1000
    return TR


def get_rows(dcm: pydicom.Dataset) -> float:
    """Get the Rows field from the DICOM header.

    Args:
        dcm (pydicom.Dataset): DICOM image object

    Returns:
        float: value of the Rows field from the DICOM header, or defaults to 256

    """
    try:
        rows = dcm.Rows

    except AttributeError:
        rows = 256
        logger.warning(
            "Could not find Number of matrix rows. Using default value of %i",
            rows,
        )

    return rows


def get_columns(dcm: pydicom.Dataset) -> float:
    """Get the Columns field from the DICOM header.

    Args:
        dcm (pydicom.Dataset): DICOM image object

    Returns:
        float: value of the Columns field from the DICOM header, or defaults to 256

    """
    try:
        columns = dcm.Columns

    except AttributeError:
        columns = 256
        logger.warning(
            "Could not find matrix size (columns). Using default value of %i",
            columns,
        )
    return columns


def get_pe_direction(dcm: pydicom.Dataset):
    """Get the PhaseEncodingDirection field from the DICOM header.

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
        logger.error("Manufacturer %s not supported", manufacturer)
        raise NotImplementedError(
            "Manufacturer not GE, Siemens, Toshiba or Philips so FOV cannot be calculated."
        )

    return fov


def get_datatype_max(dtype=np.uint8):
    """Get max value of the numpy datatype range.

    This is the machine max and not the data's max used value.
    For example, a np.uint8 has a machine range of 0 to 255.

    Args:
        dtype (np.dtype): Numpy datatype

    Returns:
        int|float: max machine value of data type

    """
    try:
        return np.iinfo(dtype).max
    except ValueError:
        return np.finfo(dtype).max


def get_datatype_min(dtype=np.uint8):
    """Get min value of the numpy datatype range.

    This is the machine min and not the data's min used value.
    For example, a np.uint8 has a machine range of 0 to 255.

    Args:
        dtype (np.dtype): Numpy datatype

    Returns:
        int|float: min machine value of data type

    """
    try:
        return np.iinfo(dtype).min
    except ValueError:
        return np.finfo(dtype).min


def get_image_IOP(dcm):
    """Get the IOP vectors from the DICOM header.

    Args:
        dcm (pydicom.dataset.FileDataset): DICOM Dataset

    Returns:
        pydicom.multival.MultiValue: vector
    """
    if is_enhanced_dicom(dcm):
        return (
            dcm.PerFrameFunctionalGroupsSequence[-1]
            .PlaneOrientationSequence[-1]
            .ImageOrientationPatient
        )
    return dcm.ImageOrientationPatient


def get_image_IPP(dcm):
    """Get the IPP vector from the DICOM header.

    Args:
        dcm (pydicom.dataset.FileDataset): DICOM Dataset

    Returns:
        pydicom.multival.MultiValue: vector
    """
    if is_enhanced_dicom(dcm):
        return (
            dcm.PerFrameFunctionalGroupsSequence[-1]
            .PlanePositionSequence[-1]
            .ImagePositionPatient
        )
    return dcm.ImagePositionPatient


def get_image_spacing(dcm):
    """Get the pixel resolution from the DICOM header.

    Args:
        dcm (pydicom.dataset.FileDataset): DICOM Dataset

    Returns:
        pydicom.multival.MultiValue: vector
    """
    if is_enhanced_dicom(dcm):
        return (
            dcm.PerFrameFunctionalGroupsSequence[-1]
            .PixelMeasuresSequence[-1]
            .PixelSpacing
        )
    return dcm.PixelSpacing


def get_image_orientation(dcm):
    """
    From http://dicomiseasy.blogspot.com/2013/06/getting-oriented-using-image-plane.html

    Args:
        dcm (list): values of dcm.ImageOrientationPatient - list of float

    Returns:
        str: Sagittal, Coronal or Transverse
    """
    iop = get_image_IOP(dcm)

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
    # Get the number of images in the list,
    # assuming each have a unique position in one of the 3 directions
    expected = len(dcm_list)

    iop = [np.round(c) for c in get_image_IOP(dcm_list[0])]
    x = np.array([round(get_image_IPP(dcm)[0]) for dcm in dcm_list])
    y = np.array([round(get_image_IPP(dcm)[1]) for dcm in dcm_list])
    z = np.array([round(get_image_IPP(dcm)[2]) for dcm in dcm_list])

    # Determine phantom orientation based on DICOM header metadata
    # Assume phantom orientation based on ImageOrientationPatient
    logger.debug(
        "Checking phantom orientation based on ImageOrientationPatient"
    )
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
        logger.debug(
            "Checking phantom orientation based on ImagePositionPatient"
        )
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
            logger.warning(
                "Unable to determine orientation based on DICOM metadata"
            )
            logger.info(f"Image orientation cosines => {iop}")
            logger.info("x %s\ny %s\nz %s", set(x), set(y), set(z))
            return "unexpected", [x, y, z]


def compute_dicom_frame_size(dcm):
    """Computes the full size in bytes of a Enhanced Multiframe DICOM frame. This value can be used for selecting the
    range of bytes for a given frame when manipulating the individual frames.

    Args:
        dcm (pydicom.dataset.FileDataset): DICOM Dataset

    Returns:
        float|int: byte count of a single frame
    """
    bits_allocated = dcm.BitsAllocated
    bytes_per_pixel = ((bits_allocated - 1) // 8) + 1
    samples_per_pixel = dcm.SamplesPerPixel
    rows, columns = dcm.Rows, dcm.Columns
    initial_frame_length = rows * columns * samples_per_pixel
    if bits_allocated == 1:
        return initial_frame_length // 8 + (
            initial_frame_length % 8 > 0
        )  # From pydicom's upcoming 3.0
    return initial_frame_length * bytes_per_pixel


def new_dicom(
    dcm: pydicom.dataset.FileDataset,
    pixel_data: np.ndarray,
    instance_number: int | None = None,
) -> pydicom.dataset.FileDataset:
    """Return a new DICOM with updated pixel data.

    Return a copy of dcm but with updated pixel data.
    Ensures Photometric Interpretation and Bits Stored
    are correct.

    Args:
        dcm : The original DICOM to copy.
        pixel_data : The updated pixel data.
        instance_number : If not None, used set the instance number.
            Defaults to None.

    Returns:
        new_dcm : A new dicom instance with the updated pixel data.

    """
    new_dcm = copy.deepcopy(dcm)

    # Extract single frame pixel data
    new_dcm.set_pixel_data(
        pixel_data,
        dcm[(0x0028, 0x0004)].value,  # Photometric Interpretation
        dcm[(0x0028, 0x0101)].value,  # Bits Stored
    )

    return new_dcm


def split_dicom(
    dcm: pydicom.dataset.FileDataset,
) -> list[pydicom.dataset.FileDataset]:
    """Split a multiframe DICOM into individual single-frame DICOMs.

    If the input is enhanced, we assume that it is a multiframe dicom
    and split it into constituent frames. We return this list.

    Otherwise, return a list whose single element is the given dicom

    Args:
        dcm : DICOM Dataset

    Returns:
        dcm_list : List of DICOM datasets

    """
    # In the case of non-enhanced DICOM - just wrap in a list
    if not is_enhanced_dicom(dcm):
        return [dcm]

    frame_count = dcm.NumberOfFrames
    single_frames = []

    for frame_idx in range(frame_count):
        try:
            pixel_data = dcm.pixel_array[frame_idx, :, :]

        # Case for (2D) sagittal localizer stored as an enhanced DICOM
        except IndexError:
            pixel_data = dcm.pixel_array

        single_frames.append(
            new_dicom(dcm, pixel_data, instance_number=frame_idx + 1),
        )

    return single_frames


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


def expand_data_range(
    data: np.array | np.ma.MaskedArray,
    valid_range: tuple | None = None,
    target_type: np.dtype = np.uint8,
) -> np.array | np.ma.MaskedArray:
    """Take a dataset and expands its data range to fill the value range possible in the target type.

    For example, if you have the data [1, 2, 3]
    and need it to fill the absolute values in uint16, you get:
    [0, 32767, 65,535]

    Args:
        data : dataset containing pixel values
        valid_range : tuple of values to use as the minimum and maximum
            of the dataset range.
            If None, we use the datasets' minimum and maximum.
        target_type : Numpy datatype to target in the expansion

    Returns:
        np.array: expanded range

    """
    dtype_max = get_datatype_max(target_type)
    lower, upper = (
        (data.min(), data.max()) if valid_range is None else valid_range
    )
    return (((data - lower) / (upper - lower)) * dtype_max).astype(target_type)


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
        cv.HOUGH_GRADIENT_ALT,
        1,
        param1=300,
        param2=0.9,
        minDist=int(10 / dx),
        minRadius=int(5 / (2 * dx)),
        maxRadius=int(16 / (2 * dx)),
    )
    if detected_circles is None:
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
    # debug_image_sample(normalised_img)
    return detected_circles


def detect_centroid(img, dx, dy):
    """Attempt to detect circle locations using cv2.HoughCircles().

    We do the following preprocessing of the input to improve accuracy:

        #. Blur image with a boxcard kernel to help the Laplacian accentuate the circular features.
            It boosts the center accuracy by ~3% in the worst case I tested.
        #. Compute laplacian of the image to extract edges.
        #. Normalize to 8 bit.
        #. Attempt circle detection with HughesCircle Transform from OpenCV.

    These preprocessing steps improve the circle contours, which are then fed into the Hough Transform.
    By the way, we try the Hough Transform with cv.HOUGH_GRADIENT_ALT first. Failing that, we try the default Hough
    Transform mode. Also, we use 2 separate sets of parameter for each transform until we hopefully detect the phantom.

    Args:
        img (np.ndarray): pixel array containing the data to perform circle detection on
        dx (float): The coordinates of the point to rotate
        dy (float, optional): The amplitude threshold for peak identification. Defaults to 1.

    Returns:
        np.ndarray: Flattened array of tuples. tuples follow the form (x, y, r)

    """
    img_smoothed = cv.blur(img, (5, 5))
    img_grad = cv.Laplacian(img_smoothed, cv.CV_64F)
    img_grad_8u = cv.normalize(
        src=img_grad,
        dst=None,
        alpha=0,
        beta=255,
        norm_type=cv.NORM_MINMAX,
        dtype=cv.CV_8U,
    )

    try:
        detected_circles = cv.HoughCircles(
            img_grad_8u,
            cv.HOUGH_GRADIENT_ALT,
            1,
            param1=300,
            param2=0.9,
            minDist=int(180 / dy),
            minRadius=int(180 / (2 * dy)),
            maxRadius=int(200 / (2 * dx)),
        )
        if detected_circles is None:
            detected_circles = cv.HoughCircles(
                img_grad_8u,
                cv.HOUGH_GRADIENT,
                1,
                param1=50,
                param2=30,
                minDist=int(180 / dy),
                minRadius=int(180 / (2 * dy)),
                maxRadius=int(200 / (2 * dx)),
            )
    except AttributeError:
        detected_circles = cv.HoughCircles(
            img_grad_8u,
            cv.HOUGH_GRADIENT_ALT,
            1,
            param1=300,
            param2=0.9,
            minDist=int(180 / dy),
            minRadius=80,
            maxRadius=200,
        )
        if detected_circles is None:
            detected_circles = cv.HoughCircles(
                img_grad_8u,
                cv.HOUGH_GRADIENT,
                1,
                param1=50,
                param2=30,
                minDist=int(180 / dy),
                minRadius=80,
                maxRadius=200,
            )
    return detected_circles.flatten()


def compute_radius_from_area(area, voxel_resolution, conversion_value=10):
    """Calculates the radius of an ROI given an area. The radius is in pixel count. Meaning, if we want to get the
    radius for a 200cm2 ROI in a 0.5mm in-plane resolution, we call this function `with area = 200`, `voxel_resolution
    = 0.5`, and `conversion_value = 10`. This will yield a radius in mm which immediately gets divided by the
    resolution to yield the radius in pixel count units.

    Args:
        area (int): Area of ROI that will be generated with the radius calculated in this function
        voxel_resolution (float): voxel/pixel resolution as given by the PixelSpacing attribute in the DICOM header.
            This is typically in millimeter units.
        conversion_value (int): Value to use to convert the radius from area units (cm, etc) to mm.

    Returns:
        int: Integer radius length.
    """
    return np.ceil(
        np.divide(
            np.sqrt(np.divide(area, np.pi)) * conversion_value,
            voxel_resolution,
        )
    ).astype(int)


def create_cross_mask(img, length, x_coord, y_coord):
    """Generates a mask for an roi at the given coordinates

    Args:
        img (np.ndarray|np.ma.MaskedArray): pixel array containing the data where to generate roi
        radius (int): Integer radius of the circular roi
        x_coord (int): x coordinate of the center of the roi
        y_coord (int): y coordinate of the center of the roi

    Returns:
        np.ma.MaskedArray: Masked Array containing data for area of interest and zeros everywhere else.
    """
    grid = np.zeros(img.shape, dtype=np.bool_)

    height = int(length / 2)
    half_length = int(length / 2)
    half_height = int(height / 2)
    quarter_height = int(height / 4)

    x_start = int(x_coord - quarter_height)
    y_start = int(y_coord - half_length)
    grid[y_start : y_start + length, x_start : x_start + half_height] = True

    x_start = int(x_coord - half_length)
    y_start = int(y_coord - quarter_height)
    grid[y_start : y_start + half_height, x_start : x_start + length] = True
    return grid


def create_rectangular_mask(img, width, height, x_coord, y_coord):
    """Generates a mask for an roi at the given coordinates

    Args:
        img (np.ndarray|np.ma.MaskedArray): pixel array containing the data where to generate roi
        width (int): Integer radius of the circular roi
        height (int): Integer radius of the circular roi
        x_coord (int): x coordinate of the center of the roi
        y_coord (int): y coordinate of the center of the roi

    Returns:
        np.ma.MaskedArray: Masked Array containing data for area of interest and zeros everywhere else.
    """
    grid = np.zeros(img.shape, dtype=np.bool_)

    half_width = int(width / 2)
    half_height = int(height / 2)

    x_start = int(x_coord - half_width)
    y_start = int(y_coord - half_height)
    grid[y_start : y_start + height, x_start : x_start + width] = True
    return grid


def create_circular_mask(img, radius, x_coord, y_coord):
    """Generates a mask for an roi at the given coordinates

    Args:
        img (np.ndarray|np.ma.MaskedArray): pixel array containing the data where to generate roi
        radius (int): Integer radius of the circular roi
        x_coord (int): x coordinate of the center of the roi
        y_coord (int): y coordinate of the center of the roi

    Returns:
        np.ma.MaskedArray: Masked Array containing data for area of interest and zeros everywhere else.
    """
    height, width = img.shape
    y_grid, x_grid = np.ogrid[:height, :width]
    return (x_grid - x_coord) ** 2 + (y_grid - y_coord) ** 2 <= radius**2


def create_circular_kernel(radius):
    """Generate ROI kernel that can be used during convolutions. This is for generating circular kernels.

    Args:
        radius (int): Integer radius of the circular roi

    Returns:
        np.ndarray: Arrays of 1s and 0s comprising the circular kernel to use for convolution.
    """
    diameter = (radius * 2) + 1
    kernel_arr = np.zeros((diameter, diameter), dtype=np.bool_)
    return create_circular_mask(kernel_arr, radius, radius, radius).astype(
        np.int_
    )


def create_circular_mean_kernel(radius):
    """Generate ROI kernel that can be used during convolutions. This is for generating circular kernels.
    Uses :py:func:`create_roi_kernel` to generate the initial kernel mask.

    avg_kernel = kernel / kernel.sum()

    Convoluting against this kernel should yield

    Args:
        radius (int): Integer radius of the circular roi

    Returns:
        np.ndarray: Arrays of 1s and 0s comprising the circular kernel to use for convolution.
    """
    mask = create_circular_kernel(radius)
    return mask / mask.sum()


def create_cross_roi_at(img, width, x_coord, y_coord):
    """Generates a masked array delimiting the area of interest. It assists numpy in determining what data to use in
    math operations.

    Args:
        img (np.ndarray|np.ma.MaskedArray): pixel array containing the data where to generate roi
        radius (int): Integer radius of the circular roi
        x_coord (int): x coordinate of the center of the roi
        y_coord (int): y coordinate of the center of the roi

    Returns:
        np.ma.MaskedArray: Masked Array containing data for area of interest and zeros everywhere else.
    """
    mask = create_cross_mask(img, width, x_coord, y_coord)
    masked_img = np.ma.masked_array(img.copy(), mask=~mask, fill_value=0)
    return masked_img


def create_rectangular_roi_at(img, width, height, x_coord, y_coord):
    """Generates a masked array delimiting the area of interest. It assists numpy in determining what data to use in
    math operations.

    Args:
        img (np.ndarray|np.ma.MaskedArray): pixel array containing the data where to generate roi
        width (int): Integer radius of the circular roi
        height (int): Integer radius of the circular roi
        x_coord (int): x coordinate of the center of the roi
        y_coord (int): y coordinate of the center of the roi

    Returns:
        np.ma.MaskedArray: Masked Array containing data for area of interest and zeros everywhere else.
    """
    mask = create_rectangular_mask(img, width, height, x_coord, y_coord)
    masked_img = np.ma.masked_array(img.copy(), mask=~mask, fill_value=0)
    return masked_img


def create_circular_roi_at(img, radius, x_coord, y_coord):
    """Generates a masked array delimiting the area of interest. It assists numpy in determining what data to use in
    math operations.

    Args:
        img (np.ndarray|np.ma.MaskedArray): pixel array containing the data where to generate roi
        radius (int): Integer radius of the circular roi
        x_coord (int): x coordinate of the center of the roi
        y_coord (int): y coordinate of the center of the roi

    Returns:
        np.ma.MaskedArray: Masked Array containing data for area of interest and zeros everywhere else.
    """
    mask = create_circular_mask(img, radius, x_coord, y_coord)
    masked_img = np.ma.masked_array(img.copy(), mask=~mask, fill_value=0)
    return masked_img


def create_circular_roi_with_numpy_index(img, radius, argx):
    """Wrapper around :py:func:`create_roi_at` meant to use flat element indices and return an roi centered around
    this element.

    Args:
        img (np.ndarray): pixel array containing the data where to generate roi
        radius (int): Integer radius of the circular roi
        argx (int): index to nd.array element if the array was flattened. Example, value from argmax().

    Returns:
        np.ma.MaskedArray: Masked Array containing data for area of interest and zeros everywhere else.
    """
    x_coord, y_coord = detect_roi_center(img, argx)
    return (
        create_circular_roi_at(img, radius, x_coord, y_coord),
        x_coord,
        y_coord,
    )


def detect_roi_center(img, argx):
    """Finds the x and y coordinates of the center of an roi given the flat index in the numpy array!

    Args:
        img (np.ndarray): pixel array containing the data where to generate roi
        argx (int): index to nd.array element if the array was flattened. Example, value from argmax().

    Returns:
        x_coord (int): x coordinate of the center of the roi
        y_coord (int): y coordinate of the center of the roi
    """
    height, width = img.shape
    y, x = np.divmod(
        argx, width
    )  # returns x //y, x % y per docs but x is found with x % y
    return x, y


def wait_on_parallel_results(fxn, arg_list=None, *, debug: bool = False):
    """Parallelises a function into n number of jobs. It uses Python's multiprocessing Pool to spawn several processes
    that accept each job instance and processes it. The main use in this project is as a way to keep the report writing
    as fast as possible when we have multiple images to write to disk.

    Args:
        fxn (function): function symbol to execute on arguments
        arg_list (list of tuple): List of tuples. Each tuple has the list of parameters to pass to function. Therefore,
            each tuple symbolizes a job we need to process using the specified function.
        debug : If True, runs sequentially rather than using parallel.

    Returns:
        list: List of values returned by each job.
    """
    arg_list = () if arg_list is None else arg_list
    # Check if already in a daemon process
    if multiprocessing.current_process().daemon or debug:
        return [fxn(*args) for args in arg_list]

    with Pool() as pool:
        results = []
        result_handles = []
        for args in arg_list:
            result_handles.append(pool.apply_async(fxn, args))

        pool.close()
        pool.join()

        for r in result_handles:
            results.append(r.get())
        return results


def debug_image_sample(img, out_path=None):
    """Uses :py:class:`DebugSnapshotShow` to display the current image snapshot.
    Use this function to force a display of an intermediate numpy image array to visually inspect results.

    Args:
        img (np.ndarray): pixel array containing the data to display
        out_path (str): file path where you would like to save a copy of the image

    """
    if len(img):
        snapshot = DebugSnapshotShow(img).image
        if out_path is not None:
            snapshot.save(out_path, format="PNG", dpi=(300, 300))


def debug_plot_sample(img, plot_indx=0):
    """Uses :py:class:`DebugSnapshotShow` to display the current image snapshot.
    Use this function to force a display of an intermediate numpy image array to visually inspect results.

    Args:
        img (np.ndarray): pixel array containing the data to display
        out_path (str): file path where you would like to save a copy of the image

    """
    if len(img):
        import matplotlib.pyplot as plt

        plt.plot(img)
        plt.savefig(f"/tmp/hazen_debug_plot_{plot_indx}.png")


def debug_image_sample_circles(img, circles=[], out_path=None):
    """Uses :py:class:`DebugSnapshotShow` to display the current image snapshot.
    Use this function to force a display of an intermediate numpy image array to visually inspect results.

    Args:
        img (np.ndarray): pixel array containing the data to display
        out_path (str): file path where you would like to save a copy of the image

    """
    for circle in circles[-1]:
        logger.info(f"Center {circle[0]}, {circle[1]}")
        center = (int(circle[0]), int(circle[1]))
        cv.circle(img, center, int(circle[2]), (0, 255, 0), 1)
    debug_image_sample(img, out_path)


class DebugSnapshotShow:
    """
    This class manages presentation of an image (file path or instance of PIL.Image. This class is used as if it were
    a function.
    See the `Pillow ImageShow Documentation <https://pillow.readthedocs.io/en/stable/reference/ImageShow.html>`_.
    You will need to install Pillow/PIL library and dependencies separately.
    This class is meant to assist during debugging of image processing steps.
    """

    def __init__(self, image_instance):
        from PIL import Image, ImageShow

        if isinstance(image_instance, str):
            image_instance = Image.open(image_instance)
        elif isinstance(image_instance, np.ndarray) or isinstance(
            image_instance, np.ma.MaskedArray
        ):
            image_instance = expand_data_range(
                image_instance, target_type=np.uint8
            )
            image_instance = Image.fromarray(image_instance)
        presenter = ImageShow.EogViewer()
        presenter.show_image(image_instance)
        self.image = image_instance


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
            logger.error(
                "Object other (type %s) has no property 'centroid'"
                " which is required for comparison.",
                type(other),
            )
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
        self.blurred = cv.GaussianBlur(
            self.arr.copy(), (5, 5), 0
        )  # magic numbers

        optimal_threshold = filters.threshold_li(
            self.blurred, initial_guess=np.quantile(self.blurred, 0.50)
        )
        self.thresh = np.where(
            self.blurred > optimal_threshold, 255, 0
        ).astype(np.uint8)

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
                logger.debug(
                    "%i vertices detected - assuming a circle", len(approx)
                )
                shape = "circle"

            # return the name of the shape
            logger.debug("%s detected", shape)
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

        if shape not in self.shapes:
            logger.error(
                "No valid shape detected - got %s but expected one of %s",
                shape,
                self.shapes.keys(),
            )
            raise exc.ShapeDetectionError(shape)

        if len(self.shapes[shape]) > 1:
            shapes = [
                {shape: len(contours)}
                for shape, contours in self.shapes.items()
            ]
            logger.error(
                "Multiple (%i) shapes were detected - should be just 1",
                len(self.shapes[shape]),
            )
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
            #
            # The above seems to have reverted in OpenCV v4.12
            # https://github.com/opencv/opencv/issues/28051
            # so the corrections are no longer needed.
            return (x, y), size, angle
