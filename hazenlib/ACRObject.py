from __future__ import annotations

# Typing imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pydicom

import sys
from typing import Any

import cv2
import numpy as np
import scipy
import skimage

from hazenlib.logger import logger
from hazenlib.utils import (
    detect_centroid,
    detect_circle,
    determine_orientation,
    expand_data_range,
    get_datatype_max,
    get_datatype_min,
    get_image_spacing,
    is_enhanced_dicom,
    split_dicom,
)


class ACRObject:
    """Base class for performing tasks on image sets of the ACR phantom.

    acquired following the ACR Large phantom guidelines
    """

    def __init__(self, dcm_list: list[pydicom.FileDataset]) -> None:
        """Initialise an ACR object instance.

        Args:
            dcm_list (list): list of pydicom.Dataset objects
                - DICOM files loaded

        """
        # First, need to determine if input DICOMs are
        # enhanced or normal, single or multi-frame
        # may be 11 in 1 or 11 separate DCM objects

        # # Initialise an ACR object from a list of images of the ACR phantom
        # Store pixel spacing value from the first image (expected to be the same for all)
        first_dcm = dcm_list[0]
        self.dx, self.dy = get_image_spacing(first_dcm)
        self.dx, self.dy = float(self.dx), float(self.dy)
        logger.info(
            f"In-plane acquisition resolution is {self.dx} x {self.dy}"
        )

        if is_enhanced_dicom(first_dcm):
            # We do not need to sort an enhanced multiframe object
            self.slice_stack = split_dicom(first_dcm)
        else:
            # Perform sorting of the input DICOM list based on position
            sorted_dcms = self.sort_dcms(dcm_list)

            # Perform sorting of the image slices based on phantom orientation
            self.slice_stack = self.order_phantom_slices(sorted_dcms)
        logger.info(
            f"Ordered slices => {[sl.InstanceNumber for sl in self.slice_stack]}"
        )

    def acquisition_type(self, *, strict: bool = True) -> str:
        """Get the acquisition type (T1w, T2w, Sagittal Localiser.

        Identified by the following:
        | Acquisition        | TR   |   TE | Slice Thick (mm) | Slice Gap |
        |--------------------+------+------+------------------+-----------|
        | Sagittal Localiser |  200 |   20 |              10* |       N/A |
        | T1                 |  500 |   20 |                5 |         5 |
        | T2                 | 2000 |   80 |                5 |         5 |
        * Older protocols use 20mm
        """
        try:
            TR = self.slice_stack[0][(0x0018, 0x0080)].value  # noqa: N806
            TE = self.slice_stack[0][(0x0018, 0x0081)].value  # noqa: N806
        except KeyError:
            # Assuming enhanced DICOM
            logger.debug(self.slice_stack[0])
            TR = (  # noqa: N806
                self.slice_stack[0]
                .SharedFunctionalGroupsSequence[0][(0x0018, 0x9112)][0]
                .RepetitionTime
            )
            TE = (  # noqa: N806
                self.slice_stack[0]
                .PerFrameFunctionalGroupsSequence[0]
                .MREchoSequence[0]
                .EffectiveEchoTime
            )

        def is_close(value: float, target: float, rel_tol: float) -> bool:
            return abs(value - target) / target <= rel_tol

        sequence_params = {
            "Sagittal Localiser": (200, 20),
            "T1": (500, 20),
            "T2": (2000, 80),
        }
        rel_tol = 1e-9 if strict else 1e-2

        for sequence, (tr, te) in sequence_params.items():
            if is_close(TE, te, rel_tol=rel_tol) and is_close(
                TR, tr, rel_tol=rel_tol
            ):
                msgs = []
                if te != TE:
                    msgs.append(
                        f"TE ({TE}) is within tolerance of {te} +- {rel_tol}%",
                    )
                if tr != TR:
                    msgs.append(
                        f"TR ({TR}) is within tolerance of {tr} +- {rel_tol}%",
                    )
                if msgs:
                    logger.warning(
                        "%s so sequence identified as %s"
                        " but not with an exact match.",
                        " and ".join(msgs),
                        sequence,
                    )

                return sequence

        sequence = "Unknown"
        logger.error(
            "Could not match acquisition type from TE (%f) and TR (%f)"
            " setting acquisition type to %s",
            TE,
            TR,
            sequence,
        )
        return sequence

    def sort_dcms(self, dcm_list):
        """Sort a stack of DICOM images based on slice position.

        Args:
            dcm_list (list): list of pyDICOM image objects

        Returns:
            list: sorted list of pydicom.Dataset objects
        """
        orientation, positions = determine_orientation(dcm_list)
        if orientation == "unexpected":
            # TODO: error out for now,
            # in future allow manual override based on optional CLI args
            logger.error(f"Unknown orientation detected => {orientation}")
            sys.exit()

        logger.info("image orientation is %s", orientation)
        dcm_stack = [dcm_list[i] for i in np.argsort(positions)]
        # img_stack = [dcm.pixel_array for dcm in dcm_stack]
        return dcm_stack  # , img_stack

    def order_phantom_slices(self, dcm_list):
        """Determine slice order based on the detection of the small circle in the first slice
        # or an LR orientation swap is required. \n

        # This function analyzes the given set of images and their associated DICOM objects to determine if any
        # adjustments are needed to restore the correct slice order and view orientation.

        Args:
            dcm_list (list): list of pyDICOM image objects

        Returns:
            list: sorted list of pydicom.Dataset objects corresponding to ordered phantom slices
        """
        # Check whether the circle is on the first or last slice

        # Get pixel array of first and last slice
        first_slice = dcm_list[0].pixel_array
        last_slice = dcm_list[-1].pixel_array
        # Detect circles in the first and last slice
        detected_circles_first = detect_circle(first_slice, self.dx)
        detected_circles_last = detect_circle(last_slice, self.dx)

        # It is assumed that only the first or the last slice has circles
        if (
            detected_circles_first is not None
            and detected_circles_last is None
        ):
            # If first slice has the circle then slice order is correct
            logger.info("Slice order inversion is not required.")
            return dcm_list
        if (
            detected_circles_first is None
            and detected_circles_last is not None
        ):
            # If last slice has the circle then slice order needs to be reversed
            logger.info("Performing slice order inversion.")
            return dcm_list[::-1]

        logger.debug("Neither slices had a circle detected")
        return dcm_list

    @staticmethod
    def determine_rotation(img):
        """Determine the rotation angle of the phantom using edge detection and the Hough transform.
        only relevant for MTF-based spatial resolution - need to convince David Price!!!!!

        Args:
            img (np.ndarray): pixel array of a DICOM object

        Returns:
            float: The rotation angle in degrees.
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

        thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]
        dilate = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)
        diff = cv2.absdiff(dilate, thresh)

        h, theta, d = skimage.transform.hough_line(diff)
        _, angles, _ = skimage.transform.hough_line_peaks(h, theta, d)

        if len(angles) >= 1:
            angle = np.rad2deg(scipy.stats.mode(angles, keepdims=False).mode)
            if angle < 0:
                rot_angle = angle + 90
            else:
                rot_angle = angle - 90
        else:
            rot_angle = 0
        logger.info("Phantom is rotated by %s degrees", rot_angle)

        return rot_angle

    def rotate_images(self, dcm_list, rot_angle):
        """Rotate the images by a specified angle. The value range and dimensions of the image are preserved.

        Args:
            dcm_list (list): list of pyDICOM image objects
            rot_angle (float): angle in degrees that image (pixel array) should be rotated by

        Returns:
            list of np.ndarray: The rotated images.
        """
        rotated_images = skimage.transform.rotate(
            dcm_list, rot_angle, resize=False, preserve_range=True
        )
        return rotated_images

    @staticmethod
    def find_phantom_center(img, dx, dy, axial=True):
        """Find the center of the ACR phantom in a given slice (pixel array) \n
        using the Hough circle detector on a blurred image.

        Args:
            img (np.ndarray): pixel array of the DICOM image.
            dx (float): pixel array of the DICOM image.
            dy (float): pixel array of the DICOM image.
            axial (bool): Whether we are attempting to detect the center of the axial dataset.

        Returns:
            tuple of ints: (x, y) coordinates of the center of the image
        """
        logger.info("Detecting centroid location ...")
        if axial:
            detected_circles = detect_centroid(img, dx, dy)
            centre_x = int(np.round(detected_circles[0]))
            centre_y = int(np.round(detected_circles[1]))
            radius = np.round(detected_circles[2])
        else:
            # This is meant for sagittal localizers so that we can identify their center.
            bbox, area = ACRObject.find_largest_rectangle(img)
            centre_x = int(np.round(bbox[0] + (bbox[2] / 2)))
            centre_y = int(np.round(bbox[1] + (bbox[3] / 2)))
            radius = (bbox[2], bbox[3])
        logger.info(f"Centroid (x, y) => {centre_x}, {centre_y}")

        logger.info(
            "Phantom center found at (%i,%i) with radius %s",
            centre_x,
            centre_y,
            radius,
        )
        return (centre_x, centre_y), radius

    @staticmethod
    def find_all_rectangles(img):
        """Find all contours and report the rectangles found in image.

        Preprocessing Steps
        ___________________

            #. Normalize image to 8bit.
            #. Smooth it with a sigma of 1.
            #. Apply Canny operator to obtain the edge feature map.

        Processing Steps
        ________________

            #. Pass edges to OpenCV
            #. Remove convexity from contours.
            #. Force rough rectangular approximation. We just want the general rectangles in image.
            #. Return list of rectangles (points).

        Args:
            img (np.ndarray): pixel array of the DICOM image.

        Returns:
            list of np.ndarray: List of arrays containing rectangle points in space.
        """
        normalized = ACRObject.normalize(img)
        img_blur = ACRObject.filter_with_gaussian(
            normalized, dtype=normalized.dtype
        )
        canny = cv2.Canny(img_blur, 10, 240)
        contours = cv2.findContours(
            canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )[0]

        rectangles = []
        for c in contours:
            convex = cv2.convexHull(c)
            epsilon = 0.1 * cv2.arcLength(convex, True)
            approx = cv2.approxPolyDP(convex, epsilon, True)
            rectangles.append(approx)
        return rectangles

    @staticmethod
    def find_largest_rectangle(img):
        """Calls :py:meth:`find_all_rectangles` to obtain the set of rectangles in image.
        We then iterate through this list of rectangles and return the bounding box and area of the largest rectangle
        in image.

        Args:
            img (np.ndarray): pixel array of the DICOM image.

        Returns:
            tuple: Bounding box and area of rectangle.
        """
        contours = ACRObject.find_all_rectangles(img)
        largest_c = None
        largest_area = 0
        for c in contours:
            area = cv2.contourArea(c)
            if area > largest_area:
                largest_area = area
                largest_c = c
        bounding_box = cv2.boundingRect(largest_c)
        return bounding_box, largest_area

    @staticmethod
    def get_presentation_pixels(dcm):
        """Automatically resolves the pixel values as would have been expected for presentation.
        We follow section `C.11.2.1.2.2 General Requirements for Window Center and Window Width <https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.11.2.html#sect_C.11.2.1.3>`_
        from the DICOM standard.

        This means we do:

            #. Look for expected windowing function as per section C.11.2.1.3 VOI LUT Function. We default to LINEAR as
                the LUT Function if no other function is defined.
            #. Look for Window Center (0028,1050), Window Width (0028,1051) and VOI LUT Function (0028,1056) Attributes.
                If VOI LUT and Window settings are present, I currently default to the Window Settings and assume the
                LUT is an alternative or secondary display setting.
            #. Find max value for pixel array data type.
            #. Cast initial pixel array to float64 for calculation.
            #. Apply base rescaling of pixel data.
            #. Apply window function (default is linear) with default window settings.
            #. Return transformed data.

        .. note::

            #. For the purpose of this definition, a floating point calculation without integer truncation is assumed,
                though the manner of implementation may vary as long as the result is the same.
            #. The pseudo-code function computes a continuous value over the output range without any discontinuity at
                the boundaries. The value of 0 for w is expressly forbidden, and the value of 1 for w does not cause
                division by zero, since the continuous segment of the function will never be reached for that case.

            #. For example, for an output range 0 to 255:

                c=2048, w=4096 becomes:

                    if (x <= 0) then y = 0

                    else if (x > 4095) then y = 255

                    else y = ((x - 2047.5) / 4095 + 0.5) * (255-0) + 0

                c=2048, w=1 becomes:

                    if (x <= 2047.5) then y = 0

                    else if (x > 2047.5) then y = 255

                    else /* not reached */

                c=0, w=100 becomes:

                    if (x <= -50) then y = 0

                    else if (x > 49) then y = 255

                    else y = ((x + 0.5) / 99 + 0.5) * (255-0) + 0

                c=0, w=1 becomes:

                    if (x <= -0.5) then y = 0

                    else if (x > -0.5) then y = 255

                    else /* not reached */

            #. A Window Center of 2n-1 and a Window Width of 2n selects the range of input values from 0 to 2n-1.
                This represents a mathematical identity VOI LUT transformation over the possible input values
                (whether used or not) in the case where no Modality LUT is specified and the stored pixel data are n
                bit unsigned integers.

            #. In the case where x1 is the lowest input value actually used in the Pixel Data and x2 is the highest,
                a Window Center of (x1+x2+1)/2 and a Window Width of (x2-x1+1) selects the range of input values from
                x1 to x2, which represents the full range of input values present as opposed to possible. This is
                distinct from the mathematical identity VOI LUT transformation, which instead selects the full range
                of input values possible as opposed to those actually used. The mathematical identity and full input
                range transformations are the same when x1 = 0 and x2 is 2n-1 and the input values are n bit unsigned
                integers. See also Note 7.

            #. A Window Width of 1 is typically used to represent a "threshold" operation in which those integer
                input values less than the Window Center are represented as the minimum displayed value and those
                greater than or equal to the Window Center are represented as the maximum displayed value. A Window
                Width of 2 will have the same result for integral input values.

            #. The application of Window Center (0028,1050) and Window Width (0028,1051) may select a signed input
                range. There is no implication that this signed input range is clipped to zero.

            #. The selected input range may exceed the actual range of the input values, thus effectively "compressing"
                the contrast range of the displayed data into a narrower band of the available contrast range, and
                "flattening" the appearance. There are no limits to the maximum value of the window width, or to the
                minimum or maximum value of window level, both of which may exceed the actual or possible range of
                input values.

            #. Input values "below" the window are displayed as the minimum output value and input values "above" the
                window are displayed as the maximum output value. This is the common usage of the window operation
                in medical imaging. There is no provision for an alternative approach in which all values "outside"
                the window are displayed as the minimum output value.

            #. The output of the Window Center/Width or VOI LUT transformation is either implicitly scaled to the
                full range of the display device if there is no succeeding transformation defined, or implicitly scaled to
                the full input range of the succeeding transformation step (such as the Presentation LUT), if present.
                See Section C.11.6.1.

            #. Fractional values of Window Center and Window Width are permitted (since the VR of these Attributes is
                Decimal String), and though they are not often encountered, applications should be prepared to accept
                them.

        .. warning::

            I do not implement all aspects of section C.11.2.1.2. Any elements that are found as needed in the wild
            should be retrospectively implemented.

        Args:
            dcm (pydicom.Dataset): DICOM instance used to find the default windowing and rescaling settings.

        Returns:
            raw (np.ndarray): Original pixel array.
            rescaled (np.ndarray): Rescaled pixel array.
            presentation (np.ndarray): Presentation ready pixel array.

        """
        img = dcm.pixel_array
        dtype = img.dtype
        slope = dcm.get("RescaleSlope", 1)
        intercept = dcm.get("RescaleIntercept", 1)
        center = dcm.get("WindowCenter", None)
        width = dcm.get("WindowWidth", None)

        def get_from_frame_voi_lut_sequence(prop: str) -> Any:
            return dcm[
                (0x5200, 0x9230)  # Per-Frame Functional Groups Sequence
            ][0][
                (0x0028, 0x9132)  # Frame VOI LUT Sequence
            ][0][prop].value

        if center is None:
            center = get_from_frame_voi_lut_sequence("WindowCenter")

        if width is None:
            width = get_from_frame_voi_lut_sequence("WindowWidth")

        voi_lut_function = dcm.get("VOILUTFunction", "linear").lower()
        float_data = img.copy().astype(
            np.float32
        )  # Cast to float to maintain precision
        rescaled = ACRObject.rescale_data(float_data, slope, intercept)
        windowed = ACRObject.apply_window_center_width(
            rescaled, center, width, voi_lut_function, dtype
        )
        rounded_window = np.round(
            windowed.data
        )  # Round so that we have integers. Realistically, we should only be
        # dealing with uint16, but adjust this step if there's other data.
        return (
            img,
            np.round(rescaled).astype(dtype),
            rounded_window.astype(dtype),
        )  # Cast back to original type, allow truncation

    def get_mask_image(
        self, image, centre, mag_threshold=0.07, open_threshold=500
    ):
        """Create a masked pixel array. \n
        Mask an image by magnitude threshold before applying morphological opening to remove small unconnected
        features. The convex hull is calculated in order to accommodate for potential air bubbles.

        Args:
            image (np.ndarray): pixel array of the dicom
            centre (tuple): x,y coordinates of the circle centre.
            mag_threshold (float, optional): magnitude threshold. Defaults to 0.07.
            open_threshold (int, optional): open threshold. Defaults to 500.

        Returns:
            np.ndarray: the masked image
        """
        test_mask = self.circular_mask(centre, (80 // self.dx), image.shape)
        test_image = image * test_mask
        # get range of values in the mask
        test_vals = test_image[np.nonzero(test_image)]
        if np.percentile(test_vals, 80) - np.percentile(
            test_vals, 10
        ) > 0.9 * np.max(image):
            logger.warning(
                "Large intensity variations detected in image."
                " Using local thresholding!"
            )
            initial_mask = skimage.filters.threshold_sauvola(
                image, window_size=3, k=0.95
            )
        else:
            initial_mask = image > mag_threshold * np.max(image)

        # unconnected region of pixels, to remove noise
        opened_mask = skimage.morphology.area_opening(
            initial_mask, area_threshold=open_threshold
        )
        # remove air bubbles from image area
        final_mask = skimage.morphology.convex_hull_image(opened_mask)

        return final_mask

    @staticmethod
    def circular_mask(centre, radius, dims):
        """Generates a circular mask using given centre coordinates and a given radius. Generates a linspace grid the
        size of the given dimensions and checks whether each point on the linspace grid is within the desired radius
        from the given centre coordinates. Each linspace value within the chosen radius then becomes part of the mask.


        Args:
            centre (tuple): centre coordinates of the circular mask.
            radius (int): radius of the circular mask.
            dims (tuple): dimensions to create the base linspace grid from.

        Returns:
            np.ndarray: A sorted stack of images, where each image is represented as a 2D numpy array.
        """
        # Define a circular logical mask
        x = np.linspace(1, dims[0], dims[0])
        y = np.linspace(1, dims[1], dims[1])

        X, Y = np.meshgrid(x, y)
        mask = (X - centre[0]) ** 2 + (Y - centre[1]) ** 2 <= radius**2

        return mask

    def measure_orthogonal_lengths(
        self, mask, cxy, h_offset=(0, 0), v_offset=(0, 0)
    ):
        """Compute the horizontal and vertical lengths of a mask, based on the centroid.

        Args:
            mask (np.ndarray): Boolean array of the image where pixel values meet threshold
            cxy  (tuple): x,y coordinates of the circle centre.

        Returns:
            dict: a dictionary with the following:
                'Horizontal Start'      | 'Vertical Start' : tuple of int
                    Horizontal/vertical starting point of the object.
                'Horizontal End'        | 'Vertical End' : tuple of int
                    Horizontal/vertical ending point of the object.
                'Horizontal Extent'     | 'Vertical Extent' : np.ndarray of int
                    Indices of the non-zero elements of the horizontal/vertical line profile.
                'Horizontal Distance'   | 'Vertical Distance' : float
                    The horizontal/vertical length of the object.
        """
        dims = mask.shape
        (vertical, horizontal) = cxy

        horizontal_start = (horizontal + h_offset[1], 0)
        horizontal_end = (horizontal + h_offset[1], dims[0] - 1 + h_offset[0])
        horizontal_line_profile = skimage.measure.profile_line(
            mask, horizontal_start, horizontal_end
        )
        horizontal_extent = np.nonzero(horizontal_line_profile)[0]
        horizontal_distance = (
            horizontal_extent[-1] - horizontal_extent[0]
        ) * self.dx

        vertical_start = (0 + v_offset[1], vertical + v_offset[0])
        vertical_end = (dims[1] - 1 + v_offset[1], vertical + v_offset[0])
        vertical_line_profile = skimage.measure.profile_line(
            mask, vertical_start, vertical_end
        )
        vertical_extent = np.nonzero(vertical_line_profile)[0]
        vertical_distance = (
            vertical_extent[-1] - vertical_extent[0]
        ) * self.dy

        length_dict = {
            "Horizontal Start": horizontal_start,
            "Horizontal End": horizontal_end,
            "Horizontal Extent": horizontal_extent,
            "Horizontal Distance": horizontal_distance,
            "Vertical Start": vertical_start,
            "Vertical End": vertical_end,
            "Vertical Extent": vertical_extent,
            "Vertical Distance": vertical_distance,
        }

        return length_dict

    @staticmethod
    def rotate_point(origin, point, angle):
        """Compute the horizontal and vertical lengths of a mask, based on the centroid.

        Args:
            origin (tuple): The coordinates of the point around which the rotation is performed.
            point (tuple): The coordinates of the point to rotate.
            angle (int): Angle in degrees.

        Returns:
            tuple of float: Floats representing the x and y coordinates of the input point
            after being rotated around an origin.
        """
        theta = np.radians(angle)
        c, s = np.cos(theta), np.sin(theta)

        x_prime = (
            origin[0] + c * (point[0] - origin[0]) - s * (point[1] - origin[1])
        )
        y_prime = (
            origin[1] + s * (point[0] - origin[0]) + c * (point[1] - origin[1])
        )
        return x_prime, y_prime

    @staticmethod
    def find_n_highest_peaks(data, n, height=1):
        """Find the indices and amplitudes of the N highest peaks within a 1D array.

        Args:
            data (np.ndarray): pixel array containing the data to perform peak extraction on
            n (int): The coordinates of the point to rotate
            height (int, optional): The amplitude threshold for peak identification. Defaults to 1.

        Returns:
            tuple of np.ndarray:
                peak_locs: A numpy array containing the indices of the N highest peaks identified. \n
                peak_heights: A numpy array containing the amplitudes of the N highest peaks identified.

        """
        peaks = scipy.signal.find_peaks(data, height)
        pk_heights = peaks[1]["peak_heights"]
        pk_ind = peaks[0]

        peak_heights = pk_heights[
            (-pk_heights).argsort()[:n]
        ]  # find n highest peak amplitudes
        peak_locs = pk_ind[
            (-pk_heights).argsort()[:n]
        ]  # find n highest peak locations

        return np.sort(peak_locs), np.sort(peak_heights)

    @staticmethod
    def rescale_data(data, slope=1, intercept=0):
        """Rescales the data by a given slope and intercept.

        The equation is y = mx + b

        Args:
            data (np.ndarray): pixel array containing the data to perform peak extraction on
            slope (float): The slope m.
            intercept (float): The intercept b.

        Returns:
            np.ndarray: Scaled data
        """
        return np.add(np.multiply(slope, data), intercept)

    @staticmethod
    def apply_linear_window_center_width(
        data, center, width, dtmin=0, dtmax=255
    ):
        """Filters data by the specified center and width using the DICOM linear equation.
        See `C.11.2.1.2.1 Default LINEAR Function <https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.11.2.html#sect_C.11.2.1.2.1>`_.

        ::

            if (x <= c - 0.5 - (w-1) /2), then y = ymin
            else if (x > c - 0.5 + (w-1) /2), then y = ymax
            else y = ((x - (c - 0.5)) / (w-1) + 0.5) * (ymax- ymin) + ymin

        ..info::

            #. Window Width (0028,1051) shall always be greater than or equal to 1.
            #. When Window Width (0028,1051) is greater than 1, these Attributes select the range of input values that
                are to be mapped to the full range of the displayed output.
            #. When Window Width (0028,1051) is equal to 1, they specify a threshold below which input values will be
                displayed as the minimum output value.

        Args:
            data (np.ndarray): pixel array containing the data to perform peak extraction on
            center (int): The desired Window Center setting.
            width (int): The desired Window Width setting.
            dtmin (int): The min value of datatype.
            dtmax (int): The max value of datatype.

        Returns:
            np.ndarray: Windowed data
        """
        width = 1 if width <= 0 else width
        logger.info(
            f"Applying Window Settings using the Linear method => Center: {center} Width: {width}"
        )
        adjusted_width = width - 1
        half_width = adjusted_width / 2
        lower_bound = center - 0.5 - half_width
        upper_bound = center - 0.5 + half_width
        logger.info(
            f"Half Width: {half_width} Lower Window Bound: {lower_bound} Upper Window Bound: {upper_bound}"
        )
        logger.info(f"Min: {dtmin} Max: {dtmax}")
        lower_mask = data <= lower_bound
        upper_mask = data > upper_bound
        mid_mask = lower_mask ^ upper_mask
        data_copy = data.copy()
        # Apply thresholds
        data_copy[~mid_mask] = (
            (data_copy[~mid_mask] - (center - 0.5)) / adjusted_width + 0.5
        ) * (dtmax - dtmin) + dtmin
        data_copy[lower_mask] = dtmin
        data_copy[upper_mask] = dtmax
        return data_copy

    @staticmethod
    def apply_linear_exact_window_center_width(
        data, center, width, dtmin=0, dtmax=255
    ):
        """Filters data by the specified center and width using the DICOM linear exact equation.
        See `C.11.2.1.2.1 LINEAR_EXACT Function <https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.11.2.html#sect_C.11.2.1.3.2>`_.

        ::

            if (x <= c - w/2), then y = ymin
            else if (x > c + w/2), then y = ymax
            else y = ((x - c) / w + 0.5) * (ymax- ymin) + ymin

        ..info::

            #. Window Width (0028,1051) shall always be greater than 0

        This equation is similar to described in Radiopaedia

        ::

            Murphy A, Wilczek M, Feger J, et al. Windowing (CT). Reference article,
            Radiopaedia.org (Accessed on 24 Jan 2025) https://doi.org/10.53347/rID-52108

        Args:
            data (np.ndarray): pixel array containing the data to perform peak extraction on
            center (int): The desired Window Center setting.
            width (int): The desired Window Width setting.
            dtmin (int): The min value of datatype.
            dtmax (int): The max value of datatype.

        Returns:
            np.ndarray: Windowed data
        """
        width = 1 if width <= 0 else width
        logger.info(
            f"Applying Window Settings using the Linear Exact method => Center: {center} Width: {width}"
        )
        half_width = width / 2
        lower_bound = center - half_width
        upper_bound = center + half_width
        logger.info(
            f"Half Width: {half_width} Lower Window Bound: {lower_bound} Upper Window Bound: {upper_bound}"
        )
        logger.info(f"Min: {dtmin} Max: {dtmax}")
        lower_mask = data <= lower_bound
        upper_mask = data > upper_bound
        mid_mask = lower_mask ^ upper_mask
        data_copy = data.copy()
        # Apply thresholds
        data_copy[~mid_mask] = (
            (data_copy[~mid_mask] - center) / width + 0.5
        ) * (dtmax - dtmin) + dtmin
        data_copy[lower_mask] = dtmin
        data_copy[upper_mask] = dtmax
        return data_copy

    @staticmethod
    def apply_sigmoid_window_center_width(
        data, center, width, dtmin=0, dtmax=255
    ):
        """Filters data by the specified center and width using the DICOM sigmoid equation.
        See `C.11.2.1.2.1 SIGMOID Function <https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.11.2.html#sect_C.11.2.1.3.1>`_.

        ::

            y = ((ymax - ymin) / (1 + exp(-4 * (x - c) / w))) + ymin

        Args:
            data (np.ndarray): pixel array containing the data to perform peak extraction on
            center (int): The desired Window Center setting.
            width (int): The desired Window Width setting.
            dtmin (int): The min value of datatype.
            dtmax (int): The max value of datatype.

        Returns:
            np.ndarray: Windowed data
        """
        logger.info(
            f"Applying Window Settings using the Sigmoid method => Center: {center} Width: {width}"
        )
        mid_mask = dtmin <= data <= dtmax
        data_copy = data.copy()
        # Apply thresholds
        data_copy[~mid_mask] = (
            (dtmax - dtmin)
            / (1 + np.exp((-4 * data_copy[~mid_mask] - center) / width))
        ) + dtmin
        return data_copy

    @staticmethod
    def apply_clip_window_center_width(
        data, center, width, dtmin=0, dtmax=255
    ):
        """Filters data by the specified center and width using the custom clip method. This method is not a standard
        DICOM windowing method. It is a modified form of the linear_exact method. We basically do not rescale the window
        data. It uses numpy.clip() to generate the window data.

        Somehow, this works better than the linear method for the object detectability task.

        This method is similar to described in Radiopaedia

        ::

            Murphy A, Wilczek M, Feger J, et al. Windowing (CT). Reference article,
            Radiopaedia.org (Accessed on 24 Jan 2025) https://doi.org/10.53347/rID-52108

        Args:
            data (np.ndarray): pixel array containing the data to perform peak extraction on
            center (int): The desired Window Center setting.
            width (int): The desired Window Width setting.
            dtmin (int): The min value of datatype.
            dtmax (int): The max value of datatype.

        Returns:
            np.ndarray: Windowed data
        """
        logger.info(
            f"Applying Window Settings using the Clip method => Center: {center} Width: {width}"
        )
        half_width = width / 2
        upper_grey = center + half_width
        lower_grey = center - half_width
        logger.info(
            f"Half Width: {half_width} Lower Window Bound: {lower_grey} Upper Window Bound: {upper_grey}"
        )
        logger.info(f"Min: {dtmin} Max: {dtmax}")
        upper_mask = data > upper_grey
        lower_mask = data <= lower_grey
        mid_mask = lower_mask ^ upper_mask
        data_copy = data.copy()
        # Apply thresholds
        data_copy[~mid_mask] = np.clip(
            data_copy[~mid_mask], lower_grey, upper_grey
        )
        data_copy[lower_mask] = dtmin
        data_copy[upper_mask] = dtmax
        return data_copy

    @staticmethod
    def apply_window_center_width(
        data, center, width, function="linear", dtype=None
    ):
        """Filters data by the specified center and width. We support 3 functions defined by the DICOM standard. These
        functions are linear (default), linear_exact, and sigmoid.

        See `DICOM Chapter 11.2 <https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.11.2.html>`_
        for more details.

        Each specific function is defined as apply_<function>_window_center_width and each has a copy of the DICOM
        documentation. See those for details as well.

        Args:
            data (np.ndarray): pixel array containing the data to perform peak extraction on
            center (int): The desired Window Center setting.
            width (int): The desired Window Width setting.
            function (str): Function t apply to data windowing. Defaults to linear.
            dtype (np.dtype): The data's natural datatype. We use the input's datatype if no specific data type is
                specified. This parameter is mainly present for other higher level methods to use to maintain
                logical consistency during processing.

        Returns:
            np.ma.MaskedArray: Windowed data.

        """
        dtype = data.dtype if dtype is None else dtype
        dtype_min = get_datatype_min(dtype)
        dtype_max = get_datatype_max(dtype)
        if function == "linear_exact":
            return ACRObject.apply_linear_exact_window_center_width(
                data, center, width, dtype_min, dtype_max
            )
        if function == "sigmoid":
            return ACRObject.apply_sigmoid_window_center_width(
                data, center, width, dtype_min, dtype_max
            )
        if function == "clip":
            return ACRObject.apply_clip_window_center_width(
                data, center, width, dtype_min, dtype_max
            )
        return ACRObject.apply_linear_window_center_width(
            data, center, width, dtype_min, dtype_max
        )

    @staticmethod
    def compute_center_and_width(data):
        """Automatically resolves the center and width settings from the given data. If you wish to derive the
        center and width from roi

        Args:
            data (np.ndarray|np.ma.MaskedArray): pixel array containing the data to perform center and width calculation

        Returns:
            tuple of int:
                center (float): The desired Window Center setting. \n
                width (float): The desired Window Width setting.

        """
        return np.round(ACRObject.compute_histogram_mean(data)), np.round(
            ACRObject.compute_histogram_width(data)
        )

    @staticmethod
    def compute_histogram_mode(data):
        """Computes the mode of the given dataset using a 256 bins histogram. This method ignores the zeros.

        Args:
            data (np.ndarray|np.ma.MaskedArray): pixel array containing the data

        Returns:
            mode (float): non-zero mode of the dataset.

        """
        search_data = data[data > 0]
        hist, bins = np.histogram(search_data, bins=256)
        mode = bins[np.argmax(hist)]
        logger.info(f"Histogram mode: {mode}")
        return mode

    @staticmethod
    def compute_histogram_mean(data):
        """Computes the mean of the given dataset using a 256 bins histogram. This method ignores the zeros.

        Args:
            data (np.ndarray|np.ma.MaskedArray): pixel array containing the data

        Returns:
            mean (float): non-zero mean of the dataset.

        """
        search_data = data[data > 0]
        hist, bins = np.histogram(search_data, bins=256)
        edges = np.histogram_bin_edges(search_data, bins=256)
        mean_bins = np.mean(np.vstack([edges[:-1], edges[1:]]), axis=0)
        mean = np.quantile(
            mean_bins, 0.945, method="inverted_cdf", weights=hist
        )
        # mean = np.quantile(mean_bins, 0.90, method='inverted_cdf', weights=hist)
        logger.info(f"Histogram mean: {mean}")
        return mean

    @staticmethod
    def compute_histogram_width(data):
        """Computes the width of the given dataset using a 256 bins histogram.
        This method ignores the zeros. The width is computed as bins.max() - bins.min()

        Args:
            data (np.ndarray|np.ma.MaskedArray): pixel array containing the data

        Returns:
            std (float): non-zero std of the dataset.

        """
        search_data = data[data > 0]
        hist, bins = np.histogram(search_data, bins=256)
        std = bins.max() - bins.min()
        logger.info(f"Histogram width: {std}")
        return std

    @staticmethod
    def compute_percentile(data, percentile):
        """Computes the mode of the given dataset using a 100 bins histogram. This method ignores the zeros.

        Args:
            data (np.ndarray|np.ma.MaskedArray): pixel array containing the data

        Returns:
            mode (float): non-zero mode of the dataset.

        """
        try:
            non_zero_data = data[np.nonzero(data)]
            return np.percentile(non_zero_data, percentile)
        except (IndexError, ValueError, TypeError):
            return 0

    @staticmethod
    def compute_percentile_median(data, percentile):
        """Computes the mode of the given dataset using a 100 bins histogram. This method ignores the zeros.

        Args:
            data (np.ndarray|np.ma.MaskedArray): pixel array containing the data

        Returns:
            mode (float): non-zero mode of the dataset.

        """
        perc = ACRObject.compute_percentile(data, percentile)
        perc_data = data[data >= perc]
        return np.median(perc_data)

    @staticmethod
    def threshold_data(data, intensity, fill=0, greater_than=False):
        """Thresholds the data. Meaning, every pixel with value < intensity or > intensity will be replaced by the
        value in fill. Which side to fill with fill value is driven by the greater_than flag. By default,
        we do the former.

        Args:
            data (np.ndarray|np.ma.MaskedArray): pixel array containing the data
            intensity (float): pixel value to use as the threshold
            fill (float, optional): pixel value to use as replacement. Defaults to 0.
            greater_than (bool, optional): Do we fill values that are greater than the specified threshold? Defaults to False.

        Returns:
            data (np.ndarray|np.ma.MaskedArray): data reference.

        """
        mask = data >= intensity if greater_than else data <= intensity
        data[mask] = fill
        return data

    @staticmethod
    def find_nearest_value(data, value):
        """Does a search for the closest value to the one supplied.

        Args:
            data (np.ndarray|np.ma.MaskedArray): pixel array containing the data
            value (float): pixel value to use in lookup

        Returns:
            data (np.ndarray|np.ma.MaskedArray): data reference.

        """
        logger.info(f"value {value}")
        search_data = data.flatten() - value
        candidate_data = search_data[search_data >= 0]
        i = np.argwhere(search_data == candidate_data[0]).flatten().min()
        logger.info(f"result {data[i]}")
        return data[i]

    @staticmethod
    def filter_with_dog(
        data, sigma1=1, sigma2=2, gamma=1.0, iterations=1, ksize=(0, 0)
    ):
        """Performs two Gaussian convolutions with each taking a sigma. Subtracts the second from the first. The idea is
        to eliminate noise.

        Steps:
        ______

            #. Copy the input data
            #. Normalize input data to range [0, 1.0]
            #. Offset results by 0.5 to avoid numpy.power() error message if gamma is not 1.
            #. Apply gamma correction
            #. Obtain 1st Gaussian blurred image using sigma1 for both sigmaX and sigmaY.
            #. Obtain 2nd Gaussian blurred image using sigma2 for both sigmaX and sigmaY.
            #. Subtract => 1 - 2
            #. Remove the offset
            #. Repeat 4 - 8 for n iterations
            #. Restore results intensity range to the input's data type.
            #. Return results

        Notes:
        ______

        Per my testing, this implementation is equivalent to `skimage.filters.difference_of_gaussians` when results are
        binarized. However, there is one fundamental difference and that is that the results there come centered as a
        bellshape which preserve grays. My implementation does not do that. It truly generates pixel subtractions
        for the same input.

        To better mirror the way GIMP implements a difference of Gaussians, I added gamma correction using the
        `Power Law Transform <https://pyimagesearch.com/2015/10/05/opencv-gamma-correction/>`_ on the blurred
        intermediates. The idea is that we can force some of the pixels that are closer to the background closer to 0,
        which can be filtered out elsewhere in your algorithm. Conversely, if you use a larger gamma value, you can
        increase the intensity of pixels and thus bias the output signal towards more surviving pixels if the output
        is meant to be used in a subtraction or threshold operation. Leave the gamma as sis if you simply want
        a DoG without gamma correction.

        An initial offset of 0.5 is applied to the data if gamma != 1.0 to avoid the following error in numpy.power()!

        .. error::

            line 4658, in _lerp
            subtract(b, diff_b_a * (1 - t), out=lerp_interpolation, where=t >= 0.5,
            ValueError: output array is read-only

        Args:
            data (np.ndarray|np.ma.MaskedArray): pixel array containing the data
            sigma1 (float, optional): sigma for the first Gaussian operation. Defaults to 1.
            sigma2 (float, optional): sigma for the second Gaussian operation. This should be bigger than the first value. Defaults to 2.
            gamma (float, optional): value to multiply against each resulting difference to enhance contrast. Defaults to 1.
            iterations (int, optional): How many DoG passes to do. Defaults to 1.

        Returns:
            data (np.ndarray|np.ma.MaskedArray): data reference.

        """
        dtype = data.dtype
        working_data = ACRObject.normalize(
            data.copy(), max=1, dtype=cv2.CV_32FC1
        )
        for i in range(iterations):
            working_data = ACRObject.apply_gamma_correction(
                working_data, gamma
            )
            blurred = cv2.GaussianBlur(
                working_data, ksize, sigmaX=sigma1, sigmaY=sigma1
            )
            blurred2 = cv2.GaussianBlur(
                blurred, ksize, sigmaX=sigma2, sigmaY=sigma2
            )
            working_data = cv2.subtract(blurred, blurred2)
        working_data = expand_data_range(working_data, target_type=dtype)
        return working_data

    @staticmethod
    def apply_gamma_correction(data, gamma):
        """Applies a gamma correction or Power Law to affect the contrast of pixels.

        Args:
            data (np.ndarray): image data to upsample.
            gamma (float): factor by which to correct the lightness of the image. A value less than 1 darkens the image.
                        A value of 1 has no effect. A value larger than 1 makes the image lighter.

        Returns:
            np.ndarray: gamma corrected image.
        """
        g = 1 / gamma
        correction = 0.5 if gamma != 1.0 else 0
        return np.power(data + correction, g)

    @staticmethod
    def filter_with_gaussian(data, sigma=1, ksize=(0, 0), dtype=np.uint16):
        """Applies a Gaussian filter to the input image. The result is then expanded to a requested data type's range.
        If dtype is the same as data's native type this step will ensure values are normalized/scaled to fit in the
        type's range.

        Args:
            data (np.ndarray): image data to upsample.
            sigma (float): sigma value to apply in both dimensions.
            ksize (tuple of int): size of the Gaussian kernel. The default value let's OpenCv know it should autodetect
                        the kernel size, which can be optimal in many situations.
            dtype (np.dtype): numpy data type to expand result range to. Keep it as original if you would like to keep
                like the input's.

        Returns:
            np.ndarray: smoothed image.
        """
        noise_removed = cv2.GaussianBlur(
            data,
            ksize=ksize,
            sigmaX=sigma,
            sigmaY=sigma,
            borderType=cv2.BORDER_ISOLATED,
        )
        return expand_data_range(noise_removed, target_type=dtype)

    @staticmethod
    def normalize(data, max=255, dtype=cv2.CV_8U):
        """Normalizes the input data to a max value of a given data type.

        Args:
            data (np.ndarray): image data to upsample.
            max (float): max value to normalize the data to. Defaults to 255 which is the max in an 8bit target data set.
            dtype (np.dtype): data type to return for normalization. Defaults to 8bit

        Returns:
            np.ndarray: resampled image.
        """
        return cv2.normalize(
            src=data,
            dst=None,
            alpha=0,
            beta=max,
            norm_type=cv2.NORM_MINMAX,
            dtype=dtype,
        )

    @staticmethod
    def resample(data, dx=1, dy=1):
        """Resamples input image using OpenCV by some pixel/voxel resolution factor. This factor can be applied
        independently in the x and y directions to generate uneven resampling.

        Args:
            data (np.ndarray): image data to upsample.
            dx (float): integer factor by which to resample in the x dimension. Defaults to 1.
            dy (float): integer factor by which to resample in the x dimension. Defaults to 1.

        Returns:
            np.ndarray: resampled image.
        """
        return cv2.resize(
            data, dsize=None, fx=dx, fy=dy, interpolation=cv2.INTER_CUBIC
        )

    @staticmethod
    def zoom(data, level=1):
        """Simulates zooming into an image by upsampling it to a given factor.

        Args:
            data (np.ndarray): image data to upsample
            level (float): integer factor by which to upsample. For example, 3 => 3x the zoom.

        Returns:
            np.ndarray: binarized image.
        """
        return ACRObject.resample(data, dx=level, dy=level)

    @staticmethod
    def binarize_image(img, percentile=95):
        """Binarizes an input image using a percentile from the histogram as threshold. The default is to look for the
        95th percentile value and threshold against that. The resulting image contains only zeros and 255.

        Args:
            img (np.ndarray): image data to binarize
            percentile (float): the cutoff level at which to binarize.

        Returns:
            np.ndarray: binarized image.
        """
        bin = expand_data_range(img, target_type=np.uint8)
        thr = ACRObject.compute_percentile(bin, percentile)
        logger.info(f"Binarization threshold selected => {thr}")
        bin[bin > thr] = 255
        bin[bin <= thr] = 0
        return bin

    @staticmethod
    def crop_image(img, x, y, width, height=None, mode="center"):
        """Return a rectangular subset of a pixel array

        Args:
            img (np.ndarray): image data to crop
            x (int): x coordinate of centre
            y (int): y coordinate of centre
            width (int): width of box
            height (int, optional): height of box. Will be set to width if None, Defaults to None
            mode (str, optional): defines what the x,y coordinates represent. Is it the center of the ROI or the corner
                                (top, left), Defaults to center

        Returns:
            np.ndarray: subset of a pixel array with given width
        """
        height = width if height is None else height
        if isinstance(mode, str) and mode == "center":
            crop_x, crop_y = (
                (int(x - width / 2), int(x + width / 2)),
                (
                    int(y - height / 2),
                    int(y + height / 2),
                ),
            )
        else:
            crop_x, crop_y = (
                (int(x), int(x + width)),
                (
                    int(y),
                    int(y + height),
                ),
            )

        crop_img = img[crop_y[0] : crop_y[1], crop_x[0] : crop_x[1]]

        return crop_img

    @staticmethod
    def invert_image(img):
        """Returns the image with its pixels inverted. This method creates a copy of the input image so the inversion
        does not affect the original input.

        Args:
            img (np.ndarray): image data to invert

        Returns:
            np.ndarray: subset of a pixel array with given width
        """
        inverted = img.copy()
        inverted = cv2.bitwise_not(inverted)
        return inverted

    @staticmethod
    def calculate_FWHM(line):
        """Calculate full width at half maximum of the line profile.

        I used a modified version of the method outlined here https://typethepipe.com/post/measure-fwhm-image-with-python/

        Args:
            line (np.ndarray): slice profile curve.

        Returns:
            tuple: center x coordinate alongside the line profile, fwhm (which is the width of the are we care about
            in the profile).
        """
        default_cx = int(np.round(len(line) / 2))
        default_fwhm = 990
        try:
            # Step 1, compute edges where the ramp curve begins and ends.
            # Note, due to imperfections in input, you might have a few weird spikes in the neighborhood but this is ok.
            # Why? Because we will look for the midpoint next as an estimate of the pixel intensity we need to filter by.
            diff = np.diff(line)
            abs_diff_profile = np.absolute(diff)

            # Step 2, find a set of top peaks. We need at least two for the boundary of the roi, but I selected 10 to give
            # ample space to pick up noise without losing accuracy.
            peak_data = ACRObject.find_n_highest_peaks(abs_diff_profile, 10)

            # Step 3, find the midpoint which is very likely to be somewhere inside the roi.
            peaks = peak_data[0]
            peak = int(np.round((peaks[0] + peaks[-1]) / 2))

            # Step 4, Get pixel intensity and calculate the half max intensity.
            peak_val = line[peak]
            half_max = peak_val / 2

            # Step 5, find the samples along the line that have a pixel intensity higher or similar to the half max.
            horizontal_half = np.where(line >= half_max)[0]

            # Step 6, calculate the width in this region
            fwhm = horizontal_half[-1] - horizontal_half[0] + 1

            # Step 7, calculate true center of peak considered.
            # I don't use the raw peak as the center because that could be biased by outliers.
            cx = horizontal_half[0] + np.round(fwhm / 2)

            return cx, fwhm
        except Exception as w:
            logger.warning(w)
            logger.warning(
                "Received an empty line sample. This often happens if region of interest is empty."
            )
            logger.warning(
                "Defaulting line x coordinate to {} and fwhm to {}!".format(
                    default_cx, default_fwhm
                )
            )
        return default_cx, default_fwhm

    @staticmethod
    def calculate_MTF(erf, dx, dy):
        """Calculate MTF

        Args:
            erf (np.array): array of ?

        Returns:
            tuple: freq, lsf, MTF
        """
        lsf = np.diff(erf)
        N = len(lsf)
        n = (
            np.arange(-N / 2, N / 2)
            if N % 2 == 0
            else np.arange(-(N - 1) / 2, (N + 1) / 2)
        )

        resamp_factor = 8
        Fs = 1 / (np.sqrt(np.mean(np.square((dx, dy)))) * (1 / resamp_factor))
        freq = n * Fs / N
        MTF = np.abs(np.fft.fftshift(np.fft.fft(lsf)))
        MTF = MTF / np.max(MTF)

        zero_freq = np.where(freq == 0)[0][0]
        freq = freq[zero_freq:]
        MTF = MTF[zero_freq:]

        return freq, lsf, MTF
