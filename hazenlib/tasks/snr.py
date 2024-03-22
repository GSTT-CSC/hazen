"""
SNR(Im)

Calculates the SNR for a single-slice image of a uniform MRI phantom

This script utilises the smoothed subtraction method described in McCann 2013:
A quick and robust method for measurement of signal-to-noise ratio in MRI, Phys. Med. Biol. 58 (2013) 3775:3790


Created by Neil Heraghty

04/05/2018
"""
import os
import pydicom
import cv2 as cv
import numpy as np
import skimage.filters
from scipy import ndimage

import hazenlib.utils
import hazenlib.exceptions as exc
from hazenlib.HazenTask import HazenTask
from hazenlib.logger import logger


class SNR(HazenTask):
    """Signal-to-noise ratio measurement class for DICOM images of the MagNet phantom

    Inherits from HazenTask class
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # measured slice width is expected to be a floating point number
        try:
            self.measured_slice_width = float(kwargs["measured_slice_width"])
        except:
            self.measured_slice_width = None

        # Determining kernel size based on coil choice. Values of 9 and 25 come from McCann 2013 paper.
        try:
            coil = kwargs["coil"]
            if coil is None or coil.lower() in ["hc", "head"]:
                self.kernel_size = 9
            elif coil.lower() in ["bc", "body"]:
                self.kernel_size = 25
        except:
            self.kernel_size = 9

    def run(self) -> dict:
        """Main function for performing signal-to-noise ratio measurement

        Notes:
            Five square ROIs are created, one at the image centre, and four peripheral
            ROIs with their centres displaced at 45, 135, 225 and 315 degrees from the
            centre.

        Returns:
            dict: results are returned in a standardised dictionary structure specifying the task name, input DICOM Series Description + SeriesNumber + InstanceNumber, task measurement key-value pairs, optionally path to the generated images for visualisation
        """
        results = self.init_result_dict()
        results["file"] = [self.img_desc(img) for img in self.dcm_list]
        results["measurement"]["snr by smoothing"] = {}

        # SUBTRACTION METHOD with a pair of input files
        if len(self.dcm_list) == 2:
            snr, normalised_snr = self.snr_by_subtraction(
                self.dcm_list[0], self.dcm_list[1], self.measured_slice_width
            )
            results["measurement"]["snr by subtraction"] = {
                "measured": round(snr, 2),
                "normalised": round(normalised_snr, 2),
            }

        # SINGLE METHOD (SMOOTHING) for every input file
        for idx, dcm in enumerate(self.dcm_list):
            snr, normalised_snr = self.snr_by_smoothing(dcm, self.measured_slice_width)
            results["measurement"]["snr by smoothing"][self.img_desc(dcm)] = {
                "measured": round(snr, 2),
                "normalised": round(normalised_snr, 2),
            }

        # only return reports if requested
        if self.report:
            results["report_image"] = self.report_files

        return results

    def two_inputs_match(self, dcm1: pydicom.Dataset, dcm2: pydicom.Dataset) -> bool:
        """Check if two DICOMs are sufficiently similar, based on the following fields
            "StudyInstanceUID", "RepetitionTime", "EchoTime", "FlipAngle"

        Args:
            dcm1 (pydicom.Dataset): DICOM object to compare
            dcm2 (pydicom.Dataset): _description_

        Returns:
            bool: _description_
        """

        fields_to_match = [
            "StudyInstanceUID",
            "RepetitionTime",
            "EchoTime",
            "FlipAngle",
        ]

        for field in fields_to_match:
            if dcm1.get(field) != dcm2.get(field):
                return False
        return True

    def get_normalised_snr_factor(
        self, dcm: pydicom.Dataset, measured_slice_width=None
    ) -> float:
        """Calculates SNR normalisation factor.

        Notes:
            Method matches MATLAB script.
            Utilises user provided slice_width if provided. Else finds from dcm.
            Finds dx, dy and bandwidth from dcm.
            Seeks to find TR, image columns and rows from dcm. Else uses default values.

        Args:
            dcm (pydicom.Dataset): DICOM image object
            measured_slice_width (float or None): slice width from user input

        Returns:
            float: normalised snr factor
        """

        dx, dy = hazenlib.utils.get_pixel_size(dcm)
        bandwidth = hazenlib.utils.get_bandwidth(dcm)
        TR = hazenlib.utils.get_TR(dcm)
        rows = hazenlib.utils.get_rows(dcm)
        columns = hazenlib.utils.get_columns(dcm)

        if measured_slice_width:
            slice_thickness = measured_slice_width
        else:
            slice_thickness = hazenlib.utils.get_slice_thickness(dcm)

        averages = hazenlib.utils.get_average(dcm)
        bandwidth_factor = np.sqrt((bandwidth * columns / 2) / 1000) / np.sqrt(30)
        voxel_factor = 1 / (0.001 * dx * dy * slice_thickness)

        normalised_snr_factor = (
            bandwidth_factor
            * voxel_factor
            * (1 / (np.sqrt(averages * rows * (TR / 1000))))
        )

        return normalised_snr_factor

    def filtered_image(self, dcm: pydicom.Dataset) -> np.array:
        """Performs a 2D convolution (for filtering images)
        using uniform_filter SciPy function and a kernel size based on user input coil

        Args:
            dcm (pydicom.Dataset): DICOM image to be filtered

        Returns:
            np.array: filtered image pixel values
        """
        a = dcm.pixel_array.astype("int")

        # filter size = 9, following MATLAB code and McCann 2013 paper for head coil, although note McCann 2013 recommends 25x25 for body coil.

        # 9 for head coil, 25 for body coil
        filtered_array = ndimage.uniform_filter(a, self.kernel_size, mode="constant")
        return filtered_array

    def get_noise_image(self, dcm: pydicom.Dataset) -> np.array:
        """Separates the image noise
        by smoothing the image and subtracting the smoothed image from the original.

        Args:
            dcm (pydicom.Dataset): DICOM image to get noise from

        Returns:
            np.array: pixel array representing the image noise
        """
        a = dcm.pixel_array.astype("int")

        # Convolve image with boxcar/uniform kernel
        imsmoothed = self.filtered_image(dcm)

        # Subtract smoothed array from original
        imnoise = a - imsmoothed

        return imnoise

    def threshold_image(self, dcm: pydicom.Dataset):
        """Determine threshold and mask based on image

        Args:
            dcm (pydicom.Dataset): DICOM image to get noise from

        Returns:
            tuple of np.array: (imthresholded, mask)
                pixel array representing the image above threshold
                and a corresponding mask
        """
        a = dcm.pixel_array.astype("int")

        # threshold_li: Pixels > this value are assumed foreground
        threshold_value = skimage.filters.threshold_li(a)
        # print('threshold_value =', threshold_value)
        mask = a > threshold_value
        imthresholded = np.zeros_like(a)
        imthresholded[mask] = a[mask]

        # # For debugging: Threshold figures:
        # from matplotlib import pyplot as plt
        # plt.figure()
        # fig, ax = plt.subplots(2, 2)
        # ax[0, 0].imshow(a)
        # ax[0, 1].imshow(mask)
        # ax[1, 0].imshow(imthresholded)
        # ax[1, 1].imshow(a-imthresholded)
        # fig.savefig("../THRESHOLD.png")

        return imthresholded, mask

    def get_binary_mask_centre(self, binary_mask) -> (int, int):
        """Determine coordinates of binary polygonal shape's centre

        Args:
            binary_mask: mask of a shape

        Returns:
            tuple of int corresponding to centroid_coords: (col:int, row:int)
        """

        from skimage import util
        from skimage.measure import label, regionprops

        img = util.img_as_ubyte(binary_mask) > 0
        label_img = label(img, connectivity=img.ndim)
        props = regionprops(label_img)
        col = int(props[0].centroid[0])
        row = int(props[0].centroid[1])
        # print('Centroid coords [x,y] =', col, row)

        return int(col), int(row)

    def get_roi_samples(
        self, ax, dcm: pydicom.Dataset or np.ndarray, centre_col: int, centre_row: int
    ) -> list:
        """Determine region of interest from a pixel array

        Args:
            ax (matplotlib axes): diagram axis for visualisation with matplotlib
            dcm (pydicom.Dataset or np.ndarray): image pixel array
            centre_col (int): center coordinate column
            centre_row (int): center coordinate row

        Returns:
            list of np.ndarray: corresponding to pixel array subsets at predefined ROI
        """
        if type(dcm) == np.ndarray:
            data = dcm
        else:
            data = dcm.pixel_array

        sample = [None] * 5
        # for array indexing: [row, column] format
        sample[0] = data[
            (centre_row - 10) : (centre_row + 10), (centre_col - 10) : (centre_col + 10)
        ]
        sample[1] = data[
            (centre_row - 50) : (centre_row - 30), (centre_col - 50) : (centre_col - 30)
        ]
        sample[2] = data[
            (centre_row + 30) : (centre_row + 50), (centre_col - 50) : (centre_col - 30)
        ]
        sample[3] = data[
            (centre_row - 50) : (centre_row - 30), (centre_col + 30) : (centre_col + 50)
        ]
        sample[4] = data[
            (centre_row + 30) : (centre_row + 50), (centre_col + 30) : (centre_col + 50)
        ]

        if ax:
            from matplotlib.patches import Rectangle
            from matplotlib.collections import PatchCollection

            # for patches: [column/x, row/y] format

            rects = [
                Rectangle((centre_col - 10, centre_row - 10), 20, 20),
                Rectangle((centre_col - 50, centre_row - 50), 20, 20),
                Rectangle((centre_col + 30, centre_row - 50), 20, 20),
                Rectangle((centre_col - 50, centre_row + 30), 20, 20),
                Rectangle((centre_col + 30, centre_row + 30), 20, 20),
            ]
            pc = PatchCollection(
                rects, edgecolors="red", facecolors="None", label="ROIs"
            )
            ax.add_collection(pc)

        return sample

    def get_object_centre(self, dcm) -> (int, int):
        """Find the phantom object within the image and return its centre col and row value.
        Note first element in output = col, second = row.

        Args:
            dcm (pydicom.Dataset): DICOM image to get noise from

        Returns:
            tuple of int corresponding to centroid_coords: (col:int, row:int)
        """

        # Shape Detection
        try:
            logger.debug("Performing phantom shape detection.")
            shape_detector = hazenlib.utils.ShapeDetector(arr=dcm.pixel_array)
            orientation = hazenlib.utils.get_image_orientation(dcm)

            if orientation in ["Sagittal", "Coronal"]:
                logger.debug("Orientation = sagittal or coronal.")
                # orientation is sagittal to patient
                try:
                    (col, row), size, angle = shape_detector.get_shape("rectangle")
                except exc.ShapeError as e:
                    # shape_detector.find_contours()
                    # shape_detector.detect()
                    # contour = shape_detector.shapes['rectangle'][1]
                    # angle, centre, size = cv.minAreaRect(contour)
                    # print((angle, centre, size))
                    # im = cv.drawContours(dcm.pixel_array.copy(), [shape_detector.contours[0]], -1, (0, 255, 255), 10)
                    # plt.imshow(im)
                    # plt.savefig("rectangles.png")
                    # print(shape_detector.shapes.keys())
                    raise e
            elif orientation == "Transverse":
                logger.debug("Orientation = transverse.")
                try:
                    col, row, r = shape_detector.get_shape("circle")
                except exc.MultipleShapesError:
                    logger.info(
                        "Warning! Found multiple circles in image, will assume largest circle is phantom."
                    )
                    col, row, r = self.get_largest_circle(
                        shape_detector.shapes["circle"]
                    )
            else:
                raise exc.ShapeError("Unable to identify phantom shape.")

        # Threshold Detection
        except exc.ShapeError:
            logger.info(
                "Shape detection failed. Performing object centre measurement by thresholding."
            )
            _, mask = self.threshold_image(dcm)
            row, col = self.get_binary_mask_centre(mask)

        return int(col), int(row)

    def snr_by_smoothing(self, dcm: pydicom.Dataset, measured_slice_width=None):
        """Calculate signal to noise ratio by smoothing

        Args:
            dcm (pydicom.Dataset): DICOM image object
            measured_slice_width (float or None): slice width from user input

        Returns:
            tuple of float: SNR and normalised SNR values

        """
        col, row = self.get_object_centre(dcm)
        noise_img = self.get_noise_image(dcm)

        signal = [
            np.mean(roi)
            for roi in self.get_roi_samples(
                ax=None, dcm=dcm, centre_col=col, centre_row=row
            )
        ]

        noise = [
            np.std(roi, ddof=1)
            for roi in self.get_roi_samples(
                ax=None, dcm=noise_img, centre_col=col, centre_row=row
            )
        ]
        # note no root_2 factor in noise for smoothed subtraction (one image) method, replicating Matlab approach and McCann 2013

        snr = np.mean(np.divide(signal, noise))

        normalised_snr = snr * self.get_normalised_snr_factor(dcm, measured_slice_width)

        if self.report:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 1)
            fig.set_size_inches(5, 5)
            fig.tight_layout(pad=1)

            axes.set_title("smoothed noise image")
            axes.imshow(noise_img, cmap="gray", label="smoothed noise image")
            axes.scatter(col, row, 10, marker="+", label="centre")
            self.get_roi_samples(axes, dcm, col, row)
            axes.legend()

            img_path = os.path.realpath(
                os.path.join(self.report_path, f"{self.img_desc(dcm)}_smoothing.png")
            )
            fig.savefig(img_path)
            self.report_files.append(img_path)

        return snr, normalised_snr

    def get_largest_circle(self, circles):
        """Determine circle with largest radius from list of detected circles

        Args:
            circles (_type_): _description_

        Returns:
            tuple: of centre coordinates (col, row) and radius (float)
        """
        largest_col, largest_row, largest_r = 0, 0, 0
        for circle in circles:
            (col, row), r = cv.minEnclosingCircle(circle)
            if r > largest_r:
                largest_col, largest_row, largest_r = col, row, r

        return largest_col, largest_row, largest_r

    def snr_by_subtraction(
        self, dcm1: pydicom.Dataset, dcm2: pydicom.Dataset, measured_slice_width=None
    ):
        """Calculate signal to noise ratio by smoothing

        Args:
            dcm1 (pydicom.Dataset): DICOM image object for signal
            dcm2 (pydicom.Dataset): DICOM image object for noise calculation
            measured_slice_width (float or None): slice width from user input

        Returns:
            tuple of float: SNR and normalised SNR values
        """
        col, row = self.get_object_centre(dcm1)

        difference = np.subtract(
            dcm1.pixel_array.astype("int"), dcm2.pixel_array.astype("int")
        )

        signal = [
            np.mean(roi)
            for roi in self.get_roi_samples(
                ax=None, dcm=dcm1, centre_col=col, centre_row=row
            )
        ]
        noise = np.divide(
            [
                np.std(roi, ddof=1)
                for roi in self.get_roi_samples(
                    ax=None, dcm=difference, centre_col=col, centre_row=row
                )
            ],
            np.sqrt(2),
        )
        snr = np.mean(np.divide(signal, noise))

        normalised_snr = snr * self.get_normalised_snr_factor(
            dcm1, measured_slice_width
        )

        if self.report:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 1)
            fig.set_size_inches(5, 5)
            fig.tight_layout(pad=1)

            axes.set_title("difference image")
            axes.imshow(difference, cmap="gray", label="difference image")
            axes.scatter(col, row, 10, marker="+", label="centre")
            self.get_roi_samples(axes, dcm1, col, row)
            axes.legend()

            img_path = os.path.realpath(
                os.path.join(
                    self.report_path, f"{self.img_desc(dcm1)}_snr_subtraction.png"
                )
            )
            fig.savefig(img_path)
            self.report_files.append(img_path)

        return snr, normalised_snr
