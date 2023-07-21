"""
SNR(Im)

Calculates the SNR for a single-slice image of a uniform MRI phantom

This script utilises the smoothed subtraction method described in McCann 2013:
A quick and robust method for measurement of signal-to-noise ratio in MRI, Phys. Med. Biol. 58 (2013) 3775:3790

and a standard subtraction SNR.

Created by Neil Heraghty

04/05/2018
"""
import os
import cv2 as cv
import numpy as np
import pydicom
import skimage.filters
from scipy import ndimage

import hazenlib.utils
import hazenlib.exceptions as exc
from hazenlib.HazenTask import HazenTask
from hazenlib.logger import logger


class SNR(HazenTask):
    """Task to measure signal to noise ratio using a MagNET phantom
    Optional arguments: measured slice width

    Args:
        HazenTask: inherits from the HazenTask class
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, measured_slice_width=None) -> dict:
        """Main function to run task with specified args

        Returns:
            results (dict): dictionary of task - value pair and optionally
                        an images key with value listing image paths
        """
        results = {}
        # SUBTRACTION METHOD (when exactly 2 images are provided)
        if len(self.data) == 2:
            dcm1 = self.data[0]
            dcm2 = self.data[1]
            logger.info("Calculating SNR by subtraction for images {} and {}".format(
                self.key(dcm1), self.key(dcm2)
            ))
            snr = self.snr_by_subtraction(dcm1, dcm2, measured_slice_width)
            normalised_snr = self.get_normalised_snr(snr, dcm1, measured_slice_width)
            results["SNR by subtraction"] = {"files": [self.key(dcm1), self.key(dcm2)]}
            results["SNR by subtraction"]["measured"] = round(snr, 2)
            results["SNR by subtraction"]["normalised"] = round(normalised_snr, 2)

        results["SNR by smoothing"] = {}
        # SMOOTHING METHOD (one image at a time)
        for dcm in self.data:
            logger.info("Calculating SNR by smoothing for image {}".format(
                self.key(dcm)
            ))
            snr_results = {}
            snr = self.snr_by_smoothing(dcm, measured_slice_width)
            normalised_snr = self.get_normalised_snr(snr, dcm, measured_slice_width)
            snr_results["measured"] = round(snr, 2)
            snr_results["normalised"] = round(normalised_snr, 2)

            results["SNR by smoothing"][self.key(dcm)] = snr_results

        # only return reports if requested
        if self.report:
            results['reports'] = {'images': self.report_files}

        return results

    def two_inputs_match(self, dcm1: pydicom.Dataset, dcm2: pydicom.Dataset) -> bool: # NOT USED
        """
        Checks if two DICOMs are sufficiently similar

        Parameters
        ----------
        dcm1
        dcm2

        Returns
        -------

        """
        fields_to_match = ['StudyInstanceUID', 'RepetitionTime', 'EchoTime', 'FlipAngle']

        for field in fields_to_match:
            if dcm1.get(field) != dcm2.get(field):
                return False
        return True

    def get_normalised_snr(self, snr, dcm: pydicom.Dataset, measured_slice_width=None) -> float:
        """Calculates normalised SNR, based on a normalisation factor
        calculated following a method that matches MATLAB script.
        Utilises measured_slice_width if provided. Else finds from dcm.
        Finds dx, dy and bandwidth from dcm.
        Seeks to find TR, image columns and rows from dcm. Else uses default values.

        Args:
            snr (float): calculated SNR
            dcm (pydicom.Dataset): DICOM image object
            measured_slice_width (float, optional): measured slice width to be
                used in the calculation, defined in mm. Defaults to None.

        Returns:
            float: normalised SNR value
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
        voxel_factor = (1 / (0.001 * dx * dy * slice_thickness))

        normalised_snr_factor = bandwidth_factor * voxel_factor * (1 / (np.sqrt(averages * rows * (TR / 1000))))
        normalised_snr = snr * normalised_snr_factor

        return normalised_snr

    def filtered_image(self, arr: np.ndarray) -> np.array:
        """
        Performs a 2D convolution (for filtering images)
        uses uniform_filter SciPy function

        parameters:
        ---------------
        a: array to be filtered

        returns:
        ---------------
        filtered numpy array
        """

        # filter size = 9, following MATLAB code and McCann 2013 paper for head coil, although note McCann 2013 recommends 25x25 for body coil.
        filtered_array = ndimage.uniform_filter(arr, 25, mode='constant')
        return filtered_array

    def get_noise_image(self, arr: np.ndarray) -> np.array:
        """Separates the image noise by smoothing the image and subtracting
        the smoothed image from the original.
        Convert image into pixel array from dcmread and .pixel_array

        Args:
            arr (np.ndarray): pixel array

        Returns:
            Imnoise (np.array): pixel array representing the image noise
        """

        # Convolve image with boxcar/uniform kernel
        imsmoothed = self.filtered_image(arr)

        # Subtract smoothed array from original
        imnoise = arr - imsmoothed

        return imnoise

    def threshold_image(self, arr):
        """Create threshold image (only when phantom could not be detected)
        Convert image into pixel array from dcmread and .pixel_array
        Apply a filter skimage.filters.threshold_li and return masked image

        Args:
            arr (dcm.pixel_array): DICOM image object

        Returns:
            imthresholded: thresholded image
            mask: threshold mask
        """
        # threshold_li: Pixels > this value are assumed foreground
        threshold_value = skimage.filters.threshold_li(arr)
        mask = arr > threshold_value
        # Initialise array of zeros in shape of original image
        imthresholded = np.zeros_like(arr)
        imthresholded[mask] = arr[mask]

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
        """Find centroid coordinates of binary polygonal shape

        Args:
            binary_mask: mask of a shape

        Returns:
            tuple: centroid_coords (col:int, row:int)
        """

        from skimage.util import img_as_ubyte
        from skimage.measure import label, regionprops
        img = img_as_ubyte(binary_mask) > 0
        label_img = label(img, connectivity=img.ndim)
        props = regionprops(label_img)
        col = int(props[0].centroid[0])
        row = int(props[0].centroid[1])
        # print('Centroid coords [x,y] =', col, row)

        return int(col), int(row)

    def get_roi_samples(self, dcm: pydicom.Dataset or np.ndarray,
                        centre_col: int, centre_row: int) -> list:
        """Get pixel arrays for regions of interest

        Args:
            dcm (pydicom.Dataset or np.ndarray): original pixel array
            centre_col (int): y coordinate of ROI centre
            centre_row (int): x coordinate of ROI centre

        Returns:
            samples: list of pixel value arrays (subsets/ROI)
        """

        if type(dcm) == np.ndarray:
            data = dcm
        else:
            data = dcm.pixel_array

        # for array indexing: [row, column] format
        samples = [
            data[(centre_row - 10):(centre_row + 10), (centre_col - 10):(centre_col + 10)],
            data[(centre_row - 50):(centre_row - 30), (centre_col - 50):(centre_col - 30)],
            data[(centre_row + 30):(centre_row + 50), (centre_col - 50):(centre_col - 30)],
            data[(centre_row - 50):(centre_row - 30), (centre_col + 30):(centre_col + 50)],
            data[(centre_row + 30):(centre_row + 50), (centre_col + 30):(centre_col + 50)]
        ]

        return samples

    def get_object_centre(self, dcm: pydicom.Dataset) -> (int, int):
        """Find the phantom object within the image and return the coords of
        its centre (col and row) values.

        Args:
            dcm (pydicom.Dataset): DICOM image object

        Raises:
            e: _description_
            exc.ShapeError: _description_

        Returns:
            centre: (col:int, row:int)
        """
        # Shape Detection
        try:
            arr = dcm.pixel_array
            logger.debug('Performing phantom shape detection.')
            shape_detector = hazenlib.utils.ShapeDetector(arr=arr)
            orientation = hazenlib.utils.get_image_orientation(dcm.ImageOrientationPatient)

            if orientation in ['Sagittal', 'Coronal']:
                logger.debug('Orientation is sagittal or coronal.')
                # orientation is sagittal to patient
                try:
                    (col, row), size, angle = shape_detector.get_shape('rectangle')
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
            elif orientation == 'Transverse':
                logger.debug('Orientation is transverse.')
                try:
                    col, row, r = shape_detector.get_shape('circle')
                except exc.MultipleShapesError:
                    logger.info('Warning! Found multiple circles in image, will assume largest circle is phantom.')
                    col, row, r = self.get_largest_circle(shape_detector.shapes['circle'])
            else:
                raise exc.ShapeError("Unable to identify phantom shape.")

        # Threshold Detection
        except exc.ShapeError:
            arr = dcm.pixel_array # .astype('int')
            logger.info('Shape detection failed. Performing object centre measurement by thresholding.')
            _, mask = self.threshold_image(arr)
            row, col = self.get_binary_mask_centre(mask)

        return int(col), int(row)

    def snr_by_smoothing(self, dcm: pydicom.Dataset) -> float:
        """Calculate signal to noise ratio by subtraction

        Args:
            dcm1 (pydicom.Dataset): image 1
            dcm2 (pydicom.Dataset): image 2

        Returns:
            tuple: measured and normalised SNR values (float)
        """
        arr = dcm.pixel_array # .astype('int')

        col, row = self.get_object_centre(dcm=dcm)
        noise_img = self.get_noise_image(arr)

        signal = [np.mean(roi) for roi in self.get_roi_samples(
                    dcm=dcm, centre_col=col, centre_row=row)
                ]

        noise = [np.std(roi, ddof=1) for roi in self.get_roi_samples(
                    dcm=noise_img, centre_col=col, centre_row=row)
                ]
        # note no root_2 factor in noise for smoothed subtraction (one image) method,
        # replicating Matlab approach and McCann 2013

        snr = np.mean(np.divide(signal, noise))

        if self.report:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle
            from matplotlib.collections import PatchCollection
            fig, axes = plt.subplots(1, 1)
            fig.set_size_inches(5, 5)
            fig.tight_layout(pad=1)

            axes.set_title('smoothed noise image')
            axes.imshow(noise_img, cmap='gray', label='smoothed noise image')
            axes.scatter(col, row, 10, marker="+", label='centre')
            rects = [Rectangle((col - 10, row - 10), 20, 20),
                     Rectangle((col - 50, row - 50), 20, 20),
                     Rectangle((col + 30, row - 50), 20, 20),
                     Rectangle((col - 50, row + 30), 20, 20),
                     Rectangle((col + 30, row + 30), 20, 20)]
            pc = PatchCollection(rects, edgecolors='red', facecolors="None", label='ROIs')
            axes.add_collection(pc)
            axes.legend()

            img_path = os.path.realpath(os.path.join(self.report_path,
                            f'{self.key(dcm)}_smoothing.png'))
            fig.savefig(img_path)
            self.report_files.append(img_path)

        return snr

    def get_largest_circle(self, circles):
        largest_r = 0
        largest_col, largest_row = 0, 0
        for circle in circles:
            (col, row), r = cv.minEnclosingCircle(circle)
            if r > largest_r:
                largest_r = r
                largest_col, largest_row = col, row

        return largest_col, largest_row, largest_r

    def snr_by_subtraction(self, dcm1: pydicom.Dataset, dcm2: pydicom.Dataset) -> (float, float):
        """Calculate signal to noise ratio by subtraction

        Args:
            dcm1 (pydicom.Dataset): image 1
            dcm2 (pydicom.Dataset): image 2

        Returns:
            tuple: measured and normalised SNR values (float)
        """
        col, row = self.get_object_centre(dcm=dcm1)

        difference = np.subtract(
            dcm1.pixel_array.astype('int'), dcm2.pixel_array.astype('int')
            )
        # TODO losing accuracy by converting pixel values to int

        signal = [np.mean(roi) for roi in self.get_roi_samples(
                    dcm=dcm1, centre_col=col, centre_row=row)
                ]
        noise = np.divide(
                    [np.std(roi, ddof=1) for roi in self.get_roi_samples(
                        dcm=difference, centre_col=col, centre_row=row
                    )], np.sqrt(2))
        snr = np.mean(np.divide(signal, noise))

        if self.report:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle
            from matplotlib.collections import PatchCollection
            fig, axes = plt.subplots(1, 1)
            fig.set_size_inches(5, 5)
            fig.tight_layout(pad=1)

            axes.set_title('difference image')
            axes.imshow(difference, cmap='gray', label='difference image')
            axes.scatter(col, row, 10, marker="+", label='centre')
            # for patches: [column/x, row/y] format

            rects = [Rectangle((col - 10, row - 10), 20, 20),
                     Rectangle((col - 50, row - 50), 20, 20),
                     Rectangle((col + 30, row - 50), 20, 20),
                     Rectangle((col - 50, row + 30), 20, 20),
                     Rectangle((col + 30, row + 30), 20, 20)]
            pc = PatchCollection(rects, edgecolors='red', facecolors="None", label='ROIs')
            axes.add_collection(pc)
            axes.legend()

            img_name = f'{self.key(dcm1)}_snr_subtraction.png'
            img_path = os.path.realpath(os.path.join(self.report_path, img_name))
            fig.savefig(img_path)
            self.report_files.append(img_path)

        return snr
