"""
ACR SNR

Calculates the SNR for slice 7 (the uniformity slice) of the ACR phantom.

This script utilises the smoothed subtraction method described in McCann 2013 [1], and a standard subtraction SNR.

Created by Neil Heraghty (Adapted by Yassine Azma, yassine.azma@rmh.nhs.uk)

09/01/2023

[1] McCann, A. J., Workman, A., & McGrath, C. (2013). A quick and robust
method for measurement of signal-to-noise ratio in MRI. Physics in Medicine
& Biology, 58(11), 3775.
"""

import os
import sys
import traceback
import pydicom

import numpy as np
from scipy import ndimage

import hazenlib.utils
from hazenlib.HazenTask import HazenTask
from hazenlib.ACRObject import ACRObject


class ACRSNR(HazenTask):
    """Signal-to-noise ratio measurement class for DICOM images of the ACR phantom."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ACR_obj = ACRObject(self.dcm_list)
        # measured slice width is expected to be a floating point number
        try:
            self.measured_slice_width = float(kwargs["measured_slice_width"])
        except:
            self.measured_slice_width = None

        # subtract is expected to be a path to a folder
        try:
            if os.path.isdir(kwargs["subtract"]):
                self.subtract = kwargs["subtract"]
        except:
            self.subtract = None

    def run(self) -> dict:
        """Main function for performing SNR measurement using slice 7 from the ACR phantom image set. Performs either
        smoothing or subtraction method depending on user-provided input.

        Notes:
            Uses the smoothing method by default or the subtraction method if a second set of images are provided (using the --subtract option with dataset in a separate folder).

        Returns:
            dict: results are returned in a standardised dictionary structure specifying the task name, input DICOM Series Description + SeriesNumber + InstanceNumber, task measurement key-value pairs, optionally path to the generated images for visualisation.
        """
        # Identify relevant slice
        snr_dcm = self.ACR_obj.slice_stack[6]
        # Initialise results dictionary
        results = self.init_result_dict()

        # SINGLE METHOD (SMOOTHING)
        if self.subtract is None:
            try:
                results["file"] = self.img_desc(snr_dcm)
                snr, normalised_snr = self.snr_by_smoothing(
                    snr_dcm, self.measured_slice_width
                )
                results["measurement"]["snr by smoothing"] = {
                    "measured": round(snr, 2),
                    "normalised": round(normalised_snr, 2),
                }
            except Exception as e:
                print(
                    f"Could not calculate the SNR for {self.img_desc(snr_dcm)} because of : {e}"
                )
                traceback.print_exc(file=sys.stdout)
        # SUBTRACTION METHOD
        else:
            # Get the absolute path to all FILES found in the directory provided
            filepaths = [
                os.path.join(self.subtract, f)
                for f in os.listdir(self.subtract)
                if os.path.isfile(os.path.join(self.subtract, f))
            ]
            data2 = [pydicom.dcmread(dicom) for dicom in filepaths]
            snr_dcm2 = ACRObject(data2).slice_stack[6]
            results["file"] = [self.img_desc(snr_dcm), self.img_desc(snr_dcm2)]
            try:
                snr, normalised_snr = self.snr_by_subtraction(
                    snr_dcm, snr_dcm2, self.measured_slice_width
                )

                results["measurement"]["snr by subtraction"] = {
                    "measured": round(snr, 2),
                    "normalised": round(normalised_snr, 2),
                }
            except Exception as e:
                print(
                    f"Could not calculate the SNR for {self.img_desc(snr_dcm)} and "
                    f"{self.img_desc(snr_dcm2)} because of : {e}"
                )
                traceback.print_exc(file=sys.stdout)

        # only return reports if requested
        if self.report:
            results["report_image"] = self.report_files

        return results

    def get_normalised_snr_factor(self, dcm, measured_slice_width=None) -> float:
        """Calculates the normalisation factor to be applied to the SNR in order to obtain the absolute SNR (ASNR). The
        normalisation factor depends on voxel size, bandwidth, number of averages and number of phase encoding steps.

        Args:
            dcm (pydicom.Dataset): DICOM image object
            measured_slice_width (float, optional): Provide the true slice width for the set of images. Defaults to None.

        Returns:
            float: normalisation factor.
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
        """Apply filtering to a pixel array (image), as per the single image SNR method outlined in McCann et al, 2013.

        Notes:
            Performs a 2D convolution (for filtering images), using uniform_filter (SciPy function).

        Args:
            dcm (pydicom.Dataset): DICOM image object.

        Returns:
            np.array: pixel array of the filtered image.
        """
        a = dcm.pixel_array.astype("int")

        # filter size = 9, following MATLAB code and McCann 2013 paper for head coil, although note McCann 2013
        # recommends 25x25 for body coil.
        # TODO add coil options, same as with MagNet SNR
        filtered_array = ndimage.uniform_filter(a, 25, mode="constant")
        return filtered_array

    def get_noise_image(self, dcm: pydicom.Dataset) -> np.array:
        """Get a noise image when only one set of DICOM data is available.

        Notes:
            Separates the image noise by smoothing the pixel array and subtracting the smoothed pixel array from the
            original, leaving only the noise.

        Args:
            dcm (pydicom.Dataset): DICOM image object

        Returns:
            np.array: pixel array representing the image noise.
        """
        a = dcm.pixel_array.astype("int")

        # Convolve image with boxcar/uniform kernel
        imsmoothed = self.filtered_image(dcm)

        # Subtract smoothed array from original
        imnoise = a - imsmoothed

        return imnoise

    def get_roi_samples(
        self, ax, dcm: pydicom.Dataset or np.ndarray, centre_col: int, centre_row: int
    ) -> list:
        """Takes the pixel array and divides it into several rectangular regions of interest (ROIs). If 'ax' is provided, then a plot of the ROIs is generated.

        Args:
            ax (matplotlib.pyplot.subplots): matplotlib axis for visualisation.
            dcm (pydicom.Dataset or np.ndarray): DICOM image object, or its pixel array.
            centre_col (int): x coordinate of the centre.
            centre_row (int): y coordinate of the centre.

        Returns:
            list of np.array: subsets of the original pixel array.
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

    def snr_by_smoothing(
        self, dcm: pydicom.Dataset, measured_slice_width=None
    ) -> float:
        """Obtains a noise image using the single-image smoothing technique. Generates a ROI within the phantom region
        of the pixel array. Then measures the mean signal within the ROI on the original pixel array, and the standard
        deviation within the ROI on the noise image. Calculates SNR using these values and multiplies the SNR by the
        normalisation factor.

        Args:
            dcm (pydicom.Dataset): DICOM image object.
            measured_slice_width (float, optional): Provide the true slice width for the set of images. Defaults to None.

        Returns:
            float: normalised_snr.
        """
        (centre_x, centre_y), _ = self.ACR_obj.find_phantom_center(
            dcm.pixel_array, self.ACR_obj.dx, self.ACR_obj.dy
        )
        noise_img = self.get_noise_image(dcm)

        signal = [
            np.mean(roi)
            for roi in self.get_roi_samples(
                ax=None, dcm=dcm, centre_col=centre_x, centre_row=centre_y
            )
        ]

        noise = [
            np.std(roi, ddof=1)
            for roi in self.get_roi_samples(
                ax=None, dcm=noise_img, centre_col=centre_x, centre_row=centre_y
            )
        ]
        # note no root_2 factor in noise for smoothed subtraction (one image) method, replicating Matlab approach and
        # McCann 2013

        snr = np.mean(np.divide(signal, noise))

        normalised_snr = snr * self.get_normalised_snr_factor(dcm, measured_slice_width)

        if self.report:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 1)
            fig.set_size_inches(8, 16)
            fig.tight_layout(pad=4)

            axes[0].imshow(dcm.pixel_array)
            axes[0].scatter(centre_x, centre_y, c="red")
            axes[0].set_title("Centroid Location")

            axes[1].set_title("Smoothed Noise Image")
            axes[1].imshow(noise_img, cmap="gray")
            self.get_roi_samples(axes[1], dcm, centre_x, centre_y)

            img_path = os.path.realpath(
                os.path.join(self.report_path, f"{self.img_desc(dcm)}_smoothing.png")
            )
            fig.savefig(img_path)
            self.report_files.append(img_path)

        return snr, normalised_snr

    def snr_by_subtraction(
        self, dcm1: pydicom.Dataset, dcm2: pydicom.Dataset, measured_slice_width=None
    ) -> float:
        """Calculates signal to noise ratio using the two image subtraction method. Obtains a noise image by subtracting
        the two pixel arrays. Obtains ROIs within the phantom region of the pixel arrays. Calculates the mean within
        the ROI on one of the pixel arrays, and the standard deviation within the ROIs on the noise image.
        Calculates the SNR with these measurements and multiplies by the normalisation factor.

        Args:
            dcm1 (pydicom.Dataset): DICOM image object to calculate signal.
            dcm2 (pydicom.Dataset): DICOM image object to calculate noise.
            measured_slice_width (float, optional): Provide the true slice width for the set of images. Defaults to None.

        Returns:
            float: normalised_snr.
        """
        (centre_x, centre_y), _ = self.ACR_obj.find_phantom_center(
            dcm1.pixel_array, self.ACR_obj.dx, self.ACR_obj.dy
        )

        difference = np.subtract(
            dcm1.pixel_array.astype("int"), dcm2.pixel_array.astype("int")
        )

        signal = [
            np.mean(roi)
            for roi in self.get_roi_samples(
                ax=None, dcm=dcm1, centre_col=centre_x, centre_row=centre_y
            )
        ]
        noise = np.divide(
            [
                np.std(roi, ddof=1)
                for roi in self.get_roi_samples(
                    ax=None, dcm=difference, centre_col=centre_x, centre_row=centre_y
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

            fig, axes = plt.subplots(2, 1)
            fig.set_size_inches(8, 16)
            fig.tight_layout(pad=4)

            axes[0].imshow(dcm1.pixel_array)
            axes[0].scatter(centre_x, centre_y, c="red")
            axes[0].axis("off")
            axes[0].set_title("Centroid Location")

            axes[1].set_title("Difference Image")
            axes[1].imshow(
                difference,
                cmap="gray",
            )
            self.get_roi_samples(axes[1], dcm1, centre_x, centre_y)
            axes[1].axis("off")

            img_path = os.path.realpath(
                os.path.join(
                    self.report_path, f"{self.img_desc(dcm1)}_snr_subtraction.png"
                )
            )
            fig.savefig(img_path)
            self.report_files.append(img_path)

        return snr, normalised_snr
