"""
ACR SNR

Calculates the SNR for slice 7 (the uniformity slice) of the ACR phantom.

This script utilises the smoothed subtraction method described in McCann 2013:
A quick and robust method for measurement of signal-to-noise ratio in MRI, Phys. Med. Biol. 58 (2013) 3775:3790

and a standard subtraction SNR.

Created by Neil Heraghty (Adapted by Yassine Azma)

09/01/2023
"""

import sys
import traceback
import os

import hazenlib.utils
from hazenlib.HazenTask import HazenTask
from scipy import ndimage

import numpy as np
import pydicom
from scipy import ndimage

from hazenlib.HazenTask import HazenTask
from hazenlib.ACRObject import ACRObject


class ACRSNR(HazenTask):

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
        snr_dcm = self.ACR_obj.slice7_dcm
        # Initialise results dictionary
        results = self.init_result_dict()

        # SINGLE METHOD (SMOOTHING)
        if self.subtract is None:
            try:
                results['file'] = self.img_desc(snr_dcm)
                snr, normalised_snr = self.snr_by_smoothing(snr_dcm, self.measured_slice_width)
                results['measurement']['snr by smoothing'] = {
                    "measured": round(snr, 2),
                    "normalised": round(normalised_snr, 2)
                }
            except Exception as e:
                print(f"Could not calculate the SNR for {self.img_desc(snr_dcm)} because of : {e}")
                traceback.print_exc(file=sys.stdout)
        # SUBTRACTION METHOD
        else:
            # Get the absolute path to all FILES found in the directory provided
            filepaths = [os.path.join(self.subtract, f) for f in os.listdir(self.subtract) \
                        if os.path.isfile(os.path.join(self.subtract, f))]
            data2 = [pydicom.dcmread(dicom) for dicom in filepaths]
            snr_dcm2 = ACRObject(data2).slice7_dcm
            results['file'] = [self.img_desc(snr_dcm), self.img_desc(snr_dcm2)]
            try:
                snr, normalised_snr = self.snr_by_subtraction(
                                snr_dcm, snr_dcm2, self.measured_slice_width)

                results['measurement']['snr by subtraction'] = {
                    "measured": round(snr, 2),
                    "normalised": round(normalised_snr, 2)
                }
            except Exception as e:
                print(f"Could not calculate the SNR for {self.img_desc(snr_dcm)} and "
                      f"{self.img_desc(snr_dcm2)} because of : {e}")
                traceback.print_exc(file=sys.stdout)

        # only return reports if requested
        if self.report:
            results['report_image'] = self.report_files

        return results

    def get_normalised_snr_factor(self, dcm, measured_slice_width=None) -> float:
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
        return normalised_snr_factor

    def filtered_image(self, dcm: pydicom.Dataset) -> np.array:
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
        a = dcm.pixel_array.astype('int')

        # filter size = 9, following MATLAB code and McCann 2013 paper for head coil, although note McCann 2013
        # recommends 25x25 for body coil.
        filtered_array = ndimage.uniform_filter(a, 25, mode='constant')
        return filtered_array

    def get_noise_image(self, dcm: pydicom.Dataset) -> np.array:
        """
        Separates the image noise by smoothing the image and subtracting the smoothed image
        from the original.

        parameters:
        ---------------
        a: image array from dcmread and .pixelarray

        returns:
        ---------------
        Imnoise: image representing the image noise
        """
        a = dcm.pixel_array.astype('int')

        # Convolve image with boxcar/uniform kernel
        imsmoothed = self.filtered_image(dcm)

        # Subtract smoothed array from original
        imnoise = a - imsmoothed

        return imnoise

    def get_roi_samples(self, ax, dcm: pydicom.Dataset or np.ndarray, centre_col: int, centre_row: int) -> list:

        if type(dcm) == np.ndarray:
            data = dcm
        else:
            data = dcm.pixel_array

        sample = [None] * 5
        # for array indexing: [row, column] format
        sample[0] = data[(centre_row - 10):(centre_row + 10), (centre_col - 10):(centre_col + 10)]
        sample[1] = data[(centre_row - 50):(centre_row - 30), (centre_col - 50):(centre_col - 30)]
        sample[2] = data[(centre_row + 30):(centre_row + 50), (centre_col - 50):(centre_col - 30)]
        sample[3] = data[(centre_row - 50):(centre_row - 30), (centre_col + 30):(centre_col + 50)]
        sample[4] = data[(centre_row + 30):(centre_row + 50), (centre_col + 30):(centre_col + 50)]

        if ax:
            from matplotlib.patches import Rectangle
            from matplotlib.collections import PatchCollection
            # for patches: [column/x, row/y] format

            rects = [Rectangle((centre_col - 10, centre_row - 10), 20, 20),
                     Rectangle((centre_col - 50, centre_row - 50), 20, 20),
                     Rectangle((centre_col + 30, centre_row - 50), 20, 20),
                     Rectangle((centre_col - 50, centre_row + 30), 20, 20),
                     Rectangle((centre_col + 30, centre_row + 30), 20, 20)]
            pc = PatchCollection(rects, edgecolors='red', facecolors="None", label='ROIs')
            ax.add_collection(pc)

        return sample

    def snr_by_smoothing(self, dcm: pydicom.Dataset, measured_slice_width=None) -> float:
        """

        Parameters
        ----------
        dcm
        measured_slice_width

        Returns
        -------
        normalised_snr: float

        """
        centre = self.ACR_obj.centre
        col, row = centre
        noise_img = self.get_noise_image(dcm)

        signal = [np.mean(roi) for roi in
                  self.get_roi_samples(ax=None, dcm=dcm, centre_col=int(col), centre_row=int(row))]

        noise = [np.std(roi, ddof=1) for roi in
                 self.get_roi_samples(ax=None, dcm=noise_img, centre_col=int(col), centre_row=int(row))]
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
            axes[0].scatter(centre[0], centre[1], c='red')
            axes[0].set_title('Centroid Location')

            axes[1].set_title('Smoothed Noise Image')
            axes[1].imshow(noise_img, cmap='gray')
            self.get_roi_samples(axes[1], dcm, int(col), int(row))

            img_path = os.path.realpath(os.path.join(self.report_path, f'{self.img_desc(dcm)}_smoothing.png'))
            fig.savefig(img_path)
            self.report_files.append(img_path)

        return snr, normalised_snr

    def snr_by_subtraction(self, dcm1: pydicom.Dataset, dcm2: pydicom.Dataset, measured_slice_width=None) -> float:
        """

        Parameters
        ----------
        dcm1
        dcm2
        measured_slice_width

        Returns
        -------

        """
        centre = self.ACR_obj.centre
        col, row = centre

        difference = np.subtract(dcm1.pixel_array.astype('int'), dcm2.pixel_array.astype('int'))

        signal = [np.mean(roi) for roi in
                  self.get_roi_samples(ax=None, dcm=dcm1, centre_col=int(col), centre_row=int(row))]
        noise = np.divide(
            [np.std(roi, ddof=1) for roi in
             self.get_roi_samples(ax=None, dcm=difference, centre_col=int(col), centre_row=int(row))],
            np.sqrt(2))
        snr = np.mean(np.divide(signal, noise))

        normalised_snr = snr * self.get_normalised_snr_factor(dcm1, measured_slice_width)

        if self.report:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(2, 1)
            fig.set_size_inches(8, 16)
            fig.tight_layout(pad=4)

            axes[0].imshow(dcm1.pixel_array)
            axes[0].scatter(centre[0], centre[1], c='red')
            axes[0].axis('off')
            axes[0].set_title('Centroid Location')

            axes[1].set_title('Difference Image')
            axes[1].imshow(difference, cmap='gray',)
            self.get_roi_samples(axes[1], dcm1, int(col), int(row))
            axes[1].axis('off')

            img_path = os.path.realpath(os.path.join(self.report_path, f'{self.img_desc(dcm1)}_snr_subtraction.png'))
            fig.savefig(img_path)
            self.report_files.append(img_path)

        return snr, normalised_snr
