"""
ACR SNR

Calculates the SNR for a single-slice image of a uniform MRI phantom

This script utilises the smoothed subtraction method described in McCann 2013:
A quick and robust method for measurement of signal-to-noise ratio in MRI, Phys. Med. Biol. 58 (2013) 3775:3790


Created by Neil Heraghty (Adapted by Yassine Azma)

09/01/2023
"""

import sys
import traceback
import os
import hazenlib
from hazenlib.HazenTask import HazenTask
from scipy import ndimage
import numpy as np
import skimage.morphology
import pydicom

class ACRSNR(HazenTask):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, measured_slice_width=None) -> dict:
        snr_results = {}

        z = []
        for dcm in self.data:
            z.append(dcm.ImagePositionPatient[2])

        idx_sort = np.argsort(z)

        # SINGLE METHOD (SMOOTHING)
        for dcm in self.data:
            if dcm.ImagePositionPatient[2] == z[idx_sort[6]]:
                try:
                    snr, normalised_snr = self.snr_by_smoothing(dcm, measured_slice_width)
                    snr_results[f"snr_smoothing_measured_{self.key(dcm)}"] = round(snr, 2)
                    snr_results[f"snr_smoothing_normalised_{self.key(dcm)}"] = round(normalised_snr, 2)
                except Exception as e:
                    print(f"Could not calculate the SNR for {self.key(dcm)} because of : {e}")
                    traceback.print_exc(file=sys.stdout)
                    continue


        results = {self.key(self.data[0]): snr_results, 'reports': {'images': self.report_files}}

        return results

    def centroid(self, dcm):
        img = dcm.pixel_array
        mask = img > 0.25 * np.max(img)
        open_img = skimage.morphology.area_opening(mask, area_threshold=500)
        mask = skimage.morphology.convex_hull_image(open_img)

        coords = np.nonzero(mask)  # row major - first array is columns

        sum_x = np.sum(coords[1])
        sum_y = np.sum(coords[0])
        cxy = sum_x / coords[0].shape, sum_y / coords[1].shape

        cxy = [cxy[0].astype(int), cxy[1].astype(int)]

        return mask, cxy

    def get_normalised_snr_factor(self, dcm, measured_slice_width=None) -> float:
        dx, dy = hazenlib.get_pixel_size(dcm)
        bandwidth = hazenlib.get_bandwidth(dcm)
        TR = hazenlib.get_TR(dcm)
        rows = hazenlib.get_rows(dcm)
        columns = hazenlib.get_columns(dcm)

        if measured_slice_width:
            slice_thickness = measured_slice_width
        else:
            slice_thickness = hazenlib.get_slice_thickness(dcm)

        averages = hazenlib.get_average(dcm)
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

        # filter size = 9, following MATLAB code and McCann 2013 paper for head coil, although note McCann 2013 recommends 25x25 for body coil.
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
        report_path

        Returns
        -------
        normalised_snr: float

        """
        _, centre = self.centroid(dcm)
        col, row = centre
        noise_img = self.get_noise_image(dcm)

        signal = [np.mean(roi) for roi in self.get_roi_samples(ax=None, dcm=dcm, centre_col=int(col), centre_row=int(row))]

        noise = [np.std(roi, ddof=1) for roi in self.get_roi_samples(ax=None, dcm=noise_img, centre_col=int(col), centre_row=int(row))]
        # note no root_2 factor in noise for smoothed subtraction (one image) method, replicating Matlab approach and McCann 2013

        snr = np.mean(np.divide(signal, noise))

        normalised_snr = snr * self.get_normalised_snr_factor(dcm, measured_slice_width)

        if self.report:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 1)
            fig.set_size_inches(5, 5)
            fig.tight_layout(pad=1)

            axes.set_title('smoothed noise image')
            axes.imshow(noise_img, cmap='gray', label='smoothed noise image')
            axes.scatter(col, row, 10, marker="+", label='centre')
            self.get_roi_samples(axes, dcm, int(col), int(row))
            axes.legend()

            img_path = os.path.realpath(os.path.join(self.report_path, f'{self.key(dcm)}_smoothing.png'))
            fig.savefig(img_path)
            self.report_files.append(img_path)

        return snr, normalised_snr
