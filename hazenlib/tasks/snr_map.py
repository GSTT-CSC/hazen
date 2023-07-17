"""
Map of local SNR across a flood phantom slice.

Introduction
============

The SNR for each voxel in an image (of a flood phantom) is estimated as the SNR
of a ROI centred on that voxel following the single image SNR method of McCann
et al. [1]_. The SNR map can show variation in SNR caused by smoothing filters.
It also highlights small regions of low signal which could be caused by micro-
bubbles or foreign bodies in the phantom. These inhomogeneities can erroneously
reduce SNR measurements made by other methods.


Algorithm overview
==================
1. Apply boxcar smoothing to original image to create smooth image.
2. Create noise image by subtracting smooth image from original image.
3. Create image mask to remove background using e.g.
     skimage.filters.threshold_minimum
4. Calculate SNR using McCann's method and overlay ROIs on image.
5. Estimate local noise as standard deviation of pixel values in ROI centred on
    a pixel. Repeat for each pixel in the noise image.
6. Plot the local noise as a heat map.


.. [1] McCann, A. J., Workman, A., & McGrath, C. (2013). A quick and robust
method for measurement of signal-to-noise ratio in MRI. Physics in Medicine
& Biology, 58(11), 3775.
"""
from hazenlib.HazenTask import HazenTask

import pathlib

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

import skimage.morphology
from scipy import ndimage
from skimage import filters

from hazenlib.logger import logger


class SNRMap(HazenTask):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def smooth(self, kernel=skimage.morphology.square(9)):
        """
        Create noise and smoothed images from original_image.

        Parameters
        ----------
        kernel : array
            Kernel used for smoothing. Default is 9x9 boxcar.

        Returns
        -------
        None
        """
        self.original_image = self.current_dcm.pixel_array.astype(float)
        kernel = kernel / kernel.sum()  # normalise kernel
        self.smooth_image = ndimage.filters.convolve(self.original_image, kernel)

        #  Alternative method 1: OpenCV.
        # smooth_image = cv2.blur(original_image, (kernel_len, kernel_len))

        #  Alternative method 2: scipy.ndimage.
        # kernel = np.ones([kernel_len, kernel_len], float)
        # kernel = kernel / kernel.sum() # normalise kernel
        # smooth_image = ndimage.filters.convolve(original_image, kernel)
        #  Note: filters.convolve and filters.correlate produce identical output
        #  for symetric kernels. Be careful with other kernels.

        self.noise_image = self.original_image - self.smooth_image

    def draw_roi_rectangles(self, ax):
        """
        Add ROI rectangle overlays to plot.

        Parameters
        ----------
        ax : matplotlib.axes
            Add the ROIs to the axes.


        Returns
        -------
        None

        """
        for corner in self.roi_corners:
            rect = patches.Rectangle(np.flip(corner), self.roi_size,
                                     self.roi_size, linewidth=1, edgecolor='r',
                                     facecolor='none')
            ax.add_patch(rect)

    def calc_snr_map(self):
        """
        Calculate SNR map from original_image and noise_image.
        """
        #  If you need a faster (less transparent) implementation, see:
        #  https://nickc1.github.io/python,/matlab/2016/05/17/Standard-Deviation-(Filters)-in-Matlab-and-Python.html

        noise_map = ndimage.filters.generic_filter(self.noise_image,
                                                   lambda x: np.std(x, ddof=1),
                                                   size=self.roi_size)
        signal_map = ndimage.filters.uniform_filter(self.original_image,
                                                    size=self.roi_size)
        self.snr_map = signal_map / noise_map

    def calc_snr(self):
        """
        Calculate SNR from original_image and noise_image.
        """
        roi_signal = []
        roi_noise = []

        for [x, y] in self.roi_corners:
            roi_signal.append(self.original_image[x:x + self.roi_size, y:y + self.roi_size].mean())
            roi_noise.append(self.noise_image[x:x + self.roi_size, y:y + self.roi_size].std(ddof=1))
            # Note: *.std(ddof=1) uses sample standard deviation, default ddof=0
            # uses population std dev. Not sure which is statistically correct,
            # but using ddof=1 for consistency with IDL code.

        roi_snr = np.array(roi_signal) / np.array(roi_noise)
        self.snr = roi_snr.mean()

        logger.debug('ROIs signal=%r, noise=%r, snr=%r',
                     roi_signal, roi_noise, roi_snr)

    def plot_snr_map(self, fig, ax):
        """
        Add SNR map to a figure axis.

        Parameters
        ----------
        fig : figure handle

        ax : axes handle within figure

        Returns
        -------
        None
        """
        para_im = ax.imshow(self.snr_map, cmap='viridis', vmin=0)
        cax = fig.add_axes([ax.get_position().x1 + 0.01,
                            ax.get_position().y0, 0.02,
                            ax.get_position().height])
        plt.colorbar(para_im, cax=cax)
        ax.set_title('SNR map')

    def plot_detailed(self):
        """
        Create 4-image detailed SNR map plots

        Returns
        -------
        fig : matplotlib.figure.Figure
            Handle to plot
        """
        fig, axs = plt.subplots(1, 4, sharex=True, sharey=True,
                                figsize=(8, 2.8))
        fig.suptitle('SNR = %.2f (file: %s)'
                     % (self.snr, os.path.basename(self.current_dcm.filename)))
        axs[0].imshow(self.original_image, cmap='gray')
        axs[0].set_title('Magnitude Image')
        axs[1].imshow(self.smooth_image, cmap='gray')
        axs[1].contour(self.mask, colors='y')
        phantom_centre_marker = patches.Circle(
            np.flip(np.rint(self.image_centre).astype('int')), color='y')
        axs[1].add_patch(phantom_centre_marker)
        axs[1].set_title('Smoothed')
        axs[2].imshow(self.noise_image, cmap='gray')
        axs[2].set_title('Noise')
        self.draw_roi_rectangles(axs[0])
        self.draw_roi_rectangles(axs[2])
        self.plot_snr_map(fig, axs[3])
        for ax in axs:
            ax.axis('off')

        return fig

    def plot_summary(self):
        """
        Create 2-image summary SNR map plot.

        Parameters
        ----------


        Returns
        -------
        fig : matplotlib.figure.Figure
            Handle to plot

        """
        fig, axs = plt.subplots(1, 2, sharex=True, sharey=True,
                                figsize=(6, 2.8))
        axs[0].imshow(self.original_image, cmap='gray')
        axs[0].set_title('Magnitude Image')

        self.draw_roi_rectangles(axs[0])
        self.plot_snr_map(fig, axs[1])
        for ax in axs:
            ax.axis('off')

        return fig


    def get_rois(self):
        """
        Identify phantom and generate ROI locations.
        """

        # Threshold from smooth_image to reduce noise effects
        threshold = filters.threshold_minimum(self.smooth_image)
        self.mask = self.smooth_image > threshold

        #  Get centroid (=centre of mass for binary image) and convert to array
        self.image_centre = \
            np.array(ndimage.measurements.center_of_mass(self.mask))
        logger.debug('image_centre = %r.', self.image_centre)

        #  Store corner of centre ROI, cast as int for indexing
        roi_corners = [np.rint(self.image_centre - self.roi_size / 2).astype(int)]

        #  Add corners of remaining ROIs
        roi_distance = self.roi_distance
        roi_corners.append(roi_corners[0] + [-roi_distance, -roi_distance])
        roi_corners.append(roi_corners[0] + [roi_distance, -roi_distance])
        roi_corners.append(roi_corners[0] + [-roi_distance, roi_distance])
        roi_corners.append(roi_corners[0] + [roi_distance, roi_distance])

        self.roi_corners = roi_corners


    def run(self):
        """
        Returns SNR parametric map on flood phantom DICOM file.

        Five square ROIs are created, one at the image centre, and four peripheral
        ROIs with their centres displaced at 45, 135, 225 and 315 degrees from the
        centre. Displays and saves a parametric map.

        Returns
        -------
        results : dict
        """
        # Initialise variables
        self.kernel_len = 9
        self.roi_size = 20
        self.roi_distance = 40

        # ----
        # * Scale ROI distance to account for different image sizes.
        # * Pass kernel_len and roi_size parameters from command line.

        results = {}
        if self.report:
            # Create nested report folder and ignore if already exists
            pathlib.Path.mkdir(pathlib.Path(self.report_path), parents=True, exist_ok=True)

        for self.current_dcm in self.data:

            key = self.key(self.current_dcm)

            #  Create original, smoothed and noise images
            #  ==========================================
            self.smooth(skimage.morphology.square(self.kernel_len))

            #  Note: access NumPy arrays by column then row. E.g.
            #
            #  t=np.array([[1,2,3],[4,5,6]])
            #  t
            #  Out[118]:
            #  array([[1, 2, 3],
            #         [4, 5, 6]])
            #
            #  t[1,0]
            #  Out[119]: 4 # not 2
            #
            #  Confusingly, patches (circles, rectangles) use traditional [x,y]
            #  positioning. To centre a circle on pixel [a,b], the circle must be
            #  centred on [b,a]. The function np.flip(coords) can help.

            #  Warn if not 256 x 256 image
            #  TODO scale distances for other image sizes
            if self.original_image.shape != (256, 256):
                logger.warning('Expected image size (256, 256). Image size is %r.'
                               ' Algorithm untested with these dimensions.',
                               self.original_image.shape)

            #  Calculate mask and ROIs
            #  =======================
            self.get_rois()

            #  Calculate SNR
            #  =============
            self.calc_snr()

            #  Generate local SNR parametric map
            #  =================================
            self.calc_snr_map()

            #  Plot images
            #  ===========
            fig_detailed = self.plot_detailed()
            fig_summary = self.plot_summary()

            #  Save images
            #  ===========
            if self.report:
                detailed_image_path = os.path.join(self.report_path, f'{key}_snr_map_detailed.png')
                summary_image_path = os.path.join(self.report_path, f'{key}_snr_map.png')

                fig_detailed.savefig(detailed_image_path, dpi=300)
                fig_summary.savefig(summary_image_path, dpi=300)

                self.report_files.append(summary_image_path)
                self.report_files.append(detailed_image_path)

                results['reports'] = {'images': self.report_files}

            results[key] = self.snr

        return results




