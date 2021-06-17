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

import logging
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

from scipy import ndimage
from skimage import filters

# Set up logger
log = logging.getLogger(__name__)
log.setLevel('WARNING')
# Need to configure STDERR logger level if not already done
if len(log.handlers) == 0:
    std_err_handle = logging.StreamHandler()
    std_err_format = logging.Formatter('%(name)s : %(levelname)s : %(message)s')
    std_err_handle.setFormatter(std_err_format)
    log.addHandler(std_err_handle)


# helper functions
def sample_std(vals):
    """
    Return sample stdev using numpy.stdev(..., ddof=1).

    Parameters
    ----------
    vals : array-like
        Array-like object to calculate sample stdev.

    Returns
    -------
    standard deviation : ndarray, same dtype as vals.
    """
    return np.std(vals, ddof=1)


def main(dcm_list, kernel_len=9, roi_size=20, roi_distance=40,
         report_path=False):
    """
    Returns SNR parametric map on flood phantom DICOM file.

    Five square ROIs are created, one at the image centre, and four peripheral
    ROIs with their centres displaced at 45, 135, 225 and 315 degrees from the
    centre. Displays and saves a parametric map.

    Parameters
    ----------
    dcm_list : list of `pydicom.dataset.Dataset` objects
        Images from which to calculate single image SNR map.
    kernel_len : int, optional
        Linear extent of the boxcar kernel, should be an odd integer. The
        default is 9.
    roi_size : int, optional
        Length of side (in pixels) of each ROI. The default is 20.
    roi_distance : int, optional
        Distance from centre of image to centre of each ROI along both
        dimensions. The default is 40.

    Returns
    -------
    results : list

    TODO
    ----
    * Scale ROI distance to account for different image sizes.
    * Pass kernel_len and roi_size parameters from command line.
    * Allow different kernels, e.g. 'disk' instead of boxcar.
    """

    results = {}

    for dcm in dcm_list:

        #  Check input is pydicom object
        if not isinstance(dcm, pydicom.dataset.Dataset):
            raise Exception('Input must be pydicom dataset (or None)')

        try:
            key = f"{dcm.SeriesDescription}_{dcm.SeriesNumber}_" \
            f"{dcm.InstanceNumber}_{os.path.basename(dcm.filename)}"
        except AttributeError as e:
            print(e)
            key = f"{dcm.SeriesDescription}_{dcm.SeriesNumber}_" \
            f"{os.path.basename(dcm.filename)}"

        if report_path:
            report_path = key

        #  Create original, smoothed and noise images
        #  ==========================================

        #  cast as float to allow non-integer values for smoothing

        original_image = dcm.pixel_array.astype(float)

        smooth_image = ndimage.filters.uniform_filter(original_image,
                                                      size=kernel_len)

        #  Alternative method 1:- can use different kernels
        #  Boxcar filter
        #  NB morphology.disk may be better
        # smoothing_kernel = morphology.square(kernel_len) # Boxcar kernel
        # smoothing_kernel = smoothing_kernel / smoothing_kernel.sum() # normalise
        # smooth_image = ndimage.filters.convolve(original_image, smoothing_kernel)

        #  Alternative method 2: OpenCV.
        # smooth_image = cv2.blur(original_image, (kernel_len, kernel_len))

        #  Alternative method 3: scipy.ndimage.
        # kernel = np.ones([kernel_len, kernel_len], float)
        # kernel = kernel / kernel.sum() # normalise kernel
        # smooth_image = ndimage.filters.convolve(original_image, kernel)
        #  Note: filters.convolve and filters.correlate produce identical output
        #  for symetric kernels. Be careful with other kernels.

        noise_image = original_image - smooth_image

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
        if original_image.shape != (256, 256):
            log.warning('Expected image size (256, 256). Image size is %r.'
                        ' Algorithm untested with these dimensions.',
                        original_image.shape)

        #  Calculate ROIs
        #  ==============

        # Threshold from smooth_image to reduce noise effects
        threshold = filters.threshold_minimum(smooth_image)
        phantom_mask = smooth_image > threshold

        #  Get centroid (=centre of mass for binary image) and convert to array
        image_centre = np.array(ndimage.measurements.center_of_mass(phantom_mask))
        log.debug('image_centre = %r.', image_centre)

        #  Store corner of centre ROI, cast as int for indexing
        roi_corners = [np.rint(image_centre - roi_size / 2).astype(int)]

        #  Add corners of remaining ROIs
        roi_corners.append(roi_corners[0] + [-roi_distance, -roi_distance])
        roi_corners.append(roi_corners[0] + [roi_distance, -roi_distance])
        roi_corners.append(roi_corners[0] + [-roi_distance, roi_distance])
        roi_corners.append(roi_corners[0] + [roi_distance, roi_distance])

        #  Calculate SNR
        #  =============

        roi_signal = []
        roi_noise = []

        for [x, y] in roi_corners:
            roi_signal.append(original_image[x:x + roi_size, y:y + roi_size].mean())
            roi_noise.append(noise_image[x:x + roi_size, y:y + roi_size].std(ddof=1))
            # Note: *.std(ddof=1) uses sample standard deviation, default ddof=0
            # uses population std dev. Not sure which is statistically correct,
            # but using ddof=1 for consistency with IDL code.

        roi_snr = np.array(roi_signal) / np.array(roi_noise)
        snr = roi_snr.mean()

        log.debug('ROIs signal=%r, noise=%r, snr=%r',
                  roi_signal, roi_noise, roi_snr)

        #  Plot images
        fig_detailed, axs = plt.subplots(1, 4, sharex=True, sharey=True,
                                         figsize=(8, 2.8))
        fig_summary, axs2 = plt.subplots(1, 2, sharex=True, sharey=True,
                                         figsize=(6, 2.8))

        fig_detailed.suptitle('SNR = %.2f (file: %s)'
                     % (snr, os.path.basename(dcm.filename)))
        axs[0].imshow(original_image, cmap='gray')
        axs[0].set_title('Signal')
        axs[1].imshow(smooth_image, cmap='gray')
        axs[1].contour(phantom_mask, colors='y')
        phantom_centre_marker = patches.Circle(
            np.flip(np.rint(image_centre).astype('int')), color='y')
        #  See comment on np.flip(...) above.
        axs[1].add_patch(phantom_centre_marker)
        axs[1].set_title('Smoothed')
        axs[2].imshow(noise_image, cmap='gray')
        axs[2].set_title('Noise')

        axs2[0].imshow(original_image, cmap='gray')
        axs2[0].set_title('Signal')

        #  Add ROI rectangles
        for ax in axs[[0, 2]]:  # plot on signal and noise images, not smoothed
            for corner in roi_corners:
                rect = patches.Rectangle(np.flip(corner), roi_size, roi_size,
                                         linewidth=1, edgecolor='r',
                                         facecolor='none')
                # See comment on np.flip(...) above.
                ax.add_patch(rect)

        for corner in roi_corners:
            rect = patches.Rectangle(np.flip(corner), roi_size, roi_size,
                                     linewidth=1, edgecolor='r',
                                     facecolor='none')
            # See comment on np.flip(...) above.
            axs2[0].add_patch(rect)

        #  Generate local SNR parametric map
        #  =================================

        #  If you need a faster (less transparent) implementation, see:
        #  https://nickc1.github.io/python,/matlab/2016/05/17/Standard-Deviation-(Filters)-in-Matlab-and-Python.html

        noise_map = ndimage.filters.generic_filter(noise_image, sample_std,
                                                   size=roi_size)
        signal_map = ndimage.filters.uniform_filter(original_image, size=roi_size)
        snr_map = signal_map / noise_map
        para_im = axs[3].imshow(snr_map, cmap='viridis', vmin=0)
        cax = fig_detailed.add_axes([axs[3].get_position().x1 + 0.01,
                            axs[3].get_position().y0, 0.02,
                            axs[3].get_position().height])
        plt.colorbar(para_im, cax=cax)
        axs[3].set_title('SNR map')

        para_im2 = axs2[1].imshow(snr_map, cmap='viridis', vmin=0)
        cax2 = fig_summary.add_axes([axs2[1].get_position().x1 + 0.01,
                              axs2[1].get_position().y0, 0.02,
                              axs2[1].get_position().height])
        plt.colorbar(para_im, cax=cax2)
        axs2[1].set_title('SNR map')

        for ax in np.hstack((axs, axs2)):
            ax.axis('off')

        # save images
        if report_path:
            detailed_image_path = f'{report_path}_snr_map_detailed.png'
            summary_image_path = f'{report_path}_snr_map.png'

            fig_detailed.savefig(detailed_image_path, dpi=300)
            fig_summary.savefig(summary_image_path, dpi=300)

            results[f'snr_map_detailed_{key}'] = detailed_image_path
            results[f'snr_map_{key}'] = summary_image_path

        results[f'snr_map_snr_{key}'] = snr

    return results
