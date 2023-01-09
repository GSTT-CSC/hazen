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
import pathlib

import pydicom
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

import skimage.morphology
from scipy import ndimage
from skimage import filters

from hazenlib.logger import logger


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


def smooth(dcm, kernel=skimage.morphology.square(9)):
    """
    Create noise and smoothed images from original.

    Parameters
    ----------
    dcm : `pydicom.dataset.Dataset` object containing one SNR image of flood
        phantom.

    kernel : array
        Kernel used for smoothing. Default is 9x9 boxcar.

    Returns
    -------
    original_image, smooth_image, noise_image
    """
    original_image = dcm.pixel_array.astype(float)
    kernel = kernel / kernel.sum()  # normalise kernel
    smooth_image = ndimage.filters.convolve(original_image, kernel)

    #  Alternative method 1: OpenCV.
    # smooth_image = cv2.blur(original_image, (kernel_len, kernel_len))

    #  Alternative method 2: scipy.ndimage.
    # kernel = np.ones([kernel_len, kernel_len], float)
    # kernel = kernel / kernel.sum() # normalise kernel
    # smooth_image = ndimage.filters.convolve(original_image, kernel)
    #  Note: filters.convolve and filters.correlate produce identical output
    #  for symetric kernels. Be careful with other kernels.

    noise_image = original_image - smooth_image
    return original_image, smooth_image, noise_image


def get_rois(smooth_image, roi_distance, roi_size):
    """
    Identify phantom and generate ROI locations.

    Parameters
    ----------
    smooth_image : array
        Smooted image.

    roi_distance : int
        Distance from centre of image to centre of each ROI along both
        dimensions.

    roi_size : int
        length of rectangular ROI in pixels.

    Returns
    -------
        mask : logical array
            Overlay to identify phantom location on map.

        roi_corners : list of coordinates of corners of ROIs used for sampling
            regions for SNR calculation.

        image_centre : array
            Coordinates of centre of mass of phantom.
    """

    # Threshold from smooth_image to reduce noise effects
    threshold = filters.threshold_minimum(smooth_image)
    mask = smooth_image > threshold

    #  Get centroid (=centre of mass for binary image) and convert to array
    image_centre = np.array(ndimage.measurements.center_of_mass(mask))
    logger.debug('image_centre = %r.', image_centre)

    #  Store corner of centre ROI, cast as int for indexing
    roi_corners = [np.rint(image_centre - roi_size / 2).astype(int)]

    #  Add corners of remaining ROIs
    roi_corners.append(roi_corners[0] + [-roi_distance, -roi_distance])
    roi_corners.append(roi_corners[0] + [roi_distance, -roi_distance])
    roi_corners.append(roi_corners[0] + [-roi_distance, roi_distance])
    roi_corners.append(roi_corners[0] + [roi_distance, roi_distance])

    return mask, roi_corners, image_centre


def calc_snr(original_image, noise_image, roi_corners, roi_size):
    """
    Calculate SNR

    Parameters
    ----------
    original_image : array

    noise_image : array

    roi_corners : list of 2-element arrays
        Coordinates of corners of ROIs.
        E.g. [array([114, 121]), array([74, 81]), array([154,  81]),
        array([ 74, 161]), array([154, 161])]

    roi_size : int
        Length of ROI square in pixels

    Returns
    -------
    snr : float
        Average SNR measured over ROIs

    """
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

    logger.debug('ROIs signal=%r, noise=%r, snr=%r',
                 roi_signal, roi_noise, roi_snr)

    return snr


def calc_snr_map(original_image, noise_image, roi_size):
    """
    Calculate SNR map.

    Parameters
    ----------
    original_image : array

    noise_image : array

    roi_size : int
        Length of ROI square in pixels.

    Returns
    -------
    snr_map : array
        Map of local SNR values
    """
    #  If you need a faster (less transparent) implementation, see:
    #  https://nickc1.github.io/python,/matlab/2016/05/17/Standard-Deviation-(Filters)-in-Matlab-and-Python.html

    noise_map = ndimage.filters.generic_filter(noise_image, sample_std,
                                               size=roi_size)
    signal_map = ndimage.filters.uniform_filter(original_image, size=roi_size)
    snr_map = signal_map / noise_map

    return snr_map


def plot_snr_map(snr_map, fig, ax):
    """
    Add SNR map to a figure axis.

    Parameters
    ----------
    snr_map : array

    fig : figure handle

    ax : axes handle within figure

    Returns
    -------
    None
    """
    para_im = ax.imshow(snr_map, cmap='viridis', vmin=0)
    cax = fig.add_axes([ax.get_position().x1 + 0.01,
                        ax.get_position().y0, 0.02,
                        ax.get_position().height])
    plt.colorbar(para_im, cax=cax)
    ax.set_title('SNR map')


def draw_roi_rectangles(ax, roi_corners, roi_size):
    """
    Add ROI rectangle overlays to plot.

    Parameters
    ----------
    ax : matplotlib.axes
        Add the ROIs to the axes.

    roi_corners : list of 2-element arrays
        Coordinates of corners of ROIs.
        E.g. [array([114, 121]), array([74, 81]), array([154,  81]),
        array([ 74, 161]), array([154, 161])]

    roi_size : int
        Length of ROI rectangle.

    Returns
    -------
    None

    """
    for corner in roi_corners:
        rect = patches.Rectangle(np.flip(corner), roi_size, roi_size,
                                 linewidth=1, edgecolor='r',
                                 facecolor='none')
        ax.add_patch(rect)


def plot_detailed(dcm, original_image, smooth_image, noise_image, snr_map,
                  phantom_mask, image_centre, roi_corners, roi_size, snr):
    """

    Parameters
    ----------
    dcm : pydicom object
        Used to get original file name

    original_image : array

    smooth_image : array

    noise_image : array

    snr_map : array

    phantom_mask : bool array

    image_centre : array
            Coordinates of centre of mass of phantom

    roi_corners : list of coordinates of corners of ROIs used for sampling
            regions for SNR calculation

    roi_size : int
        Length of ROI square in pixels

    snr : float
        Average SNR measured over ROIs (for plot title)

    Returns
    -------
    fig : matplotlib.figure.Figure
        Handle to plot
    """
    fig, axs = plt.subplots(1, 4, sharex=True, sharey=True,
                            figsize=(8, 2.8))
    fig.suptitle('SNR = %.2f (file: %s)'
                 % (snr, os.path.basename(dcm.filename)))
    axs[0].imshow(original_image, cmap='gray')
    axs[0].set_title('Magnitude Image')
    axs[1].imshow(smooth_image, cmap='gray')
    axs[1].contour(phantom_mask, colors='y')
    phantom_centre_marker = patches.Circle(
        np.flip(np.rint(image_centre).astype('int')), color='y')
    axs[1].add_patch(phantom_centre_marker)
    axs[1].set_title('Smoothed')
    axs[2].imshow(noise_image, cmap='gray')
    axs[2].set_title('Noise')
    draw_roi_rectangles(axs[0], roi_corners, roi_size)
    draw_roi_rectangles(axs[2], roi_corners, roi_size)
    plot_snr_map(snr_map, fig, axs[3])
    for ax in axs:
        ax.axis('off')

    return fig


def plot_summary(original_image, snr_map, roi_corners, roi_size):
    """

    Parameters
    ----------
    original_image : array

    snr_map : array

    roi_corners : list of coordinates of corners of ROIs used for sampling
            regions for SNR calculation

    roi_size : int
        Length of ROI square in pixels


    Returns
    -------
    fig : matplotlib.figure.Figure
        Handle to plot

    """
    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True,
                            figsize=(6, 2.8))
    axs[0].imshow(original_image, cmap='gray')
    axs[0].set_title('Magnitude Image')

    draw_roi_rectangles(axs[0], roi_corners, roi_size)
    plot_snr_map(snr_map, fig, axs[1])
    for ax in axs:
        ax.axis('off')

    return fig


def main(dcm_list, kernel_len=9, roi_size=20, roi_distance=40,
         report_path=False, report_dir=pathlib.Path.joinpath(pathlib.Path.cwd(), 'report', 'SNRMap')):
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
    report_path:
    report_dir:

    Returns
    -------
    results : dict
    """
    # ----
    # * Scale ROI distance to account for different image sizes.
    # * Pass kernel_len and roi_size parameters from command line.

    results = {}
    if report_path:
        # Create nested report folder and ignore if already exists
        pathlib.Path.mkdir(report_dir, parents=True, exist_ok=True)

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
        original_image, smooth_image, noise_image = \
            smooth(dcm, skimage.morphology.square(kernel_len))

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
            logger.warning('Expected image size (256, 256). Image size is %r.'
                           ' Algorithm untested with these dimensions.',
                           original_image.shape)

        #  Calculate mask and ROIs
        #  =======================
        phantom_mask, roi_corners, image_centre = \
            get_rois(smooth_image, roi_distance, roi_size)

        #  Calculate SNR
        #  =============
        snr = calc_snr(original_image, noise_image, roi_corners, roi_size)

        #  Generate local SNR parametric map
        #  =================================
        snr_map = calc_snr_map(original_image, noise_image, roi_size)

        #  Plot images
        #  ===========
        fig_detailed = plot_detailed(dcm, original_image, smooth_image,
                                     noise_image, snr_map, phantom_mask,
                                     image_centre, roi_corners, roi_size, snr)
        fig_summary = plot_summary(original_image, snr_map, roi_corners, roi_size)

        #  Save images
        #  ===========
        if report_path:
            detailed_image_path = pathlib.Path.joinpath(report_dir, f'{report_path}_snr_map_detailed.png')
            summary_image_path = pathlib.Path.joinpath(report_dir, f'{report_path}_snr_map.png')

            fig_detailed.savefig(detailed_image_path, dpi=300)
            fig_summary.savefig(summary_image_path, dpi=300)

            results[f'snr_map_detailed_{key}'] = detailed_image_path
            results[f'snr_map_{key}'] = summary_image_path

        results[f'snr_map_snr_{key}'] = snr

    return results
