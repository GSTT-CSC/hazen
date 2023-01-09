"""
Measure T1 and T2 in Caliber relaxometry phantom.

Introduction
============

This module determines the T1 and T2 decay constants for the relaxometry
spheres in the Caliber (HPD) system phantom
qmri.com/qmri-solutions/t1-t2-pd-imaging-phantom (plates 4 and 5). Values are
compared to published values (without temperature correction). Graphs of fit
and phantom registration images can optionally be produced.


Scan parameters
===============

Manufacturer's details of recommended scan parameters for GE, Philips and
Siemens scanners are available in the 'System Phantom Manual' which can be
downloaded from the above website (T1-VTI and T2 sequences). However, these may
result in long scan times. The parameters below were used to acquire the
images used in testing this module. They are provided for information only.


T1 Relaxometry
--------------

Sequence: Spin echo with inversion recovery
Plane: Coronal
TR (ms): 1000 (or minimum achievable if longer--see note)
TE (ms): 10
TI (ms): {50.0, 100.0, 200.0, 400.0, 600.0, 800.0}
Flip angle: 180 degrees
Matrix: 192 x 192
FoV (mm): 250 x 250
Slices: 2 (or 3 to acquire plate 3 with PD spheres)
Slice width (mm): 5
Distance factor: 35 mm / 700%
NSA: 1
Receive bandwidth:
    GE (kHz): 15.63
    Philips (Hz / px): 109
    Siemens (Hz / px): 130
Reconstruction: Normalised

Note: Some scanners may require a longer TR for long TI values. This algorithm
will accommodate a variation in TR with TI and incomplete recovery due to short
TR.


T2 Relaxometry
--------------

Sequence:
    GE: T2 map (TE values fixed)
    Other manufacturers: Spin echo multi contrast
Plane: Coronal
TR (ms): 2000
Number of contrasts: maximum
TE (ms): minimum
Flip angle: 90 degrees
Matrix: 192 x 192
FoV (mm): 250 x 250
Slices: 2 (or 3 to acquire plate 3 with PD spheres)
Slice width (mm): 5
Distance factor: 35 mm / 700%
NSA: 1
Receive bandwidth:
    GE (kHz): 15.63
    Philips (Hz / px): 109
    Siemens (Hz / px): 130
Reconstruction: Normalised


Algorithm overview
==================
1. Create ``T1ImageStack`` or ``T2ImageStack`` object which stores a list
    of individual DICOM files (as ``pydicom`` objects) in the ``.images``
    attribute.
2. Obtain the RT (rotation / translation) matrix to register the template
    image to the test image. Four template images are provided, one for
    each relaxation parameter (T1 or T2) on plates 4 and 5, and regression
    is performed on the first image in the sequence. Optionally output the
    overlay image to visually check the fit.
3. An ROI is generated for each target sphere using stored coordinates, the
    RT transformation above, and a structuring element (default is a 5x5
    boxcar).
4. Store pixel data for each ROI, at various times, in an ``ROITimeSeries``
    object. A list of these objects is stored in 
    ``ImageStack.ROI_time_series``.
5. Generate the fit function. For T1 this looks up TR for the given TI 
    (using piecewise linear interpolation if required) and determines if a
    magnitude or signed image is used. No customisation is required for T2
    measurements.
6. Determine relaxation time (T1 or T2) by fitting the decay equation to
    the ROI data for each sphere. The published values of the relaxation
    times are used to seed the optimisation algorithm. For T2 fitting the
    input data are truncated for TE > 5*T2 to avoid fitting Rician noise in
    magnitude images with low signal intensity. Optionally plot and save the
    decay curves.
7. Return plate number, relaxation type (T1 or T2), measured relaxation
    times, published relaxation times, and fractional differences in a
    dictionary.


Feature enhancements
====================
Template fit on bolt holes--possibly better with large rotation angles
    -have bolthole template, find 3 positions in template and image, figure out
    transformation.

Template fit on outline image--poss run though edge detection algorithms then
fit.

Use normalised structuring element in ROITimeSeries. This will allow correct
calculation of mean if elements are not 0 or 1.

Get r-squared measure of fit.

"""
import pydicom
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import os.path
import skimage.morphology
import scipy.ndimage
import scipy.optimize
from scipy.interpolate import UnivariateSpline

import hazenlib.exceptions

# Use dict to store template and reference information
# Coordinates are in array format (row,col), rather than plt.patches 
# format (col,row)
#
# Access as:
#    TEMPLATE_VALUES[f'plate{plate_num}']['sphere_centres_row_col']
#    TEMPLATE_VALUES[f'plate{plate_num}']['t1'|'t2']['filename']
#    TEMPLATE_VALUES[f'plate{plate_num}']['t1'|'t2']['1.5T'|'3.0T']['relax_times']

TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            'data', 'relaxometry')
TEMPLATE_VALUES = {
    'plate3': {
        'sphere_centres_row_col': (),
        'bolt_centres_row_col': (),
        't1': {
            'filename': '',
            'relax_times': []}},

    'plate4': {
        'sphere_centres_row_col': (
            (56, 94), (62, 117), (81, 132), (105, 134), (125, 120), (133, 99),
            (127, 75), (108, 60), (84, 59), (64, 72), (80, 81), (78, 111),
            (109, 113), (111, 82), (148, 118)),
        'bolt_centres_row_col': (),
        't1': {
            'filename': os.path.join(TEMPLATE_DIR, 'Plate4_T1_signed'),
            'relax_times': {
                '1.5T':
                    np.array([2376, 2183, 1870, 1539, 1237, 1030, 752.2, 550.2,
                              413.4, 292.9, 194.9, 160.2, 106.4, 83.3, 2700]),
                '3.0T':
                    np.array([2480, 2173, 1907, 1604, 1332, 1044, 801.7, 608.6,
                              458.4, 336.5, 244.2, 176.6, 126.9, 90.9, 2700])}},
        't2': {
            'filename': os.path.join(TEMPLATE_DIR, 'Plate4_T2'),
            'relax_times': {
                '1.5T':
                    np.array([939.4, 594.3, 416.5, 267.0, 184.9, 140.6, 91.76,
                              64.84, 45.28, 30.62, 19.76, 15.99, 10.47, 8.15,
                              2400]),
                '3.0T':
                    np.array([581.3, 403.5, 278.1, 190.94, 133.27, 96.89,
                              64.07, 46.42, 31.97, 22.56, 15.813, 11.237,
                              7.911, 5.592, 2400])}}},

    'plate5': {
        'sphere_centres_row_col': (
            (56, 95), (62, 117), (81, 133), (104, 134), (124, 121), (133, 98),
            (127, 75), (109, 61), (84, 60), (64, 72), (80, 81), (78, 111),
            (109, 113), (110, 82), (97, 43)),
        'bolt_centres_row_col': ((52, 80), (92, 141), (138, 85)),
        't1': {
            'filename': os.path.join(TEMPLATE_DIR, 'Plate5_T1_signed'),
            'relax_times': {
                '1.5T':
                    np.array([2033, 1489, 1012, 730.8, 514.1, 367.9, 260.1,
                              184.6, 132.7, 92.7, 65.4, 46.32, 32.45, 22.859,
                              2700]),
                '3.0T':
                    np.array([1989, 1454, 984.1, 706, 496.7, 351.5, 247.13,
                              175.3, 125.9, 89.0, 62.7, 44.53, 30.84,
                              21.719, 2700])}},

        't2': {
            'filename': os.path.join(TEMPLATE_DIR, 'Plate5_T2'),
            'relax_times': {
                '1.5T':
                    np.array([1669.0, 1244.0, 859.3, 628.5, 446.3, 321.2,
                              227.7, 161.9, 117.1, 81.9, 57.7, 41.0, 28.7,
                              20.2, 2400]),
                '3.0T':
                    np.array([1465, 1076, 717.9, 510.1, 359.6, 255.5, 180.8,
                              127.3, 90.3, 64.3, 45.7, 31.86, 22.38,
                              15.83, 2400])}}}}


def outline_mask(im):
    """
    Create contour lines to outline pixels.
    
    Creates a series of ``line`` objects to outline contours on an image. Used
    to add ROIs from a mask array. Adapted from [1]_

    Parameters
    ----------
    im : array
        Pixel array used to create outlines. Array values should be 0 or 1.

    Returns
    -------
    lines : list
        List of coordinates of outlines (see Example below).
    
    Example
    -------
    >>> lines = outline_mask(combined_ROI_map)
    >>> for line in lines:
            plt.plot(line[1], line[0], color='r', alpha=1)
    
    References
    ----------
    .. [1] stackoverflow.com/questions/40892203/can-matplotlib-contours-match-pixel-edges

    """
    lines = []
    pad = np.pad(im, [(1, 1), (1, 1)])  # zero padding

    im0 = np.abs(np.diff(pad, n=1, axis=0))[:, 1:]
    im1 = np.abs(np.diff(pad, n=1, axis=1))[1:, :]

    im0 = np.diff(im0, n=1, axis=1)
    starts = np.argwhere(im0 == 1)
    ends = np.argwhere(im0 == -1)
    lines += [([s[0] - .5, s[0] - .5], [s[1] + .5, e[1] + .5]) for s, e
              in zip(starts, ends)]

    im1 = np.diff(im1, n=1, axis=0).T
    starts = np.argwhere(im1 == 1)
    ends = np.argwhere(im1 == -1)
    lines += [([s[1] + .5, e[1] + .5], [s[0] - .5, s[0] - .5]) for s, e
              in zip(starts, ends)]

    return lines


def transform_coords(coords, rt_matrix, input_row_col=True,
                     output_row_col=True):
    """
    Convert coordinates using RT transformation matrix.

    Note that arrays containing pixel information as displayed using
    plt.imshow(pixel_array), for example are referenced using the row_col (y,x)
    notation, e.g. pixel_array[row,col]. Plotting points or patches using
    matplotlib requires col_row (x,y) notation, e.g. plt.scatter(col,row). The
    correct input and output notation must be selected for the correct
    transformation.

    Parameters
    ----------
    coords : np.array or tuple
        Array (n,2) of coordinates to transform.
    rt_matrix : np.array
        Array (2,3) of transform matrix (Rotation and Translation). See e.g.
        cv2.transform() for details.
    input_row_col : bool, optional
        Select the input coordinate format relative to the image.
        If True, input array has row (y-coordinate) first, i.e.:
            [[row_1,col_1],
             [row_2,col_2],
             ...,
             [row_n,col_n]].
        If False, input array has col (x-coordinate) first, i.e.:
            [[col_1,row_1],
             [col_2,row_2],
             ...,
             [col_n,row_n].
        The default is True.
    output_row_col : bool, optional
        Select the output coordinate order. If True, output matrix is in
        row_col order, otherwise it is in col_row order. The default is True.

    Returns
    -------
    out_coords : np.array
        Returns (n,2) array of transformed coordinates.

    """
    in_coords = np.array(coords)  # ensure using np array

    if input_row_col:  # convert to col_row (xy) format
        in_coords = np.flip(in_coords, axis=1)

    out_coords = cv.transform(np.array([in_coords]), rt_matrix)
    out_coords = out_coords[0]  # reduce to two dimensions

    if output_row_col:
        out_coords = np.flip(out_coords, axis=1)

    return out_coords


def pixel_rescale(dcmfile):
    """
    Transforms pixel values according to scale values in DICOM header.
    
    DICOM pixel values arrays cannot directly represent signed or float values.
    This function converts the ``.pixel_array`` using the scaling values in the
    DICOM header.

    For Philips scanners the private DICOM fields 2005,100d (=SI) and 2005,100e
    (=SS) are used as inverse scaling factors to perform the inverse
    transformation [1]_

    Parameters
    ----------
    dcmfile : Pydicom.dataset.FileDataset
        DICOM file containing one image.

    Returns
    -------
    numpy.array
        Values in ``dcmfile.pixel_array`` transformed using DICOM scaling.

    References
    ----------
    .. [1] Chenevert, Thomas L., et al. "Errors in quantitative image analysis
     due to platform-dependent image scaling." Translational Oncology 7.1
     (2014): 65-71. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3998685/

    """
    # Check for Philips
    if dcmfile.Manufacturer.startswith('Philips'):
        ss = dcmfile['2005100e'].value  # Scale slope
        si = dcmfile['2005100d'].value  # Scale intercept

        return (dcmfile.pixel_array - si) / ss
    else:
        return pydicom.pixel_data_handlers.util.apply_modality_lut(
            dcmfile.pixel_array, dcmfile)


def generate_t1_function(ti_interp_vals, tr_interp_vals, mag_image=False):
    """
    Generate T1 signal function and jacobian with interpolated TRs.
    
    Signal intensity on T1 decay is a function of both TI and TR. Ideally, TR
    should be constant and at least 5*T1. However, scan time can be reduced by
    allowing a shorter TR which increases at long TIs. For example::
        TI |   50 |  100 |  200 |  400 |  600 |  800 
        ---+------+------+------+------+------+------
        TR | 1000 | 1000 | 1000 | 1260 | 1860 | 2460
    
    This function factory returns a function which calculates the signal
    magnitude using the expression::
        S = S0 * (1 - a1 * np.exp(-TI / t1) + np.exp(-TR / t1))
    where ``S0`` is the recovered intensity, ``a1`` is theoretically 2.0 but
    varies due to inhomogeneous B0 field, ``t1`` is the longitudinal
    relaxation time, and the repetition time, ``TR``, is calculated  from
    ``TI`` using piecewise linear interpolation.

    Parameters
    ----------
    ti_interp_vals : array_like
        Array of TI values used as a look-up table to calculate TR
    tr_interp_vals : array_like
        Array of TR values used as a lookup table to calculate TR from the TI
        used in the sequence.
    mag_image : bool, optional
        If True, the generated function returns the magnitude of the signal
        (i.e. negative outputs become positive). The default is False.

    Returns
    -------
    t1_function : function
        S = S0 * (1 - a1 * np.exp(-TI / t1) + np.exp(-TR / t1))
    
    t1_jacobian : function
        Tuple of partial derivatives for curve fitting.

    eqn_str : string
        String representation of fit function.
    """
    #  Create piecewise liner fit function. k=1 gives linear, s=0 ensures all
    #  points are on line. Using UnivariateSpline (rather than numpy.interp())
    #  enables derivative calculation if required.
    tr = UnivariateSpline(ti_interp_vals, tr_interp_vals, k=1, s=0)
    # tr_der = tr.derivative()

    eqn_str = 's0 * (1 - a1 * np.exp(-TI / t1) + np.exp(-TR / t1))'
    if mag_image:
        eqn_str = f'abs({eqn_str})'

    def _t1_function_signed(ti, t1, s0, a1):
        pv = s0 * (1 - a1 * np.exp(-ti / t1) + np.exp(-tr(ti) / t1))
        return pv

    def t1_function(ti, t1, s0, a1):
        pv = _t1_function_signed(ti, t1, s0, a1)
        if mag_image:
            return abs(pv)
        else:
            return pv

    def t1_jacobian(ti, t1, s0, a1):
        t1_der = s0 / (t1 ** 2) * (-ti * a1 * np.exp(-ti / t1) + tr(ti)
                                   * np.exp(-tr(ti) / t1))
        s0_der = 1 - a1 * np.exp(-ti / t1) + np.exp(-tr(ti) / t1)
        a1_der = -s0 * np.exp(-ti / t1)
        jacobian = np.array([t1_der, s0_der, a1_der])

        if mag_image:
            pv = _t1_function_signed(ti, t1, s0, a1)
            jacobian = (jacobian * (pv >= 0)) - (jacobian * (pv < 0))

        return jacobian.T

    return t1_function, t1_jacobian, eqn_str


def est_t1_s0(ti, tr, t1, pv):
    """
    Return initial guess of s0 to seed T1 curve fitting.

    Parameters
    ----------
    ti : array_like
        TI values.
    tr : array_like
        TR values.
    t1 : array_like
        Estimated T1 (typically from manufacturer's documentation).
    pv : array_like
        Mean pixel value (signal) in ROI.

    Returns
    -------
    array_like
        Initial s0 guess for calculating T1 relaxation time.

    """
    return -pv / (1 - 2 * np.exp(-ti / t1) + np.exp(-tr / t1))


def t2_function(te, t2, s0):
    """
    Calculated pixel value given TE, T2, S0 and C.
    
    Calculates pixel value from::
        S = S0 * np.exp(-te / t2)

    Parameters
    ----------
    te : array_like
        Echo times.
    t2 : float
        T2 decay constant.
    S0 : float
        Initial pixel magnitude.

    Returns
    -------
    pv : array_like
        Theoretical pixel values (signal) at each TE.

    """
    pv = s0 * np.exp(-te / t2)
    return pv


def t2_jacobian(te, t2, s0):
    """
    Jacobian of ``t2_function`` used for curve fitting.

    Parameters
    ----------
    te : array_like
        Echo times.
    t2 : float
        T2 decay constant.
    s0 : float
        Initial signal magnitude.

    Returns
    -------
    array
        [t2_der, s0_der].T, where x_der is a 1-D array of the partial
        derivatives at each ``te``.

    """
    t2_der = s0 * te / t2 ** 2 * np.exp(-te / t2)
    s0_der = np.exp(-te / t2)
    jacobian = np.array([t2_der, s0_der])
    return jacobian.T


def est_t2_s0(te, t2, pv, c=0.0):
    """
    Initial guess for s0 to seed curve fitting.

    Parameters
    ----------
    te : array_like
        Echo time(s).
    t2 : array_like
        T2 decay constant.
    pv : array_like
        Mean pixel value (signal) in ROI with ``te`` echo time.
    c : array_like
        Constant offset, theoretically ``full_like(te, 0.0)``.

    Returns
    -------
    array_like
        Initial s0 estimate ``s0 = (pv - c) / np.exp(-te/t2)``.

    """
    return (pv - c) / np.exp(-te / t2)
    
    
def rms(arr):
    """
    Calculate RMS of an array.

    Parameters
    ----------
    arr : array_like
         Input array

    Returns
    -------
    rms : float
        sqrt(mean(square(arr)))
    """
    return np.sqrt(np.mean(np.square(arr)))


class ROITimeSeries:
    """
    Samples at one image location (ROI) at numerous sample times.
    
    Estimating T1 and T2 relaxation parameters at any ROI requires a series
    of pixel values and sequence times (e.g. TI, TE, TR). This class is a
    wrapper for storing and accessing these parameters.
    
    Attributes
    ----------
    POI_mask : array
        Array the same size as the image. All values are 0, except a single 1
        at the point of interest (POI), the centre of the ROI.
        
    ROI_mask : array
        Array the same size as the image. Values in the ROI are coded as 1s,
        all other values are zero.
        
    pixel_values : list of arrays
        List of 1-D arrays of pixel values in ROI. The variance could be used
        as a measure of ROI homogeneity to identify  incorrect sphere location.
        
    times : list of floats
        If ``time_attr`` was used in the constructor, this list contains the
        value of ``time_attr``. Typically ``'EchoTime'`` or 
        ``'InversionTime'``.
        
    trs : list of floats
        Values of TR for each image.
        
    means : list of floats
        Mean pixel value of ROI for each image in series.
    """

    SAMPLE_ELEMENT = skimage.morphology.square(5)

    def __init__(self, dcm_images, poi_coords_row_col, time_attr=None,
                 kernel=None):
        """
        Create ROITimeSeries for ROI parameters at sequential scans.

        Parameters
        ----------
        dcm_images : list
            List of pydicom images of same object with different scan
            parameters (e.g. TIs or TEs). Typically ``ImageStack.images``.
        poi_coords_row_col : array
            Two element array with coordinates of point of interest (POI),
            typically the centre of the ROI, in row_col (y,x) format.
        time_attr : string, optional
            If present, lookup the DICOM attribute ``[time_attr]`` (typically
            ``'InversionTime'`` or ``'EchoTime'``) and store in the list
            ``self.times``. The default is ``None``, which does not create
            ``self.times``
        kernel : array_like, optional
            Structuring element which defines ROI size and shape, centred on
            POI. Each element should be 1 or 0, otherwise calculation of mean
            will be incorrect. If ``None``, use a 5x5 square. The default is
            ``None``.
        """

        if kernel is None:
            kernel = self.SAMPLE_ELEMENT
        self.POI_mask = np.zeros((dcm_images[0].pixel_array.shape[0],
                                  dcm_images[0].pixel_array.shape[1]),
                                 dtype=np.int8)
        self.POI_mask[poi_coords_row_col[0], poi_coords_row_col[1]] = 1

        self.ROI_mask = np.zeros_like(self.POI_mask)
        self.ROI_mask = scipy.ndimage.filters.convolve(self.POI_mask, kernel)
        self._time_attr = time_attr

        if time_attr is not None:
            self.times = [x[time_attr].value.real for x in dcm_images]
        self.pixel_values = [
            pixel_rescale(img)[self.ROI_mask > 0] for img in dcm_images]

        self.trs = [x['RepetitionTime'].value.real for x in dcm_images]

    def __len__(self):
        """Number of time samples in series."""
        return len(self.pixel_values)

    @property
    def means(self):
        """
        List of mean ROI values at different times.

        Returns
        -------
        List of mean pixel value in ROI for each sample.
        """
        return [np.mean(pvs) for pvs in self.pixel_values]


class ImageStack():
    """
    Object to hold image_slices and methods for T1, T2 calculation.
    """

    def __init__(self, image_slices, template_dcm, plate_number=None,
                 dicom_order_key=None):
        """
        Create ImageStack object.

        Parameters
        ----------
        image_slices : list of pydicom.FileDataSet objects
            List of pydicom objects to perform relaxometry analysis on.
            
        template_dcm : pydicom FileDataSet (or None)
            DICOM template object.
            
        plate_number : int {3,4,5}, optional
            For future use. Reference to the plate in the relaxometry phantom.
            The default is None.
            
        dicom_order_key : string, optional
            DICOM attribute to order images. Typically 'InversionTime' for T1
            relaxometry or 'EchoTime' for T2.
        """
        self.plate_number = plate_number
        # Store template pixel array, after scaling in 0028,1052 and 0028,1053
        # applied
        self.template_dcm = template_dcm
        if template_dcm is not None:
            self.template_px = pixel_rescale(template_dcm)

        self.dicom_order_key = dicom_order_key
        self.images = image_slices  # store images
        if dicom_order_key is not None:
            self.order_by(dicom_order_key)

        b0_val = self.images[0]['MagneticFieldStrength'].value
        if b0_val == 1.5:
            self.b0_str = '1.5T'
        elif b0_val == 3.0:
            self.b0_str = '3.0T'
        else:
            # TODO incorporate warning through e.g. logging module
            print('Unable to match B0 to default values. Using 1.5T.\n'
                  f" {self.images[0]['MagneticFieldStrength']}")
            self.b0_str = '1.5T'

    def template_fit(self, image_index=0):
        """
        Calculate transformation matrix to fit template to image.

        The template pixel array, self.template_px, is fitted to one of the
        images in self.images (default=0). The resultant RT matrix is stored as
        self.warp_matrix.

        This matrix can be used to map coordinates from template space to image
        space using transform_coords(...), or to map masks from template space
        to image space using cv2.warpAffine(...).

        To map from image space to template space, the 'inverse' RT matrix can
        be calculated using:
          inverse_transform_matrix = cv.invertAffineTransform(self.warp_matrix)

        Parameters
        ----------
        image_index : int, optional
            Index of image to be used for template matching. The default is 0.

        Returns
        -------
        warp_matrix : np.array
            RT transform matrix (2,3).

        Further details
        ---------------
        Untested for situations where the template matrix is larger than the
        image (lack of data!). Tested for images larger than templates.

        TODO
        ----
        This routine is suboptimal. It may be better to extract the bolt
        hole locations and fit from them, or run an edge-detection algorithm
        as pixel values are highly variable between scanners and manufacturers.

        Need to check if image is real valued, typically signed then shifted so
        background is 2048, or magnitude image. Currently it forces converts
        all images to magnitude images before regression.

        Despite these limitations, this method works well in practice for small
        angle rotations.
        """
        target_px = pixel_rescale(self.images[0])
        template_px = self.template_px

        # Pad template or target pixels if required
        scale_factor = len(target_px) / len(template_px)
        pad_size = np.subtract(template_px.shape, target_px.shape)
        assert pad_size[0] == pad_size[1], "Image matrices must be square."
        if pad_size[0] > 0:  # pad target--UNTESTED
            target_px = np.pad(target_px, pad_width=(0, pad_size[0]))
        elif pad_size[0] < 0:  # pad template
            template_px = np.pad(template_px, pad_width=(0, -pad_size[0]))

        # Always fit on magnitude images for simplicity. May be suboptimal
        self.template8bit = \
            cv.normalize(abs(template_px),
                         None, 0, 255, norm_type=cv.NORM_MINMAX,
                         dtype=cv.CV_8U)

        self.target8bit = cv.normalize(abs(target_px),
                                       None, 0, 255, norm_type=cv.NORM_MINMAX,
                                       dtype=cv.CV_8U)

        # initialise transformation fitting parameters.
        number_of_iterations = 500
        termination_eps = 1e-10
        criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT,
                    number_of_iterations, termination_eps)
        self.warp_matrix = scale_factor * np.eye(2, 3, dtype=np.float32)

        self.scaled_template8bit = cv.warpAffine(self.template8bit,
                                                 self.warp_matrix,
                                                 (self.template8bit.shape[1],
                                                  self.template8bit.shape[0]))

        # Apply transformation
        self.template_cc, self.warp_matrix = \
            cv.findTransformECC(self.template8bit, self.target8bit,
                                self.warp_matrix, criteria=criteria)

        self.warped_template8bit = cv.warpAffine(self.template8bit,
                                                 self.warp_matrix,
                                                 (self.template8bit.shape[1],
                                                  self.template8bit.shape[0]))

        return self.warp_matrix

    def plot_fit(self):
        """
        Visual representation of target fitting.

        Create 2x2 subplot showing 8-bit version of:
            1. Template
            2. Original image
            3. Overlay of (1) and (2)
            4. Overlay of RT transformed template and (2)
        """
        fig = plt.figure()
        plt.subplot(2, 2, 1)
        plt.imshow(self.template8bit, cmap='gray')
        plt.title('Template')
        plt.axis('off')

        ax = plt.subplot(2, 2, 2)
        self.plot_rois(new_fig=False)
        plt.title('Image')

        plt.subplot(2, 2, 3)
        plt.imshow(self.scaled_template8bit / 2 + self.target8bit / 2, cmap='gray')
        plt.title('Image / template overlay')
        plt.axis('off')

        plt.subplot(2, 2, 4)
        plt.imshow(self.warped_template8bit / 2 + self.target8bit / 2, cmap='gray')
        plt.title('Image / fitted template overlay')
        plt.axis('off')

        plt.tight_layout()

        return fig

    def plot_rois(self, new_fig=True):
        """
        Plot ROIs on image for visual check on template fitting.

        Parameters
        ----------
        new_fig : bool, optional
            Create new figure if True. Otherwise create in current axis (e.g.
            as a subplot). The default is True.

        Returns
        -------
            matplotlib figure handle if a new figure was created, otherwise
            None.

        """
        fig = None
        if new_fig:
            # Create image in a new figure (not a subplot)
            fig = plt.figure()

        plt.imshow(self.target8bit, cmap='gray')
        plt.axis('off')
        if hasattr(self, 'ROI_time_series'):
            combined_ROI_map = np.zeros_like(self.ROI_time_series[0].ROI_mask)
            for roi in self.ROI_time_series:
                combined_ROI_map += roi.ROI_mask
            lines = outline_mask(combined_ROI_map)
            for line in lines:
                plt.plot(line[1], line[0], color='r', alpha=1)

        return fig

    def order_by(self, att):
        """Order images by attribute (e.g. EchoTime, InversionTime)."""
        self.images.sort(key=lambda x: x[att].value.real)

    def generate_time_series(self, coords_row_col, fit_coords=True,
                             kernel=None):
        """
        Create list of ROITimeSeries objects.

        Parameters
        ----------
        coords_row_col : array_like
            Array of coordinates points of interest (POIs) for each centre of
            each ROI. They should be in [[col0, row0], [col1, row1], ...]
            format.
        fit_coords : bool, optional
            If ``True``, the coordinates provided are for the template ROIs and
            will be transformed to the image space using ``transfor_coords()``.
            The default is True.
        kernel : array, optional
            Structuring element which should be an array of 1s and possibly 0s.
            If ``None``, use the default from ``ROItimeSeries`` constructor.
            The default is None.
        """
        num_coords = np.size(coords_row_col, axis=0)
        if fit_coords:
            coords_row_col = transform_coords(coords_row_col, self.warp_matrix,
                                              input_row_col=True,
                                              output_row_col=True)

        self.ROI_time_series = []
        for i in range(num_coords):
            self.ROI_time_series.append(ROITimeSeries(
                self.images, coords_row_col[i], time_attr=self.dicom_order_key,
                kernel=kernel))

    def generate_fit_function(self):
        """Null method in base class, may be overwritten in subclass."""


class T1ImageStack(ImageStack):
    """
    Calculate T1 relaxometry.

    Overloads the following methods from ``ImageStack``:
        ``generate_fit_function``
        ``initialise_fit_parameters``
        ``find_relax_times``

    """

    def __init__(self, image_slices, template_dcm=None, plate_number=None):
        super().__init__(image_slices, template_dcm, plate_number=plate_number,
                         dicom_order_key='InversionTime')

    def generate_fit_function(self):
        """"Create T1 fit function for magnitude/signed image and variable TI."""
        #  check if image is signed or magnitude
        if np.all(pixel_rescale(self.images[0]) >= 0):
            mag_image = True
        else:
            mag_image = False
        self.fit_function, self.fit_jacobian, self.fit_eqn_str = \
            generate_t1_function(self.ROI_time_series[0].times,
                                 self.ROI_time_series[0].trs,
                                 mag_image=mag_image)

    def initialise_fit_parameters(self, t1_estimates):
        """
        Estimate fit parameters (t1, s0, a1) for T1 curve fitting.
        
        T1 estimates are provided.
        
        s0 is estimated using abs(est_t1_s0(ti, tr, t1_est, mean_pv))
            For each ROI, s0 is calculated using from both the smallest and
            largest TI, and the value with the largest mean_pv used. This
            guards against the case where division by a mean_pv close to zero
            causes a large rounding error.
            
        A1 is estimated as 2.0, the theoretical value assuming homogeneous B0
   
        Parameters
        ----------
        t1_estimates : array_like
            T1 values to seed estimation. These should be the manufacturer
            provided T1 values where known.

        Returns
        -------
        None.

        """
        self.t1_est = t1_estimates
        rois = self.ROI_time_series
        rois_first_mean = np.array([roi.means[0] for roi in rois])
        rois_last_mean = np.array([roi.means[-1] for roi in rois])
        s0_est_last = abs(est_t1_s0(rois[0].times[-1], rois[0].trs[-1],
                                    t1_estimates, rois_last_mean))
        s0_est_first = abs(est_t1_s0(rois[0].times[0], rois[0].trs[0],
                                     t1_estimates, rois_first_mean))
        self.s0_est = np.where(rois_first_mean > rois_last_mean,
                               s0_est_first, s0_est_last)
        self.a1_est = np.full_like(self.s0_est, 2.0)

    def find_relax_times(self):
        """
        Calculate T1 values. Access as ``image_stack.t1s``

        Returns
        -------
        None.

        """
        rois = self.ROI_time_series
        self.relax_fit = [scipy.optimize.curve_fit(self.fit_function,
                                                   rois[i].times,
                                                   rois[i].means,
                                                   p0=[self.t1_est[i],
                                                       self.s0_est[i],
                                                       self.a1_est[i]],
                                                   jac=self.fit_jacobian,
                                                   method='lm')
                          for i in range(len(rois))]

    @property
    def t1s(self):
        """List T1 values for each ROI."""
        return [fit[0][0] for fit in self.relax_fit]

    @property
    def relax_times(self):
        """List of T1 for each ROI."""
        return self.t1s


class T2ImageStack(ImageStack):
    """
    Calculate T2 relaxometry.

    Overloads the following methods from ``ImageStack``:
        ``generate_fit_function``
        ``initialise_fit_parameters``
        ``find_relax_times``

    """

    def __init__(self, image_slices, template_dcm=None, plate_number=None):
        super().__init__(image_slices, template_dcm, plate_number=plate_number,
                         dicom_order_key='EchoTime')

        self.fit_function = t2_function
        self.fit_jacobian = t2_jacobian
        self.fit_eqn_str = 's0 * np.exp(-te / t2)'

    def initialise_fit_parameters(self, t2_estimates):
        """
        Estimate fit parameters (t2, s0, c) for T2 curve fitting.
        
        T2 estimates are provided.
        
        s0 is estimated using est_t2_s0(te, t2_est, mean_pv, c).
            
        C is estimated as 0.0, the theoretical value assuming Gaussian noise.
   
        Parameters
        ----------
        t2_estimates : array_like
            T2 values to seed estimation. These should be the manufacturer
            provided T2 values where known.

        Returns
        -------
        None.

        """
        self.t2_est = t2_estimates
        rois = self.ROI_time_series
        rois_second_mean = np.array([roi.means[1] for roi in rois])
        self.c_est = np.full_like(self.t2_est, 0.0)
        # estimate s0 from second image--first image is too low.
        self.s0_est = est_t2_s0(rois[0].times[1], t2_estimates,
                                rois_second_mean, self.c_est)
        # Get maximum time to use on fitting algorithm (5*t2_est)
        # Truncating data after this avoids fitting Rician noise
        self.max_fit_times = 5 * t2_estimates

    def find_relax_times(self):
        """
        Calculate T2 values. Access as ``image_stack.t2s``.
        
        Uses the 'skip first echo' fit method [1]_. At times >> T2, the signal
        is dwarfed by Rician noise (for magnitude images). This can lead to
        inaccuracies in determining T2 as the measured signal does not tend to
        zero. To counter this, the signal is truncated after
        ``self.max_fit_times[i]``. At least three signals are used in the fit
        even if this exceeds the above criteria.
    
        Returns
        -------
        None.
        
        References
        ----------
        .. [1] McPhee, K. C., & Wilman, A. H. (2018). Limitations of skipping 
        echoes for exponential T2 fitting. Journal of Magnetic Resonance 
        Imaging, 48(5), 1432-1440. https://doi.org/10.1002/jmri.26052
        """
        # Require at least 4 samples (nb first sample is omitted)
        min_number_times = 4
        rois = self.ROI_time_series
        #  Omit the first image data from the curve fit. This is achieved by
        #  slicing rois[i].times[1:] and rois[i].means[1:]. Skipping odd echoes
        #  can be implemented with rois[i].times[1::2] and .means[1::2]
        bounds = ([0, 0], [np.inf, np.inf])
        self.relax_fit = []
        for i in range(len(rois)):
            times = [t for t in rois[i].times if t < self.max_fit_times[i]]
            if len(times) < min_number_times:
                times = rois[i].times[:min_number_times]
            self.relax_fit.append(
                scipy.optimize.curve_fit(self.fit_function,
                                         times[1:],
                                         rois[i].means[1:len(times)],
                                         p0=[self.t2_est[i],
                                             self.s0_est[i]],
                                         jac=self.fit_jacobian,
                                         bounds=bounds,
                                         method='trf'
                                         )
            )

    @property
    def t2s(self):
        """List of T2 values for each ROI."""
        return [fit[0][0] for fit in self.relax_fit]

    @property
    def relax_times(self):
        """List of T2 values for each ROI."""
        return self.t2s


def main(dcm_target_list, *, plate_number=None,
         show_template_fit=False, show_relax_fits=False, calc_t1=False,
         calc_t2=False, report_path=False, show_rois=False, verbose=False):
    """
    Calculate T1 or T2 values for relaxometry phantom.
    
    Note: either ``calc_t1`` or ``calc_t2`` (but not both) must be True.

    Parameters
    ----------
    dcm_target_list : list of pydicom.dataset.FileDataSet objects
        List of DICOM images of a plate of the HPD relaxometry phantom.
    plate_number : int
        Plate number of the HPD relaxometry phantom (either 4 or 5)
    show_template_fit : bool, optional
        If True, displays images to show template fitting and ROIs. The 
        default is False.
    show_relax_fits : bool, optional
        If True, displays graphs to show relaxometry fitting. The  default
        is False.
    show_rois : bool, optional
        If True, display original image with ROIs overlaid. The default is
        False
    calc_t1 : bool, optional
        Calculate T1. The default is False.
    calc_t2 : bool, optional
        Calculate T2. The default is False.
    report_path : path, optional
        If a valid file root, save template_fit images and relax_fit graphs.
        These must first have been generated with ``show_template_fit=True``
        or ``show_relax_fit=True``. The default is False.
    verbose : bool, optional
        Provide verbose output. If True, the following key / values will be
        added to the output dictionary:
            plate : plate_number,
            relaxation_type : 't1' | 't2',
            calc_times : list of T1|T2 for each sphere,
            manufacturers_times : list of manufacturer's values for T1|T2,
            frac_time_difference : (calc_times - manufacturers_times) / manufacturers_times
            institution_name=index_im.InstitutionName,
            manufacturer=index_im.Manufacturer,
            model=index_im.ManufacturerModelName,
            date=index_im.StudyDate,
            output_graphics=output_files_path
            detailed_output : dict with extensive information

        The default is False.

    Returns
    -------

    dict
        {
            rms_frac_time_difference : RMS fractional difference between
                calculated relaxometry times and manufacturer provided data.
        }
    """

    # check for exactly one relaxometry.py calculation
    if all([calc_t1, calc_t2]) or not any([calc_t1, calc_t2]):
        raise hazenlib.exceptions.ArgumentCombinationError(
            'Must specify either calc_t1=True OR calc_t2=True.')

    # check plate number specified and either 4 or 5
    try:
        plate_number = int(plate_number)  # convert to int if required
    except (ValueError, TypeError):
        pass  # will raise error at next statement

    if plate_number not in [4, 5]:
        raise hazenlib.exceptions.ArgumentCombinationError(
            'Must specify plate_number (4 or 5)')

    # Set up parameters specific to T1 or T2
    if calc_t1:
        ImStack = T1ImageStack
        relax_str = 't1'
        smooth_times = range(0, 1000, 10)
        try:
            template_dcm = pydicom.read_file(
                TEMPLATE_VALUES[f'plate{plate_number}']['t1']['filename'])
        except KeyError:
            print(f'Could not find template with plate number: {plate_number}.'
                  f' Please pass plate number as arg.')
            exit()

    elif calc_t2:
        ImStack = T2ImageStack
        relax_str = 't2'
        smooth_times = range(0, 500, 5)
        try:
            template_dcm = pydicom.read_file(
                TEMPLATE_VALUES[f'plate{plate_number}']['t2']['filename'])
        except KeyError:
            print(f'Could not find template with plate number: {plate_number}.'
                  f' Please pass plate number as arg.')
            exit()

    output_files_path = {}  # save path to output files
    image_stack = ImStack(dcm_target_list, template_dcm,
                          plate_number=plate_number)
    image_stack.template_fit()
    image_stack.generate_time_series(
        TEMPLATE_VALUES[f'plate{image_stack.plate_number}']
        ['sphere_centres_row_col'])
    image_stack.generate_fit_function()

    if show_template_fit:
        fig = image_stack.plot_fit()
        if report_path:
            old_dims = fig.get_size_inches()
            # Improve saved image quality
            fig.set_size_inches(24, 24)
            save_path = f'{report_path}_template_fit.png'
            for subplt in fig.get_axes():
                subplt.title.set_fontsize(40)
            fig.savefig(save_path, dpi=150)
            output_files_path['template_fit'] = save_path
            # Restore screen quality
            for subplt in fig.get_axes():
                subplt.title.set_fontsize('large')
            fig.set_size_inches(old_dims)

    if show_rois:
        fig = image_stack.plot_rois()
        plt.title(f'ROI positions ({relax_str.upper()}, plate {plate_number})')
        if report_path:
            save_path = f'{report_path}_rois.png'
            fig.savefig(save_path, dpi=300)
            output_files_path['rois'] = save_path

    relax_published = \
        TEMPLATE_VALUES [f'plate{image_stack.plate_number}'][relax_str] \
            ['relax_times'][image_stack.b0_str]
    image_stack.initialise_fit_parameters(relax_published)
    image_stack.find_relax_times()
    frac_time_diff = (image_stack.relax_times - relax_published) \
                     / relax_published

    if show_relax_fits:
        rois = image_stack.ROI_time_series
        fig = plt.figure()
        fig.suptitle(relax_str.upper() + ' relaxometry fits')

        for i in range(15):
            plt.subplot(5, 3, i + 1)
            plt.plot(smooth_times,
                     image_stack.fit_function(
                         np.array(smooth_times),
                         *np.array(image_stack.relax_fit[i][0])),
                     'b-')
            plt.plot(rois[i].times, rois[i].means, 'rx')
            if i == 14:
                plt.title(f'[Free water] fit={image_stack.relax_times[i]:.4g}',
                          fontsize=8)
            else:
                plt.title(f'[{i + 1}] fit={image_stack.relax_times[i]:.4g}, '
                          f'pub={relax_published[i]:.4g} '
                          f'({frac_time_diff[i] * 100:+.2f}%)',
                          fontsize=8)
        # plt.tight_layout(rect=(0,0,0,0.95)) # Leave suptitle space at top
        if report_path:
            # Improve saved image quality
            old_dims = fig.get_size_inches()
            fig.set_size_inches(9, 15)
            plt.tight_layout(rect=(0, 0, 1, 0.97))
            save_path = f'{report_path}_decay_graphs.png'
            fig.savefig(save_path, dpi=300)
            output_files_path['decay_graphs'] = save_path
            # Restore screen quality
            fig.set_size_inches(old_dims)

    # Generate output dict
    index_im = image_stack.images[0]
    # last value is for background water. Strip before calculating RMS frac error
    output = {'rms_frac_time_difference' : rms(frac_time_diff[:-1])}
    if verbose:
        output.update(dict(plate=image_stack.plate_number,
                      relaxation_type=relax_str,
                      calc_times=image_stack.relax_times,
                      manufacturers_times=relax_published,
                      frac_time_difference=frac_time_diff,
                      institution_name=index_im.InstitutionName,
                      manufacturer=index_im.Manufacturer,
                      model=index_im.ManufacturerModelName,
                      date=index_im.StudyDate,
                      output_graphics=output_files_path))

        detailed_output = {
            'filenames': [im.filename for im in image_stack.images],
            'ROI_means': {i: im.means for i, im in enumerate(image_stack.ROI_time_series)},
            'TE': [im.EchoTime for im in image_stack.images],
            'TR': [im.RepetitionTime for im in image_stack.images],
            'TI': [im.InversionTime if hasattr(im, 'InversionTime') else None \
                   for im in image_stack.images],
            # fit_paramters (T1) = [[T1, s0, A1] for each ROI]
            # fit_parameters (T2) = [[T2, s0, C] for each ROI]
            'fit_parameters': [param[0] for param in image_stack.relax_fit],
            'fit_equation': image_stack.fit_eqn_str
        }

        output['detailed'] = detailed_output

    output_key = f"{index_im.SeriesDescription}_{index_im.SeriesNumber}_{index_im.InstanceNumber}_" \
                 f"P{image_stack.plate_number}_{relax_str}"

    plt.show()
    return {output_key: output}
