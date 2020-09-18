# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 14:15:35 2020

@author: Paul Wilson

Overview
========
	1. Import list of DICOM files. The Caliber (HPD) system phantom should be
        scanned https://qmri.com/system-phantom/. 
        TODO: add protocol details.
	2. Create container object / array (all-slices) containing:
        a. Target images (to be ordered by TE or TI). Should be same position on same phantom, different TE or TI
		b. Transformation matrix to map template image spcae to target image space
        c. List of coordinates of centres of each sphere in template image (to enable ROI generation)
	3. Image alignment-generate RT (rotation - translation) transformation matrix
        fitting a Euclidean transformation.
		a. Poss use https://www.learnopencv.com/image-alignment-ecc-in-opencv-c-python/ , https://alexanderpacha.com/2018/01/29/aligning-images-an-engineers-solution/
		b. Generate coordinates of sphere centres by transforming list of coordinates from template.
			i. CHECK-Display image with overlays showing sampling locations AND mounting pins / coffin (to check alignment).
        c. Create mask for each sphere by placing structuring element (e.g. binary disk, diameter=?5px) centred on taget sphere coordinates.
            i. CHECK-overlay contour map on target image.
	4. For each sphere:
		a. Find array of mean PVs foreach slice.
			i. CHK--is max/min range too big--indicates poor position localisation
		b. Fit array to decay curve.
			i. Use different fitting algorithm for T1, T2. CHK sampling is relevant--i.e. different TIs, TEs at each slice.
		c. Numeric and graphical output. Poss include known values if available.


TODO
====



FEATURE ENHANCEMENT
===================
Template fit on bolt holes--possibly better with large rotation angles and faster
    -have bolthole template, find 3 positions in template and image, figure out
    transformation.
    
Use normalised structuring element in ROITimeSeries. This will allow correct
calculation of mean if elements are not 0 or 1.

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

# import hazenlib
import hazenlib.exceptions

# Use dict to store template and reference information
# Coordinates are in array format (row,col), rather than plt.patches 
# format (col,row)
#
# Access as:
#    TEMPLATE_VALUES[f'plate{plate_num}']['sphere_centres_row_col']
#    TEMPLATE_VALUES[f'plate{plate_num}']['t1']['filename'] # or 't2'
#    TEMPLATE_VALUES[f'plate{plate_num}']['t1']['relax_times']

TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            'data', 'relaxometry')
TEMPLATE_VALUES = {
    'plate3':{
        'sphere_centres_row_col':(),
        'bolt_centres_row_col':(),
        't1':{
            'filename':'',
            'relax_times':[]}},
    
    'plate4':{
        'sphere_centres_row_col':(
            (56, 94), (62, 117), (81, 132), (105, 134), (125, 120), (133, 99),
            (127, 75), (108, 60), (84, 59), (64, 72), (80, 81), (78, 111),
            (109, 113), (111, 82)),
        'bolt_centres_row_col':(),
        't1':{
            'filename':os.path.join(TEMPLATE_DIR, 'Plate4_T1_signed'),
            'relax_times':
                np.array([2376.0, 2183.0, 1870.0, 1539.0, 1237.0, 1030.0,
                          752.2, 550.2, 413.4, 292.9, 194.9, 160.2, 106.4,
                          83.3])},
        't2':{
            'filename':os.path.join(TEMPLATE_DIR, 'Plate4_T2'),
            'relax_times':
                np.array([939.4, 594.3, 416.5, 267.0, 184.9, 140.6, 91.76,
                          64.84, 45.28, 30.62, 19.76, 15.99, 10.47, 8.15])}},
    
    'plate5':{
        'sphere_centres_row_col':(
            (56, 95), (62, 117), (81, 133), (104, 134), (124, 121), (133, 98),
            (127, 75), (109, 61), (84, 60), (64, 72), (80, 81), (78, 111),
            (109, 113), (110, 82)),
        'bolt_centres_row_col':((52, 80), (92, 141), (138, 85)),
        't1':{
            'filename':os.path.join(TEMPLATE_DIR, 'Plate5_T1_signed'),
            'relax_times':
                np.array([2033, 1489, 1012, 730.8, 514.1, 367.9, 260.1, 184.6,
                          132.7, 92.7, 65.4, 46.32, 32.45, 22.859])},
        't2':{
            'filename':os.path.join(TEMPLATE_DIR, 'Plate5_T2'),
            'relax_times':
                np.array([1669.0, 1244.0, 859.3, 628.5, 446.3, 321.2, 227.7,
                          161.9, 117.1, 81.9, 57.7, 41.0, 28.7, 20.2])}}}

    
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
    lines += [([s[0]-.5, s[0]-.5], [s[1]+.5, e[1]+.5]) for s, e
              in zip(starts, ends)]

    im1 = np.diff(im1, n=1, axis=0).T
    starts = np.argwhere(im1 == 1)
    ends = np.argwhere(im1 == -1)
    lines += [([s[1]+.5, e[1]+.5], [s[0]-.5, s[0]-.5]) for s, e
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


def pixel_LUT(dcmfile):
    """
    Transforms pixel values according to LUT in DICOM header.
    
    DICOM pixel values arrays cannot directly represent signed or float values.
    This function converts the ``.pixel_array`` using the LUT values in the
    DICOM header.

    Parameters
    ----------
    dcmfile : Pydicom.dataset.FileDataset
        DICOM file containing one image.

    Returns
    -------
    numpy.array
        Values in ``dcmfile.pixel_array`` transformed using DICOM LUT.

    """
    return pydicom.pixel_data_handlers.util.apply_modality_lut(
            dcmfile.pixel_array, dcmfile)


def generate_t1_function(ti_interp_vals, tr_interp_vals, mag_image=False):
    """
    Generate T1 signal function and jacobian with interpreted TRs.
    
    Signal intensity on T1 decay is a function of both TI and TR. Ideally, TR
    should be constant and at least 5*T1. However, scan time can be reduced by
    allowing a shorter TR which increases at long TIs. For example::
        TI |   50 |  100 |  200 |  400 |  600 |  800 
        ---+------+------+------+------+------+------
        TR | 1000 | 1000 | 1000 | 1260 | 1860 | 2460
    
    This function factory returns a function which calculates the signal
    magnitude using the expression::
        S = a0 * (1 - a1 * np.exp(-TI / t1) + np.exp(-TR / t1))
    where ``a0`` is the recovered intensity, ``a1`` is theoretically 2.0 but
    varies due to inhomogeneous B0 field, ``t1`` is the longitudinal
    relaxation time, and the repetition time, ``TR``, is calculted from ``TI``
    using piecewise linear interpolation.

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
        S = a0 * (1 - a1 * np.exp(-TI / t1) + np.exp(-TR / t1))
    
    t1_jacobian : function
        Returns tuple of partial derivatives for curve fitting.
    """
    #  Create piecewise liner fit function. k=1 gives linear, s=0 ensures all
    #  points are on line. Using UnivariateSpline (rather than numpy.interp()
    #  enables derivative calculation if required.
    tr = UnivariateSpline(ti_interp_vals, tr_interp_vals, k=1, s=0)
    # tr_der = tr.derivative()
    
    def _t1_function_signed(ti, t1, a0, a1):
        pv = a0 * (1 - a1 * np.exp(-ti / t1) + np.exp(-tr(ti) / t1)) 
        return pv
    
    def t1_function(ti, t1, a0, a1):
        pv = _t1_function_signed(ti, t1, a0, a1)
        if mag_image:
            return abs(pv)
        else:
            return pv
    
    def t1_jacobian(ti, t1, a0, a1):
        t1_der = a0 / (t1**2) * (-ti * a1* np.exp(-ti/t1) + tr(ti)
                                 * np.exp(-tr(ti)/t1))
        a0_der = 1 - a1 * np.exp(-ti/t1) + np.exp(-tr(ti)/t1)
        a1_der = -a0 * np.exp(-ti/t1)
        jacobian = np.array([t1_der, a0_der, a1_der])
        
        if mag_image:
            pv = _t1_function_signed(ti, t1, a0, a1)
            jacobian = (jacobian * (pv>=0)) - (jacobian * (pv<0))
        
        return jacobian.T
    
    return t1_function, t1_jacobian


def est_t1_a0(ti, tr, t1, pv):
    """
    Return initial guess of A0 for T1 curve fitting.

    """
    return -pv / (1 - 2*np.exp(-ti/t1) + np.exp(-tr/t1))


def t2_function(te, t2, a0, c):
    """
    Signal formaula from TE, T2, A0 and C.
    
    Calculates pixel intensity from::
        pv = a0 * np.exp(-te / t2) + c

    Parameters
    ----------
    te : array_like
        Echo times.
    t2 : float
        T2 decay constant.
    a0 : float
        Initial signal magnitude.
    c : float
        Constant offset, theoretically 0, but models Rician noise in magnitude
        data.

    Returns
    -------
    pv : array_like
        Theoretical pixel values at each TE.
        
    Notes
    -----
    The '+ c' constant models Rician noise in magnitude images (where Gaussian
    noise in low signals gets rectified producing a bias). This is an
    acceptable model for short T2 samples. However, it reduces the fit on long
    T2 samples as the slow decay resembles a constant and the signal never
    reaches the noise floor.

    """
    #c=0  # remove comment to disable constant in equation
    pv = a0 * np.exp(-te / t2) + c
    # uncomment to implement noise floor of 'c'. Also need to change t2_jacobian
    #pv = np.fmax(a0 * np.exp(-te / t2), c) 
    return pv


def t2_jacobian(te, t2, a0, c):
    """
    Jacobian of ``ts_function`` used for curve fitting.

    Parameters
    ----------
    te : array_like
        Echo times.
    t2 : float
        T2 decay constant.
    a0 : float
        Initial signal magnitude.
    c : float
        Constant offset, theoretically 0.

    Returns
    -------
    array
        [t2_der, a0_der, c_der].T, where x_der is a 1-D array of the partial
        derivatives at each ``te``.

    """
    t2_der = a0 * te / t2**2 * np.exp(-te/t2)
    a0_der = np.exp(-te / t2)
    c_der = np.full_like(t2_der, 1.0)
    #pv = t2_function(te, t2, a0, c)
    #c_der = np.where(pv > c, np.zeros_like(t2_der), np.full_like(t2_der, 1.0))
    jacobian = np.array([t2_der, a0_der, c_der])
    return jacobian.T


def est_t2_a0(te, t2, pv, c):
    """
    Initial guess at A0 to seed curve fitting.

    Parameters
    ----------
    te : array_like
        Echo time(s).
    t2 : array_like
        T2 decay constant.
    pv : array_like
        Mean pixel value in ROI with ``te`` echo time.
    c : array_like
        Constant offset, theoretically ``full_like(te, 0.0)``.

    Returns
    -------
    array_like
        Initial A0 estimate ``A0 = (pv - c) / np.exp(-te/t2)``.

    """
    return (pv - c) / np.exp(-te/t2)


class ROITimeSeries():
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
        as a measure of ROI homogeneity to identiy incorrect sphere location.
        
    times : list of floats
        If ``time_attr`` was used in the constructor, this list contains the
        value of ``time_attr``. Typically ``'EchoTime'`` or 
        ``'InversionTime'``.
        
    trs : list of floats
        Values of TR for each image.
        
    means :  list of floats
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
            ``'InversionTime'`` or ``'EchoTime'`` and store in the list
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
            pixel_LUT(img)[self.ROI_mask > 0] for img in dcm_images]
        
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
        # return [np.mean(self.pixel_values[i]) for i in range(len(self))]
        return [np.mean(pvs) for pvs in self.pixel_values]


class ImageStack():
    """Object to hold image_slices and methods for T1, T2 calculation."""

    
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
        self.plate_number=plate_number
        # Store template pixel array, after LUT in 0028,1052 and 0028,1053
        # applied
        self.template_dcm = template_dcm
        if template_dcm is not None:
            self.template_px = pixel_LUT(template_dcm)
            
        self.dicom_order_key = dicom_order_key
        self.images = image_slices  # store images
        if dicom_order_key is not None:
            self.order_by(dicom_order_key)


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

        TODO
        ----
        This routine is suboptimal. It may be better to extract the bolt
        hole locations and fit from them, or run an edge-detection algorithm
        as pixel values are highly variable between scanners and manufacturers.

        Need to check if image is real valued, typically signed then shifted so
        background is 2048, or magnitude image. Currently it assumes magnitude
        image.

        """
        target_px = pixel_LUT(self.images[0])

        # Always fit on magnitude images for simplicity. May be suboptimal
        # TODO check for better solution
        self.template8bit = \
            cv.normalize(abs(self.template_px),
                         None, 0, 255, norm_type=cv.NORM_MINMAX,
                         dtype=cv.CV_8U)

        self.target8bit = cv.normalize(abs(target_px),
                                       None, 0, 255, norm_type=cv.NORM_MINMAX,
                                       dtype=cv.CV_8U)

        # initialise transofrmation fitting parameters.
        number_of_iterations = 500
        termination_eps = 1e-10
        criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT,
                    number_of_iterations, termination_eps)
        self.warp_matrix = np.eye(2, 3, dtype=np.float32)

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
        plt.figure()
        plt.subplot(2, 2, 1)
        plt.imshow(self.template8bit, cmap='gray')
        plt.title('Template')
        plt.axis('off')

        plt.subplot(2, 2, 2)
        plt.imshow(self.target8bit, cmap='gray')
        plt.title('Image')
        plt.axis('off')
        if hasattr(self, 'ROI_time_series'):
            combined_ROI_map = np.zeros_like(self.ROI_time_series[0].ROI_mask)
            for roi in self.ROI_time_series:
                combined_ROI_map += roi.ROI_mask
            lines = outline_mask(combined_ROI_map)
            for line in lines:
                plt.plot(line[1], line[0], color='r', alpha=1)

        plt.subplot(2, 2, 3)
        plt.imshow(self.template8bit/2 + self.target8bit/2, cmap='gray')
        plt.title('Image / template overlay')
        plt.axis('off')

        plt.subplot(2, 2, 4)
        plt.imshow(self.warped_template8bit/2 + self.target8bit/2, cmap='gray')
        plt.title('Image / fitted template overlay')
        plt.axis('off')

        plt.tight_layout()

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
            format (i.e. y, x).
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
        """Null method in base class, may be overwritted in subclass."""


class T1ImageStack(ImageStack):
    """Calculates T1 relaxometry."""

    def __init__(self, image_slices, template_dcm=None, plate_number=None):
        super().__init__(image_slices, template_dcm, plate_number=plate_number,
                         dicom_order_key='InversionTime')
    
    def generate_fit_function(self):
        """"Create T1 fit function for magnitude/signed image and variable TI."""
        #  check if image is signed or magnitude
        if np.all(pixel_LUT(self.images[0]) >= 0):
            mag_image = True
        else:
            mag_image = False
        self.fit_function, self.fit_jacobian = \
            generate_t1_function(self.ROI_time_series[0].times,
                                 self.ROI_time_series[0].trs,
                                 mag_image = mag_image)
    
    def initialise_fit_parameters(self, t1_estimates):
        """
        Estimate fit parameters (t1, a0, a1) for T1 curve fitting.
        
        T1 estimates are provided.
        
        A0 is estimated using abs(est_t1_roi(ti, tr, t1_est, mean_pv))
            For each ROI, A0 is calculated using from both the smallest and
            largest TI, and the value with the largest mean_pv used. This
            guards against the case where division by a mean_pv close to zero
            causes a large rounding error.
            
        A1 is estimated as 2.0, the theoretical value assuming homogeneous B0
   
        Parameters
        ----------
        t1_estimates : array_like
            T1 values to seed estimation. These should be the manufacturer
            provided T1 values where known. The default is plate5_t1_values.

        Returns
        -------
        None.

        """
        self.t1_est = t1_estimates
        rois = self.ROI_time_series
        rois_first_mean = np.array([roi.means[0] for roi in rois])
        rois_last_mean = np.array([roi.means[-1] for roi in rois])
        a0_est_last = abs(est_t1_a0(rois[0].times[-1], rois[0].trs[-1],
                                    t1_estimates, rois_last_mean))
        a0_est_first = abs(est_t1_a0(rois[0].times[0], rois[0].trs[0],
                                     t1_estimates, rois_first_mean))
        self.a0_est = np.where(rois_first_mean > rois_last_mean,
                               a0_est_first, a0_est_last)
        self.a1_est = np.full_like(self.a0_est, 2.0)

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
                                                       self.a0_est[i],
                                                       self.a1_est[i]],
                                                   jac=self.fit_jacobian,
                                                   method='lm')
                    for i in range(len(rois))]
         
    @property
    def t1s(self):
        """List of T1 values for each ROI."""
        return [fit[0][0] for fit in self.relax_fit]
    
    @property
    def relax_times(self):
        return self.t1s


class T2ImageStack(ImageStack):
    """Calculate T2 relaxometry."""
    
    def __init__(self, image_slices, template_dcm=None, plate_number=None):
        super().__init__(image_slices, template_dcm, plate_number=plate_number,
                         dicom_order_key='EchoTime')
        
        self.fit_function = t2_function
        self.fit_jacobian = t2_jacobian
        
    def initialise_fit_parameters(self, t2_estimates):
        self.t2_est = t2_estimates
        rois = self.ROI_time_series
        rois_second_mean = np.array([roi.means[1] for roi in rois])
        self.c_est = np.full_like(self.t2_est, 0.0)
        # estimate a0 from second image--first image is too low.
        self.a0_est = est_t2_a0(rois[0].times[1], t2_estimates,
                                rois_second_mean, self.c_est)


    def find_relax_times(self):
        """
        Calculate T2 values. Access as ``image_stack.t2s``
        
        Uses the 'skip first echo' fit method [1]_.
    
        Returns
        -------
        None.
        
        References
        ----------
        .. [1] McPhee, K. C., & Wilman, A. H. (2018). Limitations of skipping 
        echoes for exponential T2 fitting. Journal of Magnetic Resonance 
        Imaging, 48(5), 1432-1440. https://doi.org/10.1002/jmri.26052
        """
        rois = self.ROI_time_series
        #  Omit the first image data from the curve fit. This is achieved by
        #  slicing rois[i].times[1:] and rois[i].means[1:]. Skipping odd echoes
        #  can be implemented with rois[i].times[1::2] and .means[1::2]
        bounds = ([0, 0, 0], [np.inf, np.inf, 10])
        self.relax_fit = [scipy.optimize.curve_fit(self.fit_function, 
                                                    rois[i].times[1:],
                                                    rois[i].means[1:],
                                                    p0=[self.t2_est[i],
                                                        self.a0_est[i],
                                                        self.c_est[i]],
                                                    jac=self.fit_jacobian,
                                                    bounds=bounds,
                                                    method='trf')
                          for i in range(len(rois))]
    
    @property
    def t2s(self):
        """List of T2 values for each ROI."""
        return [fit[0][0] for fit in self.relax_fit]

    @property
    def relax_times(self):
        return self.t2s



def main(dcm_target_list, template_dcm=None, show_template_fit=True,
         show_relax_fits=True, calc_t1 = False, calc_t2 = False,
         plate_number=None, report_path=False):

    # check for exactly one relaxometry calculation
    if all([calc_t1, calc_t2]) or not any([calc_t1, calc_t2]):
        raise hazenlib.exceptions.ArgumentCombinationError(
            'Must specify either calc_t1=True OR calc_t2=True.')
    
    if calc_t1:
        ImStack = T1ImageStack
        relax_str = 't1'
        smooth_times = range(0,1000,10)
    elif calc_t2:
        ImStack = T2ImageStack
        relax_str = 't2'
        smooth_times = range(0,500,5)
        
    image_stack = ImStack(dcm_target_list, template_dcm,
                         plate_number=plate_number)
    image_stack.template_fit()
    image_stack.generate_time_series(
        TEMPLATE_VALUES[f'plate{image_stack.plate_number}']
        ['sphere_centres_row_col'])
    image_stack.generate_fit_function()

    if show_template_fit:
        image_stack.plot_fit()
    
    relax_published = \
        TEMPLATE_VALUES[f'plate{image_stack.plate_number}'][relax_str]\
            ['relax_times']
    image_stack.initialise_fit_parameters(relax_published)
    image_stack.find_relax_times()
    
    if show_relax_fits:
        smooth_times = range(0,1000,10)
        rois = image_stack.ROI_time_series
        fig = plt.figure()
        fig.suptitle(relax_str.upper() + ' relaxometry fits')
        percent_diff = (image_stack.relax_times - relax_published) * 100 \
            / relax_published
        for i in range(14):
            plt.subplot(4,4,i+1)
            plt.plot(smooth_times, 
                      image_stack.fit_function(
                          np.array(smooth_times),
                          *np.array(image_stack.relax_fit[i][0])),
                      'b-')
            plt.plot(rois[i].times, rois[i].means, 'rx')
            plt.title(f'[{i+1}] fit={image_stack.relax_times[i]:.4g}, '
                      f'pub={relax_published[i]:.4g} '
                      f'({percent_diff[i]:+.2f}%)',
                      fontsize=8)
        plt.tight_layout(rect=(0,0,0.95,1)) # Leave suptitle space at top

    
    return image_stack  # for debugging only
 
      

# Code below is for development only and should be deleted before release.
if __name__ == '__main__':

    import logging  # better to set up module level logging
    from pydicom.errors import InvalidDicomError
    
    #Test Error checking
    #main([], calc_t1=True, calc_t2=True)
    #main([], calc_t1=False, calc_t2=False)
    
    calc_t1 = False
    calc_t2 = False
    # comment lines below to supress calculation
    calc_t1 = True;
    #calc_t2 = True
    plate_num = 5
    
    if calc_t1:
        template_dcm = pydicom.read_file(
            TEMPLATE_VALUES[f'plate{plate_num}']['t1']['filename'])
    
        # get list of pydicom objects
        # target_folder = os.path.join(
        #     os.path.dirname(os.path.realpath(__file__)), '..', 'tests', 'data',
        #     'relaxometry', 'T1', 'site2 20180925', 'plate 5')

        target_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                      '..', 'tests', 'data', 'relaxometry', 'T1',
                                      'site1 20200218', 'plate 5')
        
        #target_folder = "C:\OneDrive\BHSCT\OneDrive - Belfast Health & Social Care Trust\DICOM files\T1 measurement anomaly"
 
        dcm_target_list = []
        (_,_,filenames) = next(os.walk(target_folder)) # get filenames, don't go to subfolders
        for filename in filenames:
            try:
                with pydicom.dcmread(os.path.join(target_folder, filename)) as dcm_target:
                    dcm_target_list.append(dcm_target)
            except InvalidDicomError:
                logging.info(' Skipped non-DICOM file %r',
                             os.path.join(target_folder, filename))
    
        t1_image_stack = main(dcm_target_list, template_dcm, calc_t1=True,
                              show_template_fit=True, show_relax_fits=True,
                              plate_number=plate_num)
        t1_rois = t1_image_stack.ROI_time_series
    
    if calc_t2:
        template_dcm = pydicom.read_file(
            TEMPLATE_VALUES[f'plate{plate_num}']['t2']['filename'])

    
        # get list of pydicom objects
        target_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                     '..', 'tests', 'data', 'relaxometry', 'T2',
                                     'site1 20200218', 'plate 4')
        dcm_target_list = []
        (_,_,filenames) = next(os.walk(target_folder)) # get filenames, don't go to subfolders
        for filename in filenames:
            try:
                with pydicom.dcmread(os.path.join(target_folder, filename)) as dcm_target:
                    dcm_target_list.append(dcm_target)
            except InvalidDicomError:
                logging.info(' Skipped non-DICOM file %r',
                             os.path.join(target_folder, filename))
    
        t2_image_stack = main(dcm_target_list, template_dcm,
                              show_template_fit=True, calc_t2=True,
                              plate_number=plate_num)
        t2_rois = t2_image_stack.ROI_time_series
