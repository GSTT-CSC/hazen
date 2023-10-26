"""
Measure T1 and T2 in Caliber relaxometry phantom.

Introduction
============

This module determines the T1 and T2 decay constants for the relaxometry
spheres in the Caliber (HPD) system phantom
qmri.com/qmri-solutions/t1-t2-pd-imaging-phantom (plates 4 and 5).
Values are compared to published values (without temperature correction).
Graphs of fit and phantom registration images can optionally be produced.


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
3. A ROI is generated for each target sphere using stored coordinates, the RT
    transformation above, and a structuring element (default is a 5x5 boxcar).
4. Store pixel data for each ROI at various times, in an ``ROITimeSeries``
    object. A list of these objects is stored in 
    ``ImageStack.ROI_time_series``.
5. Generate the fit function. For T1 this looks up TR for the given TI 
    (using piecewise linear interpolation if required) and determines if a
    magnitude or signed image is used. No customisation is required for T2
    measurements.
6. Determine relaxation time (T1 or T2) by fitting the decay equation to
    the ROI data for each sphere. The published values of the relaxation
    times are used to seed the optimisation algorithm. A Rician noise model is
    used for T2 fitting [1]_. Optionally plot and save the decay curves.
7. Return plate number, relaxation type (T1 or T2), measured relaxation
    times, published relaxation times, and fractional differences in a
    dictionary.

References
==========
.. [1] Raya, J.G., Dietrich, O., Horng, A., Weber, J., Reiser, M.F.
and Glaser, C., 2010. T2 measurement in articular cartilage: impact of the
fitting method on accuracy and precision at low SNR. Magnetic Resonance in
Medicine: An Official Journal of the International Society for Magnetic
Resonance in Medicine, 63(1), pp.181-193.

Feature enhancements
====================
Template fit on bolt holes--possibly better with large rotation angles
    -have bolthole template, find 3 positions in template and image, figure out
    transformation.

Template fit on outline image--possibly run though edge detection algorithms
then fit.

Use normalised structuring element in ROITimeSeries. This will allow correct
calculation of mean if elements are not 0 or 1.

Get r-squared measure of fit.

"""
import json
import os.path
import pathlib

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pydicom
import scipy.ndimage
import scipy.optimize
import skimage.morphology
from scipy.interpolate import UnivariateSpline
from scipy.special import i0e, ive

import hazenlib.exceptions
from hazenlib.HazenTask import HazenTask
from hazenlib.relaxometry_params import (
    MAX_RICIAN_NOISE, SEED_RICIAN_NOISE, TEMPLATE_VALUES, SMOOTH_TIMES,
    TEMPLATE_FIT_ITERS, TERMINATION_EPS
)

# Use dict to store template and reference information
# Coordinates are in array format (row,col), rather than plt.patches 
# format (col,row)
#
# Access as:
#    TEMPLATE_VALUES[f'plate{plate_num}']['sphere_centres_row_col']
#    TEMPLATE_VALUES[f'plate{plate_num}']['t1'|'t2']['filename']
#    TEMPLATE_VALUES[f'plate{plate_num}']['t1'|'t2']['1.5T'|'3.0T']['relax_times']


class Relaxometry(HazenTask):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, calc: str = 'T1', plate_number=None, verbose=False):
        """
        Calculate T1 or T2 values for relaxometry phantom.

        Note: either ``calc_t1`` or ``calc_t2`` (but not both) must be True.
        Variables set in parent class:
        data : list of pydicom.dataset.FileDataSet objects
            List of DICOM images of a plate of the HPD relaxometry phantom.
        report : bool, optional
            Whether to save images showing the measurement details
        report_dir : path, optional
            Folder path to save images to. The default is False.

        Parameters
        ----------
        calc : str, required
            Whether to calculate T1 or T2 relaxation. Default is T1.
        plate_number : str, required
            Plate number of the HPD relaxometry phantom (either 4 or 5)
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

        # check plate number specified: should be either 4 or 5
        try:
            plate_number = int(plate_number)  # convert to int if required
        except (ValueError, TypeError):
            pass  # will raise error at next statement
        if plate_number not in [4, 5]:
            raise hazenlib.exceptions.ArgumentCombinationError(
                'Must specify plate_number (4 or 5)')

        # Set up parameters specific to T1 or T2
        relax_str = calc.lower()

        if calc in ['T1', 't1']:
            image_stack = T1ImageStack(self.dcm_list)
            try:
                template_dcm = pydicom.read_file(
                    TEMPLATE_VALUES[f'plate{plate_number}'][relax_str]['filename'])
            except KeyError:
                print(f'Could not find template with plate number: {plate_number}.'
                    f' Please pass plate number as arg.')
                exit()
        elif calc in ['T2', 't2']:
            image_stack = T2ImageStack(self.dcm_list)
            try:
                template_dcm = pydicom.read_file(
                    TEMPLATE_VALUES[f'plate{plate_number}'][relax_str]['filename'])
            except KeyError:
                print(f'Could not find template with plate number: {plate_number}.'
                    f' Please pass plate number as arg.')
                exit()
        else:
            print("Please provide 'T1' or 'T2' for the --calc argument.")
            exit()

        warp_matrix = image_stack.template_fit(template_dcm)
        image_stack.generate_time_series(
            TEMPLATE_VALUES[f'plate{plate_number}']['sphere_centres_row_col'],
            warp_matrix=warp_matrix)
        # only applies to T1
        image_stack.generate_fit_function()

        # Published relaxation time for matching plate and T1/T2
        relax_published = TEMPLATE_VALUES [f'plate{plate_number}'][relax_str] \
                ['relax_times'][image_stack.b0_str]
        s0_est = image_stack.initialise_fit_parameters(relax_published)

        image_stack.find_relax_times(relax_published, s0_est)
        frac_time_diff = (image_stack.relax_times - relax_published) \
                        / relax_published
        # last value is for background water. Strip before calculating RMS frac error
        frac_time = frac_time_diff[:-1]
        RMS_frac_error = np.sqrt(np.mean(np.square(frac_time)))

        # Generate results dict
        index_im = self.dcm_list[0]
        results = self.init_result_dict()
        output_key = '_'.join([self.img_desc(index_im), str(plate_number), relax_str])
        results['file'] = output_key

        results['measurement'] = {
            'rms_frac_time_difference' : round(RMS_frac_error, 3)
        }

        if self.report:
            img_path = os.path.join(self.report_path, output_key)
            # Show template fit
            template_fit_fig = image_stack.plot_fit()
            # Improve saved image quality
            template_fit_fig.set_size_inches(24, 24)
            for subplt in template_fit_fig.get_axes():
                subplt.title.set_fontsize(40)
            template_fit_img = f'{img_path}_template_fit.png'
            template_fit_fig.savefig(template_fit_img, dpi=150)
            self.report_files.append(('template_fit', template_fit_img))

            # Show ROIs
            roi_fig = image_stack.plot_rois()
            plt.title(f'ROI positions ({calc.upper()}, plate {plate_number})')
            roi_img = f'{img_path}_rois.png'
            roi_fig.savefig(roi_img, dpi=300)
            self.report_files.append(('rois', roi_img))

            # Show relax fits
            rois = image_stack.ROI_time_series
            relax_fit_fig = plt.figure()
            relax_fit_fig.suptitle(calc.upper() + ' relaxometry fits')
            smooth_times = SMOOTH_TIMES[calc.lower()]
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
            # Improve saved image quality
            relax_fit_fig.set_size_inches(9, 15)
            plt.tight_layout(rect=(0, 0, 1, 0.97))
            relax_fit_img = f'{img_path}_decay_graphs.png'
            relax_fit_fig.savefig(relax_fit_img, dpi=300)
            self.report_files.append(('decay_graphs', relax_fit_img))

        if verbose:
            # Dump additional details about the images and the measurement to a file
            pathlib.Path(self.report_path).mkdir(parents=True, exist_ok=True)
            detailed_output = {}
            detailed_outpath = os.path.join(self.report_path, f"{output_key}_details.json")

            metadata = dict(
                files=[im.filename for im in image_stack.images],
                plate=plate_number,
                relaxation_type=calc.upper(),
                institution_name=index_im.InstitutionName,
                manufacturer=index_im.Manufacturer,
                model=index_im.ManufacturerModelName,
                date=index_im.StudyDate,
                manufacturers_times=relax_published.tolist(),
                calc_times=image_stack.relax_times,
                frac_time_difference=frac_time_diff.tolist())
            # , output_graphics=output_files_path
            results['additional data'] = metadata

            detailed_output['measurement details'] = {
                'Echo Time': [im.EchoTime for im in image_stack.images],
                'Repetition Time': [im.RepetitionTime for im in image_stack.images],
                'Inversion Time': [im.InversionTime if hasattr(
                    im, 'InversionTime') else None for im in image_stack.images],
                # fit_paramters (T1) = [[T1, s0, A1] for each ROI]
                # fit_parameters (T2) = [[T2, s0, C] for each ROI]
                'ROI_means': {i: im.means for i, im in enumerate(image_stack.ROI_time_series)},
                'fit_parameters': [tuple(param[0].tolist()) for param in image_stack.relax_fit],
                'fit_equation': image_stack.fit_eqn_str
            }
            detailed_output['metadata'] = metadata
            json_object = json.dumps(detailed_output, indent = 4)
            with open(detailed_outpath, "w") as f:
                f.write(json_object)
            self.report_files.append(
                ('further_details', detailed_outpath))

        if self.report:
            results['report_image'] = self.report_files

        # plt.show()
        return results


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

    out_coords = cv.transform(np.array([in_coords]), rt_matrix)[0]  # reduce to two dimensions
    # out_coords = out_coords

    if output_row_col:
        out_coords = np.flip(out_coords, axis=1)

    return out_coords

def pixel_rescale(dcm):
    """
    Transforms pixel values according to scale values in DICOM header.
    
    DICOM pixel values arrays cannot directly represent signed or float values.
    This function converts the ``.pixel_array`` using the scaling values in the
    DICOM header.

    For Philips scanners the private DICOM fields 2005,100d (=SI) and 2005,100e
    (=SS) are used as inverse scaling factors to perform the inverse
    transformation [1]_.

    Parameters
    ----------
    dcm : Pydicom.dataset.FileDataset
        DICOM file containing one image.

    Returns
    -------
    numpy.array
        Values in ``dcm.pixel_array`` transformed using DICOM scaling.

    References
    ----------
    .. [1] Chenevert, Thomas L., et al. "Errors in quantitative image analysis
     due to platform-dependent image scaling." Translational Oncology 7.1
     (2014): 65-71. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3998685/

    """
    # Check for Philips
    if dcm.Manufacturer.startswith('Philips'):
        ss = dcm['2005100e'].value  # Scale slope
        si = dcm['2005100d'].value  # Scale intercept

        return (dcm.pixel_array - si) / ss
    else:
        return pydicom.pixel_data_handlers.util.apply_modality_lut(
            dcm.pixel_array, dcm)


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

    def __init__(self, dcm_images, poi_coords_row_col, time_attr):
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
        """

        self.POI_mask = np.zeros((dcm_images[0].pixel_array.shape[0],
                                  dcm_images[0].pixel_array.shape[1]),
                                 dtype=np.int8)
        self.POI_mask[poi_coords_row_col[0], poi_coords_row_col[1]] = 1


        kernel = skimage.morphology.square(5)
        """kernel
            Structuring element which defines ROI size and shape, centred on
            POI. Each element should be 1 or 0, otherwise calculation of mean
            will be incorrect. If ``None``, use a 5x5 square. The default is
            ``None``.
        """

        self.ROI_mask = np.zeros_like(self.POI_mask)
        self.ROI_mask = scipy.ndimage.filters.convolve(self.POI_mask, kernel)

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

    def __init__(self, image_slices, time_attribute=None):
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
            
        time_attribute : string, optional
            DICOM attribute to order images. Typically 'InversionTime' for T1
            relaxometry or 'EchoTime' for T2.
        """
        self.time_attr = time_attribute
        # store sorted images
        self.images = self.order_by(image_slices, time_attribute)

        b0_val = self.images[0]['MagneticFieldStrength'].value
        if b0_val not in [1.5, 3.0]:
            # TODO incorporate warning through e.g. logging module
            print('Unable to match B0 to default values. Using 1.5T.\n'
                  f" {self.images[0]['MagneticFieldStrength']}")
            self.b0_str = '1.5T'
        else:
            if b0_val == 3:
                self.b0_str = "3.0T"
            else:
                self.b0_str = f"{b0_val}T"

    def order_by(self, images, att):
        """Order images by attribute (e.g. EchoTime, InversionTime)."""
        sorted_images = sorted(images, key=lambda x: x[att].value.real)
        return sorted_images

    def template_fit(self, template_dcm, image_index=0):
        """
        Calculate transformation matrix to fit template to image.

        The template pixel array, template_px, is fitted to one of the
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
        # Store template pixel array, after scaling in 0028,1052 and
        # 0028,1053 applied
        template_px = pixel_rescale(template_dcm)
        target_px = pixel_rescale(self.images[0])

        ## Pad template or target pixels if required
        # Determine difference in shape
        pad_size = np.subtract(template_px.shape, target_px.shape)
        assert pad_size[0] == pad_size[1], "Image matrices must be square."
        if pad_size[0] > 0:  # pad target--UNTESTED
            # add pixels to target if smaller than template
            target_px = np.pad(target_px, pad_width=(0, pad_size[0]))
        elif pad_size[0] < 0:  # pad template
            # add pixels to template if smaller than target
            template_px = np.pad(template_px, pad_width=(0, -pad_size[0]))

        # Always fit on magnitude images for simplicity. May be suboptimal
        self.template8bit = cv.normalize(abs(template_px), None, 0, 255,
                            norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)

        self.target8bit = cv.normalize(abs(target_px), None, 0, 255,
                            norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)

        # initialise transformation fitting parameters.
        scale_factor = len(target_px) / len(template_px)
        scale_matrix = scale_factor * np.eye(2, 3, dtype=np.float32)

        self.scaled_template8bit = cv.warpAffine(
                    self.template8bit, scale_matrix,
                    (self.template8bit.shape[1],self.template8bit.shape[0]))

        # Apply transformation
        criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT,
                    TEMPLATE_FIT_ITERS, TERMINATION_EPS)
        # Find the geometric transform (warp) between two images in terms of the ECC criterion
        self.template_cc, warp_matrix = cv.findTransformECC(
                                        self.template8bit, self.target8bit,
                                        scale_matrix, criteria=criteria)

        self.warped_template8bit = cv.warpAffine(
                    self.template8bit, warp_matrix,
                    (self.template8bit.shape[1], self.template8bit.shape[0]))

        return warp_matrix

    def generate_time_series(self, coords_row_col, warp_matrix, fit_coords=True):
        """
        Create list of ROITimeSeries objects.

        Parameters
        ----------
        coords_row_col : array_like
            Array of coordinates points of interest (POIs) for each centre of
            each ROI. They should be in [[col0, row0], [col1, row1], ...]
            format.
        time_attribute : string, optional
            DICOM attribute to order images. Typically 'InversionTime' for T1
            relaxometry or 'EchoTime' for T2.
        warp_matrix : np.array
            RT transform matrix (2,3).
        """
        num_coords = np.size(coords_row_col, axis=0)

        # adjustment may not be required for the template DICOM
        if fit_coords:
            adjusted_coords_row_col = transform_coords(coords_row_col,
                    warp_matrix, input_row_col=True, output_row_col=True)
        else:  # used in testing
            adjusted_coords_row_col = coords_row_col

        self.ROI_time_series = []
        for i in range(num_coords):
            self.ROI_time_series.append(ROITimeSeries(
                self.images, adjusted_coords_row_col[i], self.time_attr))

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

    @property
    def relax_times(self):
        """List of T1 for each ROI."""
        return [fit[0][0] for fit in self.relax_fit]

class T1ImageStack(ImageStack):
    """
    Calculate T1 relaxometry.
    """

    def __init__(self, image_slices):
        time_attribute = "InversionTime"
        super().__init__(image_slices, time_attribute)

    def generate_t1_function(self, ti_interp_vals, tr_interp_vals, mag_image=False):
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
        #  Create piecewise linear fit function. k=1 gives linear, s=0 ensures all
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

    def generate_fit_function(self):
        """"Create T1 fit function for magnitude/signed image and variable TI."""
        #  check if image is signed or magnitude
        if np.all(pixel_rescale(self.images[0]) >= 0):
            mag_image = True
        else:
            mag_image = False
        self.fit_function, self.fit_jacobian, self.fit_eqn_str = \
            self.generate_t1_function(
                self.ROI_time_series[0].times, self.ROI_time_series[0].trs,
                mag_image=mag_image)

    def est_t1_s0(self, ti, tr, t1, pv):
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
        s0_est

        """
        self.t1_est = t1_estimates
        rois = self.ROI_time_series
        rois_first_mean = np.array([roi.means[0] for roi in rois])
        rois_last_mean = np.array([roi.means[-1] for roi in rois])
        s0_est_last = abs(self.est_t1_s0(rois[0].times[-1], rois[0].trs[-1],
                                    self.t1_est, rois_last_mean))
        s0_est_first = abs(self.est_t1_s0(rois[0].times[0], rois[0].trs[0],
                                     self.t1_est, rois_first_mean))
        s0_est = np.where(rois_first_mean > rois_last_mean,
                               s0_est_first, s0_est_last)
        self.a1_est = np.full_like(s0_est, 2.0)

        return s0_est

    def find_relax_times(self, t1_estimates, s0_est):
        """
        Calculate T1 values. Access as ``image_stack.relax_fits``

        Parameters
        ----------
        t1_estimates : array_like
            T1 values to seed estimation. These should be the manufacturer
            provided T1 values where known.

        Returns
        -------
        None.

        """
        rois = self.ROI_time_series
        self.relax_fit = [scipy.optimize.curve_fit(
            self.fit_function, rois[i].times, rois[i].means,
            p0=[t1_estimates[i], s0_est[i], self.a1_est[i]],
            jac=self.fit_jacobian, method='lm') for i in range(len(rois))]


class T2ImageStack(ImageStack):
    """
    Calculate T2 relaxometry.
    """

    def __init__(self, image_slices):
        time_attribute = "EchoTime"
        super().__init__(image_slices, time_attribute)

        self.fit_eqn_str = 'T2 with Rician noise (Raya et al 2010)'

    def generate_fit_function(self):
        """Null method in base class, may be overwritten in subclass."""

    def fit_function(self, te, t2, s0, c):
        """
        Calculated pixel value with Rician noise model.
        
        Calculates pixel value from [1]_::
            .. math::
                S=sqrt{frac{pi alpha^2}{2}} exp(- alpha) left( (1+ 2 alpha)
                text{I_0}(alpha) + 2 alpha text{I_1}(alpha) right)

                alpha() = left(frac{S_0}{2 sigma} exp{left(-frac{text{TE}}{text{T}_2}right)} right)^2

                text{I}_n() = n^text{th} text{order modified Bessel function of the first kind}

        Parameters
        ----------
        te : array_like
            Echo times.
        t2 : float
            T2 decay constant.
        S0 : float
            Initial pixel magnitude.
        C : float
            Noise parameter for Rician model (equivalent to st dev).

        Returns
        -------
        pv : array_like
            Theoretical pixel values (signal) at each TE.

        References
        ----------
        .. [1] Raya, J.G., Dietrich, O., Horng, A., Weber, J., Reiser, M.F. and
        Glaser, C., 2010. T2 measurement in articular cartilage: impact of the
        fitting method on accuracy and precision at low SNR. Magnetic Resonance in
        Medicine: An Official Journal of the International Society for Magnetic
        Resonance in Medicine, 63(1), pp.181-193.
        """

        alpha = (s0 / (2 * c) * np.exp(-te / t2)) **2
        # NB need to use `i0e` and `ive` below to avoid numeric inaccuracy from
        # multiplying by huge exponentials then dividing by the same exponential
        pv = np.sqrt(np.pi/2 * c ** 2) *  \
            ((1 + 2 * alpha) * i0e(alpha) + 2 * alpha * ive(1, alpha))

        return pv

    def est_t2_s0(self, te, t2, pv, c=0.0):
        """
        Initial guess for s0 to seed curve fitting::
            .. math::
                S_0=\\frac{pv-c}{exp(-TE/T_2)}


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
            Initial s0 estimate.

        """
        return (pv - c) / np.exp(-te / t2)

    def initialise_fit_parameters(self, t2_estimates):
        """
        Estimate fit parameters (t2, s0, c) for T2 curve fitting.
        
        T2 estimates are provided.
        
        s0 is estimated using est_t2_s0(te, t2_est, mean_pv, c).
            
        C is estimated as 5.0.

        Parameters
        ----------
        t2_estimates : array_like
            T2 values to seed estimation. These should be the manufacturer
            provided T2 values where known.

        Returns
        -------
        s0_est.

        """
        self.t2_est = t2_estimates
        rois = self.ROI_time_series
        rois_second_mean = np.array([roi.means[1] for roi in rois])
        self.c_est = np.full_like(self.t2_est, SEED_RICIAN_NOISE)
        # estimate s0 from second image--first image is too low.
        s0_est = self.est_t2_s0(rois[0].times[1], self.t2_est,
                                rois_second_mean, self.c_est)

        return s0_est

    def find_relax_times(self, t2_estimates, s0_est):
        """
        Calculate T2 values. Access as ``image_stack.relax_times``
        
        Uses the 'skip first echo' fit method [1]_ with a Rician noise model
        [2]_. Ideally the Rician noise parameter should be determined from the
        images rather than fitted. However, this is not possible as the noise
        profile varies across the image due to spatial coil sensitivities and
        whether the image is normalised or unfiltered. Fitting the noise
        parameter makes this easier. It has an upper limit of MAX_RICIAN_NOISE,
        currently set to 20.0.

        Parameters
        ----------
        t2_estimates : array_like
            T2 values to seed estimation. These should be the manufacturer
            provided T2 values where known.

        Returns
        -------
        None.
        
        References
        ----------
        .. [1] McPhee, K. C., & Wilman, A. H. (2018). Limitations of skipping 
        echoes for exponential T2 fitting. Journal of Magnetic Resonance 
        Imaging, 48(5), 1432-1440. https://doi.org/10.1002/jmri.26052

        .. [2] Raya, J.G., Dietrich, O., Horng, A., Weber, J., Reiser, M.F. and
        Glaser, C., 2010. T2 measurement in articular cartilage: impact of the
        fitting method on accuracy and precision at low SNR. Magnetic Resonance
        in Medicine: An Official Journal of the International Society for
         Magnetic Resonance in Medicine, 63(1), pp.181-193.
        """

        rois = self.ROI_time_series
        #  Omit the first image data from the curve fit. This is achieved by
        #  slicing rois[i].times[1:] and rois[i].means[1:].
        #  Skipping odd echoes can be implemented with
        #  rois[i].times[1::2] and .means[1::2]

        bounds = ([0, 0, 1], [np.inf, np.inf, MAX_RICIAN_NOISE])

        self.relax_fit = [scipy.optimize.curve_fit(
            self.fit_function, rois[i].times[1:], rois[i].means[1:],
            p0=[t2_estimates[i], s0_est[i], self.c_est[i]],
            jac=None, bounds=bounds, method='trf') for i in range(len(rois))]
