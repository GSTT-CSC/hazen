"""
Assumptions:
Square voxels, no multi-frame support
"""

from math import pi
import sys
import traceback
from copy import copy
from hazenlib.logger import logger

import numpy as np
from scipy import ndimage
from scipy.interpolate import interp1d

import hazenlib

class Rod:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f'Rod: {self.x}, {self.y}'

    def __str__(self):
        return f'Rod: {self.x}, {self.y}'

    @property
    def centroid(self):
        return self.x, self.y

    def __lt__(self, other):
        """Using "reading order" in a coordinate system where 0,0 is bottom left"""
        try:
            x0, y0 = self.centroid
            x1, y1 = other.centroid
            return (-y0, x0) < (-y1, x1)
        except AttributeError:
            return NotImplemented

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y


def sort_rods(rods):

    lower_row = sorted(rods, key=lambda rod: rod.y)[-3:]
    lower_row = sorted(lower_row, key=lambda rod: rod.x)
    middle_row = sorted(rods, key=lambda rod: rod.y)[3:6]
    middle_row = sorted(middle_row, key=lambda rod: rod.x)
    upper_row = sorted(rods, key=lambda rod: rod.y)[0:3]
    upper_row = sorted(upper_row, key=lambda rod: rod.x)
    return lower_row + middle_row + upper_row


def get_rods(dcm):
    """
    Parameters
    ----------
    dcm : array_like
        input DICOM file
    Returns
    -------
    rods : array_like
        rod positions in pixels

    Notes
    -------
    The rod indices are ordered as:
        789
        456
        123
    """

    arr = dcm.pixel_array

    # threshold and binaries the image in order to locate the rods.
    # this is achieved by masking the
    img_max = np.max(arr)  # maximum number of img intensity
    no_region = [None] * img_max

    # smooth the image with a 0.5sig kernal - this is to avoid noise being counted in .label function
    # img_tmp = ndimage.gaussian_filter(arr, 0.5)
    # commented out smoothing as not in original MATLAB - Haris
    img_tmp = arr
    # step over a range of threshold levels from 0 to the max in the image
    # using the ndimage.label function to count the features for each threshold
    for x in range(0, img_max):
        tmp = img_tmp <= x
        labeled_array, num_features = ndimage.label(tmp.astype(np.int))
        no_region[x] = num_features

    # find the indices that correspond to 10 regions and pick the median
    index = [i for i, val in enumerate(no_region) if val == 10]

    thres_ind = np.median(index).astype(np.int)

    # Generate the labeled array with the threshold chosen
    img_threshold = img_tmp <= thres_ind

    labeled_array, num_features = ndimage.label(img_threshold.astype(np.int))

    # check that we have got the 10 rods!
    if num_features != 10:
        sys.exit("Did not find the 9 rods")

    rods = ndimage.measurements.center_of_mass(arr, labeled_array, range(2, 11))

    rods = [Rod(x=x[1], y=x[0]) for x in rods]
    rods = sort_rods(rods)


    return rods


def plot_rods(ax, arr, rods): # pragma: no cover
    ax.imshow(arr, cmap='gray')
    mark = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
    for idx, i in enumerate(rods):
        ax.scatter(x=i.x, y=i.y, marker=f"${mark[idx]}$", s=10, linewidths=0.4)

    ax.set_title('find rods')
    return ax


def get_rod_distances(rods):
    """
    Calculates horizontal and vertical distances between adjacent rods in pixels

    Parameters
    ----------
    rods : array_like
        rod positions in pixels

    Returns
    -------
    horz_dist, vert_dist : array_like
        horizontal and vertical distances between rods in pixels

    """

    horz_dist = [None] * 3
    vert_dist = [None] * 3
    horz_dist[0] = round((((rods[2].y - rods[0].y) ** 2) + (rods[2].x - rods[0].x) ** 2) ** 0.5, 3)
    horz_dist[1] = round((((rods[5].y - rods[3].y) ** 2) + (rods[5].x - rods[3].x) ** 2) ** 0.5, 3)
    horz_dist[2] = round((((rods[8].y - rods[6].y) ** 2) + (rods[8].x - rods[6].x) ** 2) ** 0.5, 3)

    vert_dist[2] = round((((rods[2].y - rods[8].y) ** 2) + (rods[2].x - rods[8].x) ** 2) ** 0.5, 3)
    vert_dist[1] = round((((rods[1].y - rods[7].y) ** 2) + (rods[1].x - rods[7].x) ** 2) ** 0.5, 3)
    vert_dist[0] = round((((rods[0].y - rods[6].y) ** 2) + (rods[0].x - rods[6].x) ** 2) ** 0.5, 3)

    return horz_dist, vert_dist


def get_rod_distortion_correction_coefficients(horizontal_distances, pixel_size) -> dict:
    """
    Removes the effect of geometric distortion from the slice width measurement. Assumes that rod separation is
    120 mm.

    Parameters
    ----------
    horizontal_distances : list
        horizontal distances between rods, in pixels

    pixel_size : float
        pixel size as defined in DICOM header

    Returns
    -------
    coefficients : dict
        dictionary containing top and bottom distortion corrections, in mm
    """


    coefficients = {"top": round(np.mean(horizontal_distances[1:3])*pixel_size / 120, 4),
                    "bottom": round(np.mean(horizontal_distances[0:2])*pixel_size / 120, 4)}

    return coefficients


def get_rod_distortions(rods, dcm):

    """

    Parameters
    ----------
    rods
    dcm

    Returns
    -------
    horz_distortion, vert_distortion : float
        horizontal and vertical distortion values, in mm
    """


    pixel_spacing = dcm.PixelSpacing[0]
    horz_dist, vert_dist = get_rod_distances(rods)

    #calculate the horizontal and vertical distances

    horz_dist_mm = np.multiply(pixel_spacing, horz_dist)
    vert_dist_mm = np.multiply(pixel_spacing, vert_dist)

    horz_distortion = 100 * np.std(horz_dist_mm, ddof=1) / np.mean(horz_dist_mm) # ddof to match MATLAB std
    vert_distortion = 100 * np.std(vert_dist_mm, ddof=1) / np.mean(vert_dist_mm)
    return horz_distortion, vert_distortion


def baseline_correction(profile, sample_spacing):

    """
    Calculates quadratic fit of the baseline and subtracts from profile

    Parameters
    ----------
    profile
    sample_spacing

    Returns
    -------

    """

    profile_width = len(profile)
    padding = 30
    outer_profile = np.concatenate([profile[0:padding], profile[-padding:]])
    # create the x axis for the outer profile
    x_left = np.arange(padding)
    x_right = np.arange(profile_width - padding, profile_width)
    x_outer = np.concatenate([x_left, x_right])

    # seconds order poly fit of the outer profile
    polynomial_coefficients = np.polyfit(x_outer, outer_profile, 2)
    polynomial_fit = np.poly1d(polynomial_coefficients)

    # use the poly fit to generate a quadratic curve with 0.25 space (high res)
    x_interp = np.arange(0, profile_width, sample_spacing)
    x = np.arange(0, profile_width)

    baseline_interp = polynomial_fit(x_interp)
    baseline = polynomial_fit(x)

    # Remove the baseline effects from the profiles
    profile_corrected = profile - baseline
    f = interp1d(x, profile_corrected, fill_value="extrapolate")
    profile_corrected_interp = f(x_interp)
    profile_interp = profile_corrected_interp + baseline_interp

    return {"f": polynomial_coefficients,
            "x_interpolated": x_interp,
            "baseline_fit": polynomial_fit,
            "baseline": baseline,
            "baseline_interpolated": baseline_interp,
            "profile_interpolated": profile_interp,
            "profile_corrected_interpolated": profile_corrected_interp}


def gauss_2D(xy_tuple, A, x_0, y_0, sigma_x, sigma_y, theta, C):
    """
    Create 2D Gaussian

    Parameters
    ----------
    xy_tuple : grid of x-y coordinates
    A : amplitude of 2D Gaussian
    x_0 / y_0 : centre of 2D Gaussian
    sigma_x / sigma_y : widths of 2D Gaussian
    theta : rotation of Gaussian
    C : background/intercept of 2D Gaussian

    Returns
    -------
    gauss : 1-D list of Gaussian intensities

    """
    (x, y) = xy_tuple
    x_0 = float(x_0)
    y_0 = float(y_0)

    cos_theta_2 = np.cos(theta) ** 2
    sin_theta_2 = np.sin(theta) ** 2
    cos_2_theta = np.cos(2 * theta)
    sin_2_theta = np.sin(2 * theta)

    sigma_x_2 = sigma_x ** 2
    sigma_y_2 = sigma_y ** 2

    a = cos_theta_2 / (2 * sigma_x_2) + sin_theta_2 / (2 * sigma_y_2)
    b = -sin_2_theta / (4 * sigma_x_2) + sin_2_theta / (4 * sigma_y_2)
    c = sin_theta_2 / (2 * sigma_x_2) + cos_theta_2 / (2 * sigma_y_2)

    gauss = A * np.exp(-(a * (x - x_0) ** 2 + 2 * b * (x - x_0) * (y - y_0) + c * (y - y_0) ** 2)) + C

    return gauss.ravel()


def trapezoid(n_ramp, n_plateau, n_left_baseline, n_right_baseline, plateau_amplitude):

    """

    Parameters
    ----------
    n_ramp
    n_plateau
    n_left_baseline
    n_right_baseline
    plateau_amplitude

    Returns
    -------
    """

    if n_left_baseline < 1:
        left_baseline = []
    else:
        left_baseline = np.zeros(n_left_baseline)

    if n_ramp < 1:
        left_ramp = []
        right_ramp = []
    else:
        left_ramp = np.linspace(0, plateau_amplitude, n_ramp)
        right_ramp = np.linspace(plateau_amplitude, 0, n_ramp)

    if n_plateau < 1:
        plateau = []
    else:
        plateau = plateau_amplitude * np.ones(n_plateau)

    if n_right_baseline < 1:
        right_baseline = []
    else:
        right_baseline = np.zeros(n_right_baseline)

        trap = np.concatenate([left_baseline, left_ramp, plateau, right_ramp, right_baseline])
        fwhm = n_plateau + n_ramp

    return trap, fwhm


def get_ramp_profiles(image_array, rods, pixel_size) -> dict:
    """
    Find the central y-axis point for the top and bottom profiles
    done by finding the distance between the mid-distances of the central rods

    Parameters
    ----------
    image_array
    rods
    pixel_size

    Returns
    -------


    """


    top_profile_vertical_centre = np.round(((rods[3].y - rods[6].y) / 2) + rods[6].y).astype(int)
    bottom_profile_vertical_centre = np.round(((rods[0].y - rods[3].y) / 2) + rods[3].y).astype(int)

    # Selected 20mm around the mid-distances and take the average to find the line profiles
    top_profile = image_array[
                  (top_profile_vertical_centre - round(10/pixel_size)):(top_profile_vertical_centre + round(10/pixel_size)),
                  int(rods[3].x):int(rods[5].x)]

    bottom_profile = image_array[
                     (bottom_profile_vertical_centre - round(10/pixel_size)):(bottom_profile_vertical_centre + round(10/pixel_size)),
                     int(rods[3].x):int(rods[5].x)]

    return {"top": top_profile, "bottom": bottom_profile,
            "top-centre": top_profile_vertical_centre, "bottom-centre": bottom_profile_vertical_centre}


def get_initial_trapezoid_fit_and_coefficients(profile, slice_thickness):
    """

    Parameters
    ----------
    profile
    slice_thickness

    Returns
    -------
    trapezoid_fit_initial
    trapezoid_fit_coefficients

    """

    n_plateau, n_ramp = None, None

    if slice_thickness == 3:
        # not sure where these magic numbers are from, I subtracted 1 from MATLAB numbers
        n_ramp = 7
        n_plateau = 32

    elif slice_thickness == 5:
        # not sure where these magic numbers are from, I subtracted 1 from MATLAB numbers
        n_ramp = 47
        n_plateau = 55

    trapezoid_centre = int(round(np.median(np.argwhere(profile < np.mean(profile)))))

    n_total = len(profile)
    n_left_baseline = int(trapezoid_centre - round(n_plateau / 2) - n_ramp - 1)
    n_right_baseline = n_total - n_left_baseline - 2 * n_ramp - n_plateau
    plateau_amplitude = np.percentile(profile, 5) - np.percentile(profile, 95)
    trapezoid_fit_coefficients = [n_ramp, n_plateau, n_left_baseline, n_right_baseline, plateau_amplitude]

    trapezoid_fit_initial, _ = trapezoid(n_ramp, n_plateau, n_left_baseline, n_right_baseline, plateau_amplitude)

    return trapezoid_fit_initial, trapezoid_fit_coefficients


def fit_trapezoid(profiles, slice_thickness):

    """

    Parameters
    ----------
    profiles
    slice_thickness

    Returns
    -------
    trapezoid_fit_coefficients
    baseline_fit_coefficients

    """
    trapezoid_fit, trapezoid_fit_coefficients = get_initial_trapezoid_fit_and_coefficients(
        profiles["profile_corrected_interpolated"], slice_thickness)

    x_interp = profiles["x_interpolated"]
    profile_interp = profiles["profile_interpolated"]
    baseline_interpolated = profiles["baseline_fit"](x_interp)
    baseline_fit_coefficients = profiles["baseline_fit"]
    baseline_fit_coefficients = [baseline_fit_coefficients.c[0], baseline_fit_coefficients.c[1], baseline_fit_coefficients.c[2]]
    # sum squared differences
    current_error = sum((profiles["profile_corrected_interpolated"] - (baseline_interpolated + trapezoid_fit)) ** 2)

    def get_error(base, trap):
        """ Check if fit is improving """
        trapezoid_fit_temp,_ = trapezoid(*trap)

        baseline_fit_temp = np.poly1d(base)(x_interp)

        sum_squared_difference = sum((profile_interp - (baseline_fit_temp + trapezoid_fit_temp)) ** 2)

        return sum_squared_difference

    cont = 1
    j = 0
    """Go through a series of changes to reduce error, if error doesnt reduced in one entire loop then exit"""
    while cont == 1:
        j += 1
        cont = 0

        for i in range(14):
            baseline_fit_coefficients_temp = copy(baseline_fit_coefficients)
            trapezoid_fit_coefficients_temp = copy(trapezoid_fit_coefficients)

            if i == 0:
                baseline_fit_coefficients_temp[0] = baseline_fit_coefficients_temp[0] - 0.0001

            elif i == 1:
                baseline_fit_coefficients_temp[0] = baseline_fit_coefficients_temp[0] + 0.0001
            elif i == 2:
                baseline_fit_coefficients_temp[1] = baseline_fit_coefficients_temp[1] - 0.001
            elif i == 3:
                baseline_fit_coefficients_temp[1] = baseline_fit_coefficients_temp[1] + 0.001
            elif i == 4:
                baseline_fit_coefficients_temp[2] = baseline_fit_coefficients_temp[2] - 0.1
            elif i == 5:
                baseline_fit_coefficients_temp[2] = baseline_fit_coefficients_temp[2] + 0.1
            elif i == 6:  # Decrease the ramp width
                trapezoid_fit_coefficients_temp[0] = trapezoid_fit_coefficients_temp[0] - 1
                trapezoid_fit_coefficients_temp[2] = trapezoid_fit_coefficients_temp[2] + 1
                trapezoid_fit_coefficients_temp[3] = trapezoid_fit_coefficients_temp[3] + 1
            elif i == 7:  # Increase the ramp width
                trapezoid_fit_coefficients_temp[0] = trapezoid_fit_coefficients_temp[0] + 1
                trapezoid_fit_coefficients_temp[2] = trapezoid_fit_coefficients_temp[2] - 1
                trapezoid_fit_coefficients_temp[3] = trapezoid_fit_coefficients_temp[3] - 1
            elif i == 8:  # Decrease plateau width
                trapezoid_fit_coefficients_temp[1] = trapezoid_fit_coefficients_temp[1] - 2
                trapezoid_fit_coefficients_temp[2] = trapezoid_fit_coefficients_temp[2] + 1
                trapezoid_fit_coefficients_temp[3] = trapezoid_fit_coefficients_temp[3] + 1

            elif i == 9:  # Increase plateau width
                trapezoid_fit_coefficients_temp[1] = trapezoid_fit_coefficients_temp[1] + 2
                trapezoid_fit_coefficients_temp[2] = trapezoid_fit_coefficients_temp[2] - 1
                trapezoid_fit_coefficients_temp[3] = trapezoid_fit_coefficients_temp[3] - 1

            elif i == 10:  # Shift centre to the left
                trapezoid_fit_coefficients_temp[2] = trapezoid_fit_coefficients_temp[2] - 1
                trapezoid_fit_coefficients_temp[3] = trapezoid_fit_coefficients_temp[3] + 1

            elif i == 11:  # Shift centre to the right
                trapezoid_fit_coefficients_temp[2] = trapezoid_fit_coefficients_temp[2] + 1
                trapezoid_fit_coefficients_temp[3] = trapezoid_fit_coefficients_temp[3] - 1

            elif i == 12:  # Reduce amplitude
                trapezoid_fit_coefficients_temp[4] = trapezoid_fit_coefficients_temp[4] - 0.1

            elif i == 13:  # Increase amplitude
                trapezoid_fit_coefficients_temp[4] = trapezoid_fit_coefficients_temp[4] + 0.1

            new_error = get_error(base=baseline_fit_coefficients_temp, trap=trapezoid_fit_coefficients_temp)

            if new_error < current_error:
                cont = 1
                if i > 6:
                    trapezoid_fit_coefficients = trapezoid_fit_coefficients_temp
                else:
                    baseline_fit_coefficients = baseline_fit_coefficients_temp
                current_error = new_error

    return trapezoid_fit_coefficients, baseline_fit_coefficients


def get_slice_width(dcm, report_path=False):
    """
    Calculates slice width using double wedge image

    Parameters
    ----------
    dcm
    report_path

    Returns
    -------
    slice_width_mm : dict
        calculated slice width (top, bottom, combined; various methods) in mm

    horizontal_linearity_mm, vertical_linearity_mm : float
        calculated average rod distance in mm

    horz_distortion_mm, vert_distortion_mm : float
        calculated rod distance distortion in mm

    """
    slice_width_mm = {"top": {}, "bottom": {}, "combined": {}}
    arr = dcm.pixel_array
    sample_spacing = 0.25
    pixel_size = dcm.PixelSpacing[0]

    rods = get_rods(dcm)
    horz_distances, vert_distances = get_rod_distances(rods)
    horz_distortion_mm, vert_distortion_mm = get_rod_distortions(rods, dcm)
    correction_coefficients_mm = get_rod_distortion_correction_coefficients(horizontal_distances=horz_distances, pixel_size=pixel_size)

    ramp_profiles = get_ramp_profiles(arr, rods, pixel_size)
    ramp_profiles_baseline_corrected = {"top": baseline_correction(np.mean(ramp_profiles["top"], axis=0),
                                                                   sample_spacing),
                                        "bottom": baseline_correction(np.mean(ramp_profiles["bottom"], axis=0),
                                                                      sample_spacing)}

    trapezoid_coefficients, baseline_coefficients = fit_trapezoid(ramp_profiles_baseline_corrected["top"],
                                                                  dcm.SliceThickness)
    top_trap, fwhm = trapezoid(*trapezoid_coefficients)

    slice_width_mm["top"]["default"] = fwhm * sample_spacing * pixel_size * np.tan((11.3*pi)/180)
    # Factor of 4 because interpolated by factor of four

    slice_width_mm["top"]["geometry_corrected"] = slice_width_mm["top"]["default"]/correction_coefficients_mm["top"]

    # AAPM method directly incorporating phantom tilt
    slice_width_mm["top"]["aapm"] = fwhm * sample_spacing * pixel_size

    # AAPM method directly incorporating phantom tilt and independent of geometric linearity
    slice_width_mm["top"]["aapm_corrected"] = (fwhm * sample_spacing * pixel_size) / correction_coefficients_mm["top"]

    trapezoid_coefficients, baseline_coefficients = fit_trapezoid(ramp_profiles_baseline_corrected["bottom"], dcm.SliceThickness)
    bottom_trap, fwhm = trapezoid(*trapezoid_coefficients)

    slice_width_mm["bottom"]["default"] = fwhm * sample_spacing * pixel_size * np.tan((11.3 * pi) / 180)
    # Factor of 4 because interpolated by factor of four

    slice_width_mm["bottom"]["geometry_corrected"] = slice_width_mm["bottom"]["default"] / correction_coefficients_mm["bottom"]

    # AAPM method directly incorporating phantom tilt
    slice_width_mm["bottom"]["aapm"] = fwhm * sample_spacing * pixel_size

    # AAPM method directly incorporating phantom tilt and independent of geometric linearity
    slice_width_mm["bottom"]["aapm_corrected"] = (fwhm * sample_spacing * pixel_size) / correction_coefficients_mm["bottom"]

    # Geometric mean of slice widths (pg 34 of IPEM Report 80)
    slice_width_mm["combined"]["default"] = (slice_width_mm["top"]["default"] * slice_width_mm["bottom"]["default"]) ** 0.5
    slice_width_mm["combined"]["geometry_corrected"] = (slice_width_mm["top"]["geometry_corrected"] * slice_width_mm["bottom"]["geometry_corrected"]) ** 0.5

    # AAPM method directly incorporating phantom tilt
    theta = (180.0 - 2.0 * 11.3) * pi / 180.0
    term1 = (np.cos(theta)) ** 2.0 * (slice_width_mm["bottom"]["aapm"] - slice_width_mm["top"]["aapm"])**2.0 + (4.0 * slice_width_mm["bottom"]["aapm"] * slice_width_mm["top"]["aapm"])
    term2 = (slice_width_mm["bottom"]["aapm"] + slice_width_mm["top"]["aapm"]) * np.cos(theta)
    term3 = 2.0 * np.sin(theta)

    slice_width_mm["combined"]["aapm_tilt"] = (term1**0.5 + term2)/term3
    phantom_tilt = np.arctan(slice_width_mm["combined"]["aapm_tilt"]/slice_width_mm["bottom"]["aapm"]) + (theta/2.0) - pi/2.0
    phantom_tilt_deg = phantom_tilt * (180.0/pi)

    phantom_tilt_check = -np.arctan(slice_width_mm["combined"]["aapm_tilt"]/slice_width_mm["top"]["aapm"]) - (theta/2.0) + pi/2.0
    phantom_tilt_check_deg = phantom_tilt_check * (180.0/pi)

    # AAPM method directly incorporating phantom tilt and independent of geometric linearity
    theta = (180.0 - 2.0 * 11.3) * pi/180.0
    term1 = (np.cos(theta)) ** 2.0 * (slice_width_mm["bottom"]["aapm_corrected"] - slice_width_mm["top"]["aapm_corrected"])**2.0 + (4.0 * slice_width_mm["bottom"]["aapm_corrected"] * slice_width_mm["top"]["aapm_corrected"])
    term2 = (slice_width_mm["bottom"]["aapm_corrected"] + slice_width_mm["top"]["aapm_corrected"]) * np.cos(theta)
    term3 = 2.0 * np.sin(theta)

    slice_width_mm["combined"]["aapm_tilt_corrected"] = (term1 ** 0.5 + term2) / term3
    phantom_tilt = np.arctan(slice_width_mm["combined"]["aapm_tilt_corrected"] / slice_width_mm["bottom"]["aapm_corrected"]) + (theta / 2.0) - pi / 2.0
    phantom_tilt_deg = phantom_tilt * (180.0 / pi)

    phantom_tilt_check = -np.arctan(slice_width_mm["combined"]["aapm_tilt_corrected"] / slice_width_mm["top"]["aapm_corrected"]) - (
                theta / 2.0) + pi / 2.0
    phantom_tilt_check_deg = phantom_tilt_check * (180.0 / pi)

    # calculate linearity in mm from distances in pixels

    horizontal_linearity_mm = np.mean(horz_distances) * pixel_size
    vertical_linearity_mm = np.mean(vert_distances) * pixel_size

    # calculate horizontal and vertical distances in mm from distances in pixels, for output

    horz_distances_mm=[x*pixel_size for x in horz_distances]

    vert_distances_mm = [x*pixel_size for x in vert_distances]


    if report_path:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(6, 1)
        fig.set_size_inches(6, 16)
        fig.tight_layout(pad=1)

        plot_rods(axes[0], arr, rods)

        axes[1].plot(np.mean(ramp_profiles["top"], axis=0), label='mean top profile')
        axes[1].plot(ramp_profiles_baseline_corrected["top"]["baseline"],
                     label='top profile baseline (interpolated)')
        axes[1].legend()

        axes[2].plot(ramp_profiles_baseline_corrected["top"]["profile_corrected_interpolated"],
                     label='corrected top profile')
        axes[2].plot(top_trap, label='trapezoid fit')
        axes[2].legend()

        axes[3].plot(np.mean(ramp_profiles["bottom"], axis=0), label='mean bottom profile')
        axes[3].plot(ramp_profiles_baseline_corrected["bottom"]["baseline"],
                     label='bottom profile baseline (interpolated')
        axes[3].legend()

        axes[4].plot(ramp_profiles_baseline_corrected["bottom"]["profile_corrected_interpolated"],
                     label='corrected bottom profile')
        axes[4].plot(bottom_trap, label='trapezoid fit')
        axes[4].legend()
        axes[5].axis('off')
        axes[5].table(
            cellText=[[str(x) for x in horz_distances_mm]+[str(np.around(horizontal_linearity_mm, 3))],
                      [str(x) for x in vert_distances_mm]+[str(np.around(vertical_linearity_mm, 3))]],
            rowLabels=['H-distances (S->I)',
                       'V-distances (R->L)'],
            colLabels=['1', '2', '3', 'mean/linearity'],
            colWidths=[0.15]*(len(horz_distances) + 1),  # plus one for linearity,
            rowLoc="center",
            loc="center"
        )

        fig.savefig(report_path + '.png')

    # print(f"Series Description: {dcm.SeriesDescription}\nWidth: {dcm.Rows}\nHeight: {dcm.Columns}\nSlice Thickness(
    # mm):" f"{dcm.SliceThickness}\nField of View (mm): {hazenlib.get_field_of_view(dcm)}\nbandwidth (Hz/Px) : {
    # dcm.PixelBandwidth}\n" f"TR  (ms) : {dcm.RepetitionTime}\nTE  (ms) : {dcm.EchoTime}\nFlip Angle  (deg) : {
    # dcm.FlipAngle}\n" f"Horizontal line bottom (mm): {horz_distances[0]}\nHorizontal line middle (mm): {
    # horz_distances[2]}\n" f"Horizontal line top (mm): {horz_distances[2]}\nHorizontal Linearity (mm): {np.mean(
    # horz_distances)}\n" f"Horizontal Distortion: {horz_distortion}\nVertical line left (mm): {vert_distances[0]}\n"
    # f"Vertical line middle (mm): {vert_distances[1]}\nVertical line right (mm): {vert_distances[2]}\n" f"Vertical
    # Linearity (mm): {np.mean(vert_distances)}\nVertical Distortion: {vert_distortion}\n" f"Slice width top (mm): {
    # slice_width['top']['default']}\n" f"Slice width bottom (mm): {slice_width['bottom']['default']}\nPhantom tilt (
    # deg): {phantom_tilt_deg}\n" f"Slice width AAPM geometry corrected (mm): {slice_width['combined'][
    # 'aapm_tilt_corrected']}")

    return {'slice_width_mm': slice_width_mm['combined']['aapm_tilt_corrected'],
            'vertical_distortion_mm': vert_distortion_mm, 'horizontal_distortion_mm': horz_distortion_mm,
            'vertical_linearity_mm': vertical_linearity_mm, 'horizontal_linearity_mm': horizontal_linearity_mm,
            'horizontal_distances_mm': horz_distances_mm, 'vertical_distances_mm': vert_distances_mm }

def main(data: list, report_path=False) -> dict:
    """

    Parameters
    ----------
    data : list
    report_path : bool

    Returns
    -------

    """
    results = {}
    for dcm in data:
        try:
            key = f"{dcm.SeriesDescription}_{dcm.SeriesNumber}_{dcm.InstanceNumber}"
            if report_path:
                report_path = key
        except AttributeError as e:
            logger.info(e)
            key = f"{dcm.SeriesDescription}_{dcm.SeriesNumber}"
        try:
            result = get_slice_width(dcm, report_path)
        except Exception as e:
            print(f"Could not calculate the slice_width for {key} because of : {e}")
            traceback.print_exc(file=sys.stdout)
            continue

        results[key] = result

    return results
