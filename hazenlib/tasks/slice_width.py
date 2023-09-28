"""
Assumptions:
Square voxels, no multi-frame support
"""

import sys
import os
import traceback
from copy import copy
from copy import deepcopy
from math import pi

import numpy as np
import scipy.optimize as opt
from matplotlib import pyplot as plt
from scipy import ndimage
from scipy.interpolate import interp1d
from skimage.measure import regionprops

from hazenlib.HazenTask import HazenTask
from hazenlib.utils import Rod


class SliceWidth(HazenTask):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.single_dcm = self.dcm_list[0]
        self.pixel_size = self.single_dcm.PixelSpacing[0]

    def run(self):
        results = self.init_result_dict()
        results['file'] = self.img_desc(self.single_dcm)
        try:
            results['measurement'] = self.get_slice_width(self.single_dcm)
        except Exception as e:
            print(f"Could not calculate the slice_width for {self.img_desc(self.single_dcm)} because of : {e}")
            traceback.print_exc(file=sys.stdout)

        # only return reports if requested
        if self.report:
            results['report_image'] = self.report_files

        return results

    def sort_rods(self, rods):

        lower_row = sorted(rods, key=lambda rod: rod.y)[-3:]
        lower_row = sorted(lower_row, key=lambda rod: rod.x)
        middle_row = sorted(rods, key=lambda rod: rod.y)[3:6]
        middle_row = sorted(middle_row, key=lambda rod: rod.x)
        upper_row = sorted(rods, key=lambda rod: rod.y)[0:3]
        upper_row = sorted(upper_row, key=lambda rod: rod.x)
        return lower_row + middle_row + upper_row

    def get_rods(self, arr):
        """
        Parameters
        ----------
        arr : DICOM pixel array
        Returns
        -------
        rods : array_like – centroid coordinates of rods
        rods_initial : array_like  – initial guess at rods (center-of mass)

        Notes
        -------
        The rod indices are ordered as:
            789
            456
            123
        """

        # inverted image for fitting (maximisation)
        arr_inv = np.invert(arr)
        if np.min(arr_inv) < 0:
            arr_inv = arr_inv + abs(np.min(arr_inv))  # ensure voxel values positive for maximisation

        """
        Initial Center-of-mass Rod Locator
        """

        # threshold and binaries the image in order to locate the rods.
        img_max = np.max(arr)  # maximum number of img intensity
        no_region = [None] * img_max

        img_tmp = arr
        # step over a range of threshold levels from 0 to the max in the image
        # using the ndimage.label function to count the features for each threshold
        for x in range(0, img_max):
            tmp = img_tmp <= x
            labeled_array, num_features = ndimage.label(tmp.astype(int))
            no_region[x] = num_features

        # find the indices that correspond to 10 regions and pick the median
        index = [i for i, val in enumerate(no_region) if val == 10]

        thres_ind = np.median(index).astype(int)

        # Generate the labeled array with the threshold chosen
        img_threshold = img_tmp <= thres_ind

        labeled_array, num_features = ndimage.label(img_threshold.astype(int))

        # check that we have got the 10 rods!
        if num_features != 10:
            sys.exit("Did not find the 9 rods")

        rods = ndimage.measurements.center_of_mass(arr, labeled_array, range(2, 11))

        rods = [Rod(x=x[1], y=x[0]) for x in rods]
        rods = self.sort_rods(rods)
        rods_initial = deepcopy(rods)  # save for later

        """
        Gaussian 2D Rod Locator
        """

        # setup bounding box dict
        bbox = {"x_start": [], "x_end": [], "y_start": [], "y_end": [], "intensity_max": [], "rod_dia": [],
                "radius": []}

        # get relevant label properties
        rod_radius = []
        rod_inv_intensity = []

        rprops = regionprops(labeled_array, arr_inv)[1:]  # ignore first label
        for idx, i in enumerate(rprops):
            rod_radius.append(rprops[idx].feret_diameter_max)  # 'radius' of each label
            rod_inv_intensity.append(rprops[idx].intensity_max)

        rod_radius_mean = int(np.mean(rod_radius))
        rod_inv_intensity_mean = int(np.mean(rod_inv_intensity))
        bbox["radius"] = int(np.ceil((rod_radius_mean * 2) / 2))

        # array bounding box regions around rods
        ext = bbox["radius"]  # no. pixels to extend bounding box

        for idx, i in enumerate(rprops):
            bbox["x_start"].append(rprops[idx].bbox[0] - ext)
            bbox["x_end"].append(rprops[idx].bbox[2] + ext)
            bbox["y_start"].append(rprops[idx].bbox[1] - ext)
            bbox["y_end"].append(rprops[idx].bbox[3] + ext)
            bbox["intensity_max"].append(rprops[idx].intensity_max)
            bbox["rod_dia"].append(rprops[idx].feret_diameter_max)

            # print(f'Rod {idx} – Bounding Box, x: ({bbox["x_start"][-1]}, {bbox["x_end"][-1]}), y: ({bbox["y_start"][-1]}, {bbox["y_end"][-1]})')

        x0, y0, x0_im, y0_im = ([None] * 9 for i in range(4))

        for idx in range(len(rods)):
            cropped_data = []
            cropped_data = arr_inv[bbox["x_start"][idx]:bbox["x_end"][idx], bbox["y_start"][idx]:bbox["y_end"][idx]]
            x0_im[idx], y0_im[idx], x0[idx], y0[idx] = self.fit_gauss_2d_to_rods(
                cropped_data, bbox["intensity_max"][idx], bbox["rod_dia"][idx],
                bbox["radius"], bbox["x_start"][idx], bbox["y_start"][idx])

            # note: flipped x/y
            rods[idx].x = y0_im[idx]
            rods[idx].y = x0_im[idx]

        rods = self.sort_rods(rods)

        # save figure
        if self.report:
            fig, axes = plt.subplots(1, 3, figsize=(45, 15))
            fig.tight_layout(pad=1)
            # center-of-mass (original method)
            axes[0].set_title("Initial Estimate")
            axes[0].imshow(arr, cmap='gray')
            for idx in range(len(rods)):
                axes[0].plot(rods_initial[idx].x, rods_initial[idx].y, 'y.')
            # gauss 2D
            axes[1].set_title("2D Gaussian Fit")
            axes[1].imshow(arr, cmap='gray')
            for idx in range(len(rods)):
                axes[1].plot(rods[idx].x, rods[idx].y, 'r.')
            # combined
            axes[2].set_title("Initial Estimate vs. 2D Gaussian Fit")
            axes[2].imshow(arr, cmap='gray')
            for idx in range(len(rods)):
                axes[2].plot(rods_initial[idx].x, rods_initial[idx].y, 'y.')
                axes[2].plot(rods[idx].x, rods[idx].y, 'r.')
            img_path = os.path.realpath(os.path.join(
                self.report_path, f'{self.img_desc(self.single_dcm)}_rod_centroids.png'))
            fig.savefig(img_path)
            self.report_files.append(img_path)

        return rods, rods_initial

    def plot_rods(self, ax, arr, rods, rods_initial):  # pragma: no cover
        ax.imshow(arr, cmap='gray')
        mark = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
        for idx, i in enumerate(rods):
            # ax.plot(rods_initial[idx].x, rods_initial[idx].y, 'y.', markersize=2)  # center-of-mass method
            ax.plot(rods[idx].x, rods[idx].y, 'r.', markersize=2)  # gauss 2D
            ax.scatter(x=i.x + 5, y=i.y - 5, marker=f"${mark[idx]}$", s=30, linewidths=0.4, c="w")

        ax.set_title('Rod Centroids')
        return ax

    def get_rod_distances(self, rods):
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
        horz_dist = [
            np.sqrt(np.square((rods[2].y - rods[0].y)) + np.square(rods[2].x - rods[0].x)),
            np.sqrt(np.square((rods[5].y - rods[3].y)) + np.square(rods[5].x - rods[3].x)),
            np.sqrt(np.square((rods[8].y - rods[6].y)) + np.square(rods[8].x - rods[6].x))
            ]

        vert_dist = [
            np.sqrt(np.square((rods[0].y - rods[6].y)) + np.square(rods[0].x - rods[6].x)),
            np.sqrt(np.square((rods[1].y - rods[7].y)) + np.square(rods[1].x - rods[7].x)),
            np.sqrt(np.square((rods[2].y - rods[8].y)) + np.square(rods[2].x - rods[8].x)),
        ]

        return horz_dist, vert_dist

    def get_rod_distortion_correction_coefficients(self, horizontal_distances) -> dict:
        """
        Removes the effect of geometric distortion from the slice width measurement. Assumes that rod separation is
        120 mm.

        Parameters
        ----------
        horizontal_distances : list
            horizontal distances between rods, in pixels

        Returns
        -------
        coefficients : dict
            dictionary containing top and bottom distortion corrections, in mm
        """

        coefficients = {"top": np.mean(horizontal_distances[1:3]) * self.pixel_size / 120,
                        "bottom": np.mean(horizontal_distances[0:2]) * self.pixel_size / 120}

        return coefficients

    def get_rod_distortions(self, horz_dist, vert_dist):
        """

        Parameters
        ----------
        horizontal distances
        vertical distances

        Returns
        -------
        horz_distortion, vert_distortion : float
            horizontal and vertical distortion values, in mm
        """

        # calculate the horizontal and vertical distances
        horz_dist_mm = np.multiply(self.pixel_size, horz_dist)
        vert_dist_mm = np.multiply(self.pixel_size, vert_dist)

        horz_distortion = 100 * np.std(horz_dist_mm, ddof=1) / np.mean(horz_dist_mm)  # ddof to match MATLAB std
        vert_distortion = 100 * np.std(vert_dist_mm, ddof=1) / np.mean(vert_dist_mm)
        return horz_distortion, vert_distortion

    def baseline_correction(self, profile, sample_spacing):
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

    def gauss_2d(self, xy_tuple, A, x_0, y_0, sigma_x, sigma_y, theta, C):
        """
        Create 2D Gaussian
        Based on code by Siân Culley, UCL/KCL
        See also: https://en.wikipedia.org/wiki/Gaussian_function#Two-dimensional_Gaussian_function

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

    def fit_gauss_2d_to_rods(self, cropped_data, gauss_amp, gauss_radius, box_radius, x_start, y_start):
        """
        Fit 2D Gaussian to Rods
        - Important:
        --- This uses a cropped region around a rod. If the cropped region is too large,
        such that it includes signal with intensity similar to the rods, the fitting may fail.
        --- This is a maximisation function, hence the rods should have higher signal than the surrounding region
        Based on code by Siân Culley, UCL/KCL

        Parameters
        ----------
        cropped_data : 2D array of magnitude voxels (nb: should be inverted if rods hypointense)
        gauss_amp : initial estimate of amplitude of 2D Gaussian
        gauss_radius : initial estimate of centre of 2D Gaussian
        box_radius : 'radius' of box around rod
        x_start / y_start : coordinates of bounding box in original non-cropped data

        Returns
        -------
        x0_im / y0_im : rod centroid coordinates in dimensions of original image
        x0 / y0 : rod centroid coordinates in dimensions of cropped image
        """

        # get (x,y) coordinates for fitting
        indices = np.indices(cropped_data.shape)

        # estimate initial conditions for 2d gaussian fit
        dims_crop = cropped_data.shape
        h_crop = dims_crop[0]
        w_crop = dims_crop[1]

        A = gauss_amp  # np.max() # amp of Gaussian
        sigma = gauss_radius / 2  # radius of 2D Gaussian
        C = np.mean([cropped_data[0, 0], cropped_data[h_crop - 1, 0], cropped_data[0, w_crop - 1],
                     cropped_data[h_crop - 1, w_crop - 1]])  # background – np.min(outside of rod within cropped_data)

        # print("A:", A)
        # print("box_radius:", box_radius)
        # print("sigma:", sigma)
        # print("C:", C, "\n")

        p0 = [A, box_radius, box_radius, sigma, sigma, 0, C]
        # print(f'initial conditions for 2d gaussian fitting: {p0}\n')

        # do 2d gaussian fit to data
        popt_single, pcov_single = opt.curve_fit(self.gauss_2d, indices, cropped_data.ravel(), p0=p0)

        A = popt_single[0]
        x0 = popt_single[1]
        y0 = popt_single[2]
        sigma_x = popt_single[3]
        sigma_y = popt_single[4]
        theta = popt_single[5]
        C = popt_single[6]

        # print(f'results of 2d gaussian fitting: \n\tamplitude = {A_} \n\tx0 = {x0} \n\ty0 = {y0} \n\tsigma_x = {sigma_x} \n\tsigma_y = {sigma_y} \n\ttheta = {theta} \n\tC = {C} \n')

        # to get image coordinates need to add back on x_start and y_start
        x0_im = x0 + x_start
        y0_im = y0 + y_start

        # print(f'Initial centre was ({rods[idx].x}, {rods[idx].y}). Refined centre is ({x0_im}, {y0_im})\n')

        return x0_im, y0_im, x0, y0

    def trapezoid(self, n_ramp, n_plateau, n_left_baseline, n_right_baseline, plateau_amplitude):
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

    def get_ramp_profiles(self, image_array, rods) -> dict:
        """
        Find the central y-axis point for the top and bottom profiles
        done by finding the distance between the mid-distances of the central rods

        Parameters
        ----------
        image_array
        rods

        Returns
        -------
        """

        top_profile_vertical_centre = np.round(((rods[3].y - rods[6].y) / 2) + rods[6].y).astype(int)
        bottom_profile_vertical_centre = np.round(((rods[0].y - rods[3].y) / 2) + rods[3].y).astype(int)

        # Selected 20mm around the mid-distances and take the average to find the line profiles
        top_profile = image_array[
                      (top_profile_vertical_centre - round(10 / self.pixel_size)):(
                                  top_profile_vertical_centre + round(10 / self.pixel_size)),
                      int(rods[3].x):int(rods[5].x)]

        bottom_profile = image_array[
                         (bottom_profile_vertical_centre - round(10 / self.pixel_size)):(
                                     bottom_profile_vertical_centre + round(10 / self.pixel_size)),
                         int(rods[3].x):int(rods[5].x)]

        return {"top": top_profile, "bottom": bottom_profile,
                "top-centre": top_profile_vertical_centre, "bottom-centre": bottom_profile_vertical_centre}

    def get_initial_trapezoid_fit_and_coefficients(self, profile, slice_thickness):
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

        trapezoid_fit_initial, _ = self.trapezoid(n_ramp, n_plateau, n_left_baseline, n_right_baseline, plateau_amplitude)

        return trapezoid_fit_initial, trapezoid_fit_coefficients

    def fit_trapezoid(self, profiles, slice_thickness):
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
        trapezoid_fit, trapezoid_fit_coefficients = self.get_initial_trapezoid_fit_and_coefficients(
            profiles["profile_corrected_interpolated"], slice_thickness)

        x_interp = profiles["x_interpolated"]
        profile_interp = profiles["profile_interpolated"]
        baseline_interpolated = profiles["baseline_fit"](x_interp)
        baseline_fit_coefficients = profiles["baseline_fit"]
        baseline_fit_coefficients = [baseline_fit_coefficients.c[0], baseline_fit_coefficients.c[1],
                                     baseline_fit_coefficients.c[2]]
        # sum squared differences
        current_error = sum((profiles["profile_corrected_interpolated"] - (baseline_interpolated + trapezoid_fit)) ** 2)

        def get_error(base, trap):
            """ Check if fit is improving """
            trapezoid_fit_temp, _ = self.trapezoid(*trap)

            baseline_fit_temp = np.poly1d(base)(x_interp)

            sum_squared_difference = sum((profile_interp - (baseline_fit_temp + trapezoid_fit_temp)) ** 2)

            return sum_squared_difference

        cont = 1
        j = 0
        # Go through a series of changes to reduce error,
        # if error doesn't reduced in one entire loop then exit
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

    def get_slice_width(self, dcm):
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

        rods, rods_initial = self.get_rods(arr)
        horz_distances, vert_distances = self.get_rod_distances(rods)
        horz_distortion_mm, vert_distortion_mm = self.get_rod_distortions(
            horz_distances, vert_distances
            )
        correction_coefficients_mm = self.get_rod_distortion_correction_coefficients(
            horizontal_distances=horz_distances)

        ramp_profiles = self.get_ramp_profiles(arr, rods)
        ramp_profiles_baseline_corrected = {"top": self.baseline_correction(np.mean(ramp_profiles["top"], axis=0),
                                                                       sample_spacing),
                                            "bottom": self.baseline_correction(np.mean(ramp_profiles["bottom"], axis=0),
                                                                          sample_spacing)}

        trapezoid_coefficients, baseline_coefficients = self.fit_trapezoid(ramp_profiles_baseline_corrected["top"],
                                                                      dcm.SliceThickness)
        top_trap, fwhm = self.trapezoid(*trapezoid_coefficients)

        slice_width_mm["top"]["default"] = fwhm * sample_spacing * self.pixel_size * np.tan((11.3 * pi) / 180)
        # Factor of 4 because interpolated by factor of four

        slice_width_mm["top"]["geometry_corrected"] = slice_width_mm["top"]["default"] / correction_coefficients_mm[
            "top"]

        # AAPM method directly incorporating phantom tilt
        slice_width_mm["top"]["aapm"] = fwhm * sample_spacing * self.pixel_size

        # AAPM method directly incorporating phantom tilt and independent of geometric linearity
        slice_width_mm["top"]["aapm_corrected"] = (fwhm * sample_spacing * self.pixel_size) / correction_coefficients_mm[
            "top"]

        trapezoid_coefficients, baseline_coefficients = self.fit_trapezoid(ramp_profiles_baseline_corrected["bottom"],
                                                                      dcm.SliceThickness)
        bottom_trap, fwhm = self.trapezoid(*trapezoid_coefficients)

        slice_width_mm["bottom"]["default"] = fwhm * sample_spacing * self.pixel_size * np.tan((11.3 * pi) / 180)
        # Factor of 4 because interpolated by factor of four

        slice_width_mm["bottom"]["geometry_corrected"] = slice_width_mm["bottom"]["default"] / \
                                                         correction_coefficients_mm["bottom"]

        # AAPM method directly incorporating phantom tilt
        slice_width_mm["bottom"]["aapm"] = fwhm * sample_spacing * self.pixel_size

        # AAPM method directly incorporating phantom tilt and independent of geometric linearity
        slice_width_mm["bottom"]["aapm_corrected"] = (fwhm * sample_spacing * self.pixel_size) / correction_coefficients_mm[
            "bottom"]

        # Geometric mean of slice widths (pg 34 of IPEM Report 80)
        slice_width_mm["combined"]["default"] = (slice_width_mm["top"]["default"] * slice_width_mm["bottom"][
            "default"]) ** 0.5
        slice_width_mm["combined"]["geometry_corrected"] = (slice_width_mm["top"]["geometry_corrected"] *
                                                            slice_width_mm["bottom"]["geometry_corrected"]) ** 0.5

        # AAPM method directly incorporating phantom tilt
        theta = (180.0 - 2.0 * 11.3) * pi / 180.0
        term1 = (np.cos(theta)) ** 2.0 * (slice_width_mm["bottom"]["aapm"] - slice_width_mm["top"]["aapm"]) ** 2.0 + (
                    4.0 * slice_width_mm["bottom"]["aapm"] * slice_width_mm["top"]["aapm"])
        term2 = (slice_width_mm["bottom"]["aapm"] + slice_width_mm["top"]["aapm"]) * np.cos(theta)
        term3 = 2.0 * np.sin(theta)

        slice_width_mm["combined"]["aapm_tilt"] = (term1 ** 0.5 + term2) / term3
        phantom_tilt = np.arctan(slice_width_mm["combined"]["aapm_tilt"] / slice_width_mm["bottom"]["aapm"]) + (
                    theta / 2.0) - pi / 2.0
        phantom_tilt_deg = phantom_tilt * (180.0 / pi)

        phantom_tilt_check = -np.arctan(slice_width_mm["combined"]["aapm_tilt"] / slice_width_mm["top"]["aapm"]) - (
                    theta / 2.0) + pi / 2.0
        phantom_tilt_check_deg = phantom_tilt_check * (180.0 / pi)

        # AAPM method directly incorporating phantom tilt and independent of geometric linearity
        theta = (180.0 - 2.0 * 11.3) * pi / 180.0
        term1 = (np.cos(theta)) ** 2.0 * (
                    slice_width_mm["bottom"]["aapm_corrected"] - slice_width_mm["top"]["aapm_corrected"]) ** 2.0 + (
                            4.0 * slice_width_mm["bottom"]["aapm_corrected"] * slice_width_mm["top"]["aapm_corrected"])
        term2 = (slice_width_mm["bottom"]["aapm_corrected"] + slice_width_mm["top"]["aapm_corrected"]) * np.cos(theta)
        term3 = 2.0 * np.sin(theta)

        slice_width_mm["combined"]["aapm_tilt_corrected"] = (term1 ** 0.5 + term2) / term3
        phantom_tilt = np.arctan(
            slice_width_mm["combined"]["aapm_tilt_corrected"] / slice_width_mm["bottom"]["aapm_corrected"]) + (
                                   theta / 2.0) - pi / 2.0
        phantom_tilt_deg = phantom_tilt * (180.0 / pi)

        phantom_tilt_check = -np.arctan(
            slice_width_mm["combined"]["aapm_tilt_corrected"] / slice_width_mm["top"]["aapm_corrected"]) - (
                                     theta / 2.0) + pi / 2.0
        phantom_tilt_check_deg = phantom_tilt_check * (180.0 / pi)

        # calculate linearity in mm from distances in pixels

        horizontal_linearity_mm = np.mean(horz_distances) * self.pixel_size
        vertical_linearity_mm = np.mean(vert_distances) * self.pixel_size

        # calculate horizontal and vertical distances in mm from distances in pixels, for output

        horz_distances_mm = [round(x * self.pixel_size, 3) for x in horz_distances]

        vert_distances_mm = [round(x * self.pixel_size, 3) for x in vert_distances]

        if self.report:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(6, 1, gridspec_kw={'height_ratios': [3, 1, 1, 1, 1, 1]})
            fig.set_size_inches(6, 16)
            fig.tight_layout(pad=1)

            self.plot_rods(axes[0], arr, rods, rods_initial)

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
                cellText=[[str(x) for x in horz_distances_mm] + [str(np.around(horizontal_linearity_mm, 3))],
                          [str(x) for x in vert_distances_mm] + [str(np.around(vertical_linearity_mm, 3))]],
                rowLabels=['H-distances (S->I)',
                           'V-distances (R->L)'],
                colLabels=['1', '2', '3', 'mean/linearity'],
                colWidths=[0.15] * (len(horz_distances) + 1),  # plus one for linearity,
                rowLoc="center",
                loc="center"
            )

            img_path = os.path.realpath(os.path.join(
                            self.report_path, f'{self.img_desc(dcm)}.png'))
            fig.savefig(img_path)
            self.report_files.append(img_path)

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
        
        distortion_values = {
            "vertical mm": round(vert_distortion_mm, 2),
            "horizontal mm": round(horz_distortion_mm, 2)
        }
        
        linearity_values = {
            "vertical mm": round(vertical_linearity_mm, 2),
            "horizontal mm": round(horizontal_linearity_mm, 2)
        }

        return {'slice width mm': round(slice_width_mm['combined']['aapm_tilt_corrected'], 2),
                'distortion values': distortion_values, 'linearity values': linearity_values,
                'horizontal distances mm': horz_distances_mm, 'vertical distances mm': vert_distances_mm}
