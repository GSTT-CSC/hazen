"""
ACR Spatial Resolution (MTF)

https://www.acraccreditation.org/-/media/acraccreditation/documents/mri/largephantomguidance.pdf

Calculates the effective resolution (MTF50) for slice 1 for the ACR phantom. This is done in accordance with the
methodology described in Section 3 of the following paper:

https://opg.optica.org/oe/fulltext.cfm?uri=oe-22-5-6040&id=281325

WARNING: The phantom must be slanted for valid results to be produced. This test is not within the scope of ACR
guidance.

This script first identifies the rotation angle of the ACR phantom using slice 1. It provides a warning if the
slanted angle is less than 3 degrees.

The location of the ramps within the slice thickness are identified and a square ROI is selected around the anterior
edge of the slice thickness insert.

A rudimentary edge response function is generated based on the edge within the ROI to provide initialisation values for
the 2D normal cumulative distribution fit of the ROI.

The edge is then super-sampled in the direction of the bright-dark transition of the edge and binned at right angles
based on the edge slope determined from the 2D Normal CDF fit of the ROI to obtain the edge response function.

This super-sampled ERF is then fitted using a weighted sigmoid function. The raw data and this fit are then used to
determine the LSF and the subsequent MTF. The MTF50 for both raw and fitted data are reported.

The results are also visualised.

Created by Yassine Azma
yassine.azma@rmh.nhs.uk

22/02/2023
"""

import sys
import traceback
import scipy
import cv2
import os
import numpy as np
import skimage.morphology
import skimage.measure

from hazenlib.HazenTask import HazenTask


def find_n_peaks(data, n, height=1):
    peaks = scipy.signal.find_peaks(data, height)
    pk_heights = peaks[1]['peak_heights']
    pk_ind = peaks[0]

    peak_heights = pk_heights[(-pk_heights).argsort()[:n]]
    peak_locs = pk_ind[(-pk_heights).argsort()[:n]]  # find n highest peaks

    return np.sort(peak_locs), np.sort(peak_heights)


class ACRSpatialResolution(HazenTask):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self) -> dict:
        mtf_results = {}
        z = []
        for dcm in self.data:
            z.append(dcm.ImagePositionPatient[2])

        idx_sort = np.argsort(z)

        for dcm in self.data:
            if dcm.ImagePositionPatient[2] == z[idx_sort[0]]:
                try:
                    raw_res, fitted_res = self.get_mtf50(dcm)
                    mtf_results[f"raw_mtf50_{self.key(self.data[0])}"] = raw_res
                    mtf_results[f"fitted_mtf50_{self.key(self.data[0])}"] = fitted_res
                except Exception as e:
                    print(f"Could not calculate the spatial resolution for {self.key(dcm)} because of : {e}")
                    traceback.print_exc(file=sys.stdout)
                    continue

        results = {self.key(self.data[0]): mtf_results, 'reports': {'images': self.report_files}}

        return results

    def centroid_com(self, dcm):
        # Calculate centroid of object using a centre-of-mass calculation
        thresh_img = dcm > 0.25 * np.max(dcm)
        open_img = skimage.morphology.area_opening(thresh_img, area_threshold=500)
        bhull = skimage.morphology.convex_hull_image(open_img)
        coords = np.nonzero(bhull)  # row major - first array is columns

        sum_x = np.sum(coords[1])
        sum_y = np.sum(coords[0])
        cx, cy = sum_x / coords[0].shape[0], sum_y / coords[1].shape[0]
        cxy = (round(cx), round(cy))

        return bhull, cxy

    def find_rotation(self, dcm):
        thresh = dcm * (dcm > 0.2 * np.max(dcm))

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilate = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)
        diff = cv2.absdiff(dilate, thresh)

        edges = (diff >= 200) * 1

        h, theta, d = skimage.transform.hough_line(edges)
        accum, angles, dists = skimage.transform.hough_line_peaks(h, theta, d)

        try:
            angle = np.rad2deg(scipy.stats.mode(angles)[0][0])
            rot_angle = angle + 90 if angle < 0 else angle - 90
        except IndexError:
            rot_angle = 0

        return rot_angle

    def y_position_for_ramp(self, res, img, cxy):
        investigate_region = int(np.ceil(5.5 / res[1]).item())

        if np.mod(investigate_region, 2) == 0:
            investigate_region = (investigate_region + 1)

        line_profile_y = skimage.measure.profile_line(img, (cxy[1] - 2 * investigate_region, cxy[0]),
                                                      (cxy[1] + 2 * investigate_region, cxy[1]),
                                                      mode='constant').flatten()

        abs_diff_y_profile = np.absolute(np.diff(line_profile_y))
        y_peaks = scipy.signal.find_peaks(abs_diff_y_profile, height=1)
        pk_heights = y_peaks[1]['peak_heights']
        pk_ind = y_peaks[0]
        highest_y_peaks = pk_ind[(-pk_heights).argsort()[:2]]
        y_locs = highest_y_peaks - 1

        height_pts = cxy[1] - 2 * investigate_region - 1 + y_locs

        y = np.min(height_pts) + 2

        return y

    def crop_image(self, img, x, y, width):
        crop_x, crop_y = (x - width // 2, x + width // 2), (y - width // 2, y + width // 2)
        crop_img = img[crop_y[0]:crop_y[1], crop_x[0]:crop_x[1]]

        return crop_img

    def get_edge_type(self, crop_img):
        edge_sum_rows = np.sum(crop_img, axis=1).astype(np.int_)
        edge_sum_cols = np.sum(crop_img, axis=0).astype(np.int_)

        _, pk_rows_height = find_n_peaks(np.abs(np.diff(edge_sum_rows)), 1)
        _, pk_cols_height = find_n_peaks(np.abs(np.diff(edge_sum_cols)), 1)

        edge_type = 'vertical' if pk_rows_height > pk_cols_height else 'horizontal'

        thresh_roi_crop = crop_img > 0.6 * np.max(crop_img)
        edge_dir = np.sum(thresh_roi_crop, axis=0) if edge_type == 'vertical' else np.sum(thresh_roi_crop, axis=1)
        if edge_type == 'vertical':
            direction = 'downward' if edge_dir[-1] > edge_dir[0] else 'upward'
        else:
            direction = 'leftward' if edge_dir[-1] > edge_dir[0] else 'rightward'

        return edge_type, direction

    def edge_location_for_plot(self, crop_img, edge_type):
        thresh_roi_crop = crop_img > 0.6 * np.max(crop_img)

        naive_lsf = np.abs(np.diff(np.sum(thresh_roi_crop, 1))) > 1 if edge_type == 'vertical' else np.abs(
            np.diff(np.sum(thresh_roi_crop, 0)))
        edge_test = np.diff(np.where(naive_lsf == 0))[0]
        edge_begin = np.where(edge_test > 1)
        edge_loc = np.array([edge_begin, edge_begin + edge_test[edge_begin] - 1]).flatten()

        return edge_loc

    def fit_normcdf_surface(self, crop_img, edge_type, direction):
        thresh_roi_crop = crop_img > 0.6 * np.max(crop_img)
        temp_x = np.linspace(1, thresh_roi_crop.shape[1], thresh_roi_crop.shape[1])
        temp_y = np.linspace(1, thresh_roi_crop.shape[0], thresh_roi_crop.shape[0])
        x, y = np.meshgrid(temp_x, temp_y)

        bright = max(crop_img[thresh_roi_crop])
        dark = 20 + np.min(crop_img[~thresh_roi_crop])

        def func(x, slope, mu, bright, dark):
            norm_cdf = (bright - dark) * scipy.stats.norm.cdf(x[0], mu + slope * x[1], 0.5) + dark

            return norm_cdf

        sign = 1 if direction in ('downward', 'leftward') else -1
        x_data = np.vstack((sign * x.ravel(), y.ravel())) if edge_type == 'vertical' else np.vstack(
            (sign * y.ravel(), x.ravel()))

        popt, pcov = scipy.optimize.curve_fit(func, x_data, crop_img.ravel(), p0=[0, 0, bright, dark], maxfev=1000)
        surface = func(x_data, popt[0], popt[1], popt[2], popt[3]).reshape(crop_img.shape)

        slope = 1 / popt[0] if direction in ('leftward', 'upward') else -1 / popt[0]

        return slope, surface

    def sample_erf(self, crop_img, slope, edge_type):
        resamp_factor = 8
        if edge_type == 'horizontal':
            resample_crop_img = cv2.resize(crop_img, (crop_img.shape[0] * resamp_factor, crop_img.shape[1]))
        else:
            resample_crop_img = cv2.resize(crop_img, (crop_img.shape[0], crop_img.shape[1] * resamp_factor))

        mid_loc = [i / 2 for i in resample_crop_img.shape]

        temp_x = np.linspace(1, resample_crop_img.shape[1], resample_crop_img.shape[1])
        temp_y = np.linspace(1, resample_crop_img.shape[0], resample_crop_img.shape[0])
        x_resample, y_resample = np.meshgrid(temp_x, temp_y)

        erf = []
        n_inside_roi = []
        if edge_type == 'horizontal':
            diffY = (y_resample - 1) - mid_loc[0]
            x_prime = x_resample + resamp_factor * diffY * slope

            x_min, x_max = np.min(x_prime).astype(int), np.max(x_prime).astype(int)

            for k in range(x_min, x_max):
                erf_val = np.mean(resample_crop_img[(x_prime >= k) & (x_prime < k + 1)])
                erf.append(erf_val)
                number_nonzero = np.count_nonzero(resample_crop_img[(x_prime >= k) & (x_prime < k + 1)])
                n_inside_roi.append(number_nonzero)
        else:
            diffX = (x_resample.shape[0] - 1) - x_resample - mid_loc[1]
            y_prime = np.flipud(y_resample) + resamp_factor * diffX * slope

            y_min, y_max = np.min(y_prime).astype(int), np.max(y_prime).astype(int)

            for k in range(y_min, y_max):
                erf_val = np.mean(resample_crop_img[(y_prime >= k) & (y_prime < k + 1)])
                erf.append(erf_val)
                number_nonzero = np.count_nonzero(resample_crop_img[(y_prime >= k) & (y_prime < k + 1)])
                n_inside_roi.append(number_nonzero)

        erf = np.array(erf)
        n_inside_roi = np.array(n_inside_roi)

        erf = erf[n_inside_roi == np.max(n_inside_roi)]

        return erf

    def fit_erf(self, erf):
        true_erf = np.diff(erf) > 0.2 * np.max(np.diff(erf))
        turning_points = np.where(true_erf)[0][0], np.where(true_erf)[0][-1]
        weights = 0.5 * np.ones((len(true_erf) + 1))
        weights[turning_points[0]:turning_points[1]] = 1

        def func(x, a, b, c, d, e):
            sigmoid = a + b / (1 + np.exp(c * (x - d))) ** e

            return sigmoid

        popt, pcov = scipy.optimize.curve_fit(func, np.arange(1, len(erf) + 1), erf, sigma=np.diag(1 / weights),
                                              p0=[np.min(erf), np.max(erf), 0, sum(turning_points) / 2, 1], maxfev=5000)
        erf_fit = func(np.arange(1, len(erf) + 1), popt[0], popt[1], popt[2], popt[3], popt[4])

        return erf_fit

    def calculate_MTF(self, erf, res):
        lsf = np.diff(erf)
        N = len(lsf)
        n = np.arange(-N / 2, N / 2) if N % 2 == 0 else np.arange(-(N - 1) / 2, (N + 1) / 2)

        resamp_factor = 8
        Fs = 1 / (np.sqrt(np.mean(np.square(res))) * (1 / resamp_factor))
        freq = n * Fs / N
        MTF = np.abs(np.fft.fftshift(np.fft.fft(lsf)))
        MTF = MTF / np.max(MTF)

        zero_freq = np.where(freq == 0)[0][0]
        freq = freq[zero_freq:]
        MTF = MTF[zero_freq:]

        return freq, lsf, MTF

    def identify_MTF50(self, freq, MTF):
        freq_interp = np.arange(0, 1.005, 0.005)
        MTF_interp = np.interp(freq_interp, freq, MTF, left=None, right=None, period=None)
        equivalent_linepairs = freq_interp[np.argmin(np.abs(MTF_interp - 0.5))]
        eff_res = 1 / (equivalent_linepairs * 2)

        return eff_res

    def get_mtf50(self, dcm):
        img = dcm.pixel_array
        res = dcm.PixelSpacing
        _, cxy = self.centroid_com(img)
        rot_ang = self.find_rotation(img)

        if np.round(np.abs(rot_ang), 2) < 3:
            print(f'Rotation angle of the ACR phantom is {np.round(rot_ang, 3)}, which has an absolute is less than 3 '
                  f'degree. Results will be unreliable!')
        else:
            print(f'Rotation angle of the ACR phantom is {np.round(rot_ang, 3)}')

        ramp_x, ramp_y = int(cxy[0]), self.y_position_for_ramp(res, img, cxy)
        width = int(13 * img.shape[0] / 256)
        crop_img = self.crop_image(img, ramp_x, ramp_y, width)
        edge_type, direction = self.get_edge_type(crop_img)
        slope, surface = self.fit_normcdf_surface(crop_img, edge_type, direction)
        erf = self.sample_erf(crop_img, slope, edge_type)
        erf_fit = self.fit_erf(erf)

        freq, lsf_raw, MTF_raw = self.calculate_MTF(erf, res)
        _, lsf_fit, MTF_fit = self.calculate_MTF(erf_fit, res)

        eff_raw_res = round(self.identify_MTF50(freq, MTF_raw), 2)
        eff_fit_res = round(self.identify_MTF50(freq, MTF_fit), 2)

        if self.report:
            edge_loc = self.edge_location_for_plot(crop_img, edge_type)
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches

            fig = plt.figure(figsize=(8, 8))

            gs = fig.add_gridspec(3, 2)
            ax0 = fig.add_subplot(gs[:2, 0])
            ax2 = fig.add_subplot(gs[2:, 0])
            ax3 = fig.add_subplot(gs[0, 1])
            ax4 = fig.add_subplot(gs[1, 1])
            ax5 = fig.add_subplot(gs[2, 1])

            ax0.imshow(img, interpolation='none')
            rect = patches.Rectangle((ramp_x - width // 2 - 1, ramp_y - width // 2 - 1), width, width, linewidth=1,
                                     edgecolor='w', facecolor='none')
            ax0.add_patch(rect)
            ax0.axis('off')

            ax2.imshow(crop_img)
            if edge_type == 'vertical':
                ax2.plot(np.arange(0, width - 1), np.mean(edge_loc) - slope * np.arange(0, width - 1), color='r')
            else:
                ax2.plot(np.mean(edge_loc) + slope * np.arange(0, width - 1), np.arange(0, width - 1), color='r')
            ax2.axis('off')
            ax2.set_title('Cropped Edge', fontsize=14)

            ax3.plot(erf, 'rx', ms=5)
            ax3.plot(erf_fit, 'k', lw=3)
            ax3.set_ylabel('Signal Intensity')
            ax3.set_xlabel('Pixel')
            ax3.grid()
            ax3.set_title('ERF', fontsize=14)

            ax4.plot(lsf_raw, 'rx', ms=5)
            ax4.plot(lsf_fit, 'k', lw=3)
            ax4.set_ylabel(r'$\Delta$' + ' Signal Intensity')
            ax4.set_xlabel('Pixel')
            ax4.grid()
            ax4.set_title('LSF', fontsize=14)

            ax5.plot(freq, MTF_raw, 'rx', ms=8, label=f'Raw Data - {round(eff_raw_res, 2)}mm @ 50%')
            ax5.plot(freq, MTF_fit, 'k', lw=3, label=f'Weighted Sigmoid Fit of ERF - {round(eff_fit_res, 2)}mm @ 50%')
            ax5.set_xlabel('Spatial Frequency (lp/mm)')
            ax5.set_ylabel('Modulation Transfer Ratio')
            ax5.set_xlim([-0.05, 1])
            ax5.set_ylim([0, 1.05])
            ax5.grid()
            ax5.legend(fancybox='true', bbox_to_anchor=[1.05, -0.25, 0, 0])
            ax5.set_title('MTF', fontsize=14)

            plt.tight_layout()

            img_path = os.path.realpath(os.path.join(self.report_path, f'{self.key(dcm)}.png'))
            fig.savefig(img_path)
            self.report_files.append(img_path)

        return eff_raw_res, eff_fit_res
