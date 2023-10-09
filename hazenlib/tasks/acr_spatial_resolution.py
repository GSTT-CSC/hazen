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
from hazenlib.ACRObject import ACRObject
from hazenlib.logger import logger


class ACRSpatialResolution(HazenTask):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ACR_obj = ACRObject(self.dcm_list)

    def run(self) -> dict:

        rot_ang = self.ACR_obj.rot_angle
        if np.abs(rot_ang) < 3:
            logger.warning(f'The estimated rotation angle of the ACR phantom is {np.round(rot_ang, 3)} degrees, which '
                           f'is less than the recommended 3 degrees. Results will be unreliable!')

        # Identify relevant slices
        mtf_dcm = self.ACR_obj.dcms[0]

        # Initialise results dictionary
        results = self.init_result_dict()
        results['file'] = self.img_desc(mtf_dcm)

        try:
            raw_res, fitted_res = self.get_mtf50(mtf_dcm)
            results['measurement'] = {
                "estimated rotation angle": round(rot_ang, 2),
                "raw mtf50": round(raw_res, 2),
                "fitted mtf50": round(fitted_res, 2)
            }
        except Exception as e:
            print(f"Could not calculate the spatial resolution for {self.img_desc(mtf_dcm)} because of : {e}")
            traceback.print_exc(file=sys.stdout)

        # only return reports if requested
        if self.report:
            results['report_image'] = self.report_files

        return results

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

        _, pk_rows_height = self.ACR_obj.find_n_highest_peaks(np.abs(np.diff(edge_sum_rows)), 1)
        _, pk_cols_height = self.ACR_obj.find_n_highest_peaks(np.abs(np.diff(edge_sum_cols)), 1)

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

        popt, pcov = scipy.optimize.curve_fit(func, np.arange(1, len(erf) + 1), erf, sigma=(1 / weights),
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
        cxy = self.ACR_obj.centre

        ramp_x, ramp_y = int(cxy[0]), self.y_position_for_ramp(res, img, cxy)
        width = int(13 * img.shape[0] / 256)
        crop_img = self.crop_image(img, ramp_x, ramp_y, width)
        edge_type, direction = self.get_edge_type(crop_img)
        slope, surface = self.fit_normcdf_surface(crop_img, edge_type, direction)
        erf = self.sample_erf(crop_img, slope, edge_type)
        erf_fit = self.fit_erf(erf)

        freq, lsf_raw, MTF_raw = self.calculate_MTF(erf, res)
        _, lsf_fit, MTF_fit = self.calculate_MTF(erf_fit, res)

        eff_raw_res = self.identify_MTF50(freq, MTF_raw)
        eff_fit_res = self.identify_MTF50(freq, MTF_fit)

        if self.report:
            edge_loc = self.edge_location_for_plot(crop_img, edge_type)
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches

            fig, axes = plt.subplots(5, 1)
            fig.set_size_inches(8, 40)
            fig.tight_layout(pad=4)

            axes[0].imshow(img, interpolation='none')
            rect = patches.Rectangle((ramp_x - width // 2 - 1, ramp_y - width // 2 - 1), width, width, linewidth=1,
                                     edgecolor='w', facecolor='none')
            axes[0].add_patch(rect)
            axes[0].axis('off')
            axes[0].set_title('Segmented Edge')

            axes[1].imshow(crop_img)
            if edge_type == 'vertical':
                axes[1].plot(np.arange(0, width - 1), np.mean(edge_loc) - slope * np.arange(0, width - 1), color='r')
            else:
                axes[1].plot(np.mean(edge_loc) + slope * np.arange(0, width - 1), np.arange(0, width - 1), color='r')
            axes[1].axis('off')
            axes[1].set_title('Cropped Edge', fontsize=14)

            axes[2].plot(erf, 'rx', ms=5, label='Raw Data')
            axes[2].plot(erf_fit, 'k', lw=3, label='Fitted Data')
            axes[2].set_ylabel('Signal Intensity')
            axes[2].set_xlabel('Pixel')
            axes[2].grid()
            axes[2].legend(fancybox='true')
            axes[2].set_title('ERF', fontsize=14)

            axes[3].plot(lsf_raw, 'rx', ms=5, label='Raw Data')
            axes[3].plot(lsf_fit, 'k', lw=3, label='Fitted Data')
            axes[3].set_ylabel(r'$\Delta$' + ' Signal Intensity')
            axes[3].set_xlabel('Pixel')
            axes[3].grid()
            axes[3].legend(fancybox='true')
            axes[3].set_title('LSF', fontsize=14)

            axes[4].plot(freq, MTF_raw, 'rx', ms=8, label=f'Raw Data - {round(eff_raw_res, 2)}mm @ 50%')
            axes[4].plot(freq, MTF_fit, 'k', lw=3, label=f'Weighted Sigmoid Fit of ERF - {round(eff_fit_res, 2)}mm @ 50%')
            axes[4].set_xlabel('Spatial Frequency (lp/mm)')
            axes[4].set_ylabel('Modulation Transfer Ratio')
            axes[4].set_xlim([-0.05, 1])
            axes[4].set_ylim([0, 1.05])
            axes[4].grid()
            axes[4].legend(fancybox='true')
            axes[4].set_title('MTF', fontsize=14)

            img_path = os.path.realpath(os.path.join(
                self.report_path, f'{self.img_desc(dcm)}.png'))
            fig.savefig(img_path)
            self.report_files.append(img_path)

        return eff_raw_res, eff_fit_res
