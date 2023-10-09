"""
ACR Slice Thickness

Calculates the slice thickness for slice 1 of the ACR phantom.

The ramps located in the middle of the phantom are located and line profiles are drawn through them. The full-width
half-maximum (FWHM) of each ramp is determined to be their length. Using the formula described in the ACR guidance, the
slice thickness is then calculated.

Created by Yassine Azma
yassine.azma@rmh.nhs.uk

31/01/2022
"""

import sys
import traceback
import os
import numpy as np
import skimage.morphology
import skimage.measure
import scipy

from hazenlib.HazenTask import HazenTask
from hazenlib.ACRObject import ACRObject


class ACRSliceThickness(HazenTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ACR_obj = ACRObject(self.dcm_list)

    def run(self) -> dict:
        slice_thickness_dcm = self.ACR_obj.dcms[0]

        # Initialise results dictionary
        results = self.init_result_dict()
        results['file'] = self.img_desc(slice_thickness_dcm)

        try:
            result = self.get_slice_thickness(slice_thickness_dcm)
            results['measurement'] = {
                "slice width mm": round(result, 2)
                }
        except Exception as e:
            print(f"Could not calculate the slice thickness for {self.img_desc(slice_thickness_dcm)} because of : {e}")
            traceback.print_exc(file=sys.stdout)

        # only return reports if requested
        if self.report:
            results['report_image'] = self.report_files

        return results

    def find_ramps(self, img, centre, res):
        # X
        investigate_region = int(np.ceil(5.5 / res[1]).item())

        if np.mod(investigate_region, 2) == 0:
            investigate_region = (investigate_region + 1)

        # Line profiles around the central row
        invest_x = \
            [skimage.measure.profile_line(img, (centre[1] + k, 1), (centre[1] + k, img.shape[1]), mode='constant')
             for k in range(investigate_region)]

        invest_x = np.array(invest_x).T
        mean_x_profile = np.mean(invest_x, 1)
        abs_diff_x_profile = np.absolute(np.diff(mean_x_profile))

        # find the points corresponding to the transition between:
        # [0] - background and the hyperintense phantom
        # [1] - hyperintense phantom and hypointense region with ramps
        # [2] - hypointense region with ramps and hyperintense phantom
        # [3] - hyperintense phantom and background

        x_peaks, _ = self.ACR_obj.find_n_highest_peaks(abs_diff_x_profile, 4)
        x_locs = np.sort(x_peaks) - 1

        width_pts = [x_locs[1], x_locs[2]]
        width = np.max(width_pts) - np.min(width_pts)

        # take rough estimate of x points for later line profiles
        x = np.round([np.min(width_pts) + 0.2 * width, np.max(width_pts) - 0.2 * width])
        # Y
        c = skimage.measure.profile_line(img, (centre[1] - 2 * investigate_region, centre[0]),
                                         (centre[1] + 2 * investigate_region, centre[0]), mode='constant').flatten()

        abs_diff_y_profile = np.absolute(np.diff(c))

        y_peaks, _ = self.ACR_obj.find_n_highest_peaks(abs_diff_y_profile, 2)
        y_locs = centre[1] - 2 * investigate_region + 1 + y_peaks
        height = np.max(y_locs) - np.min(y_locs)

        y = np.round([np.max(y_locs) - 0.25 * height, np.min(y_locs) + 0.25 * height])

        return x, y

    def FWHM(self, data):
        baseline = np.min(data)
        data -= baseline
        half_max = np.max(data) * 0.5

        # Naive attempt
        half_max_crossing_indices = np.argwhere(np.diff(np.sign(data - half_max))).flatten()
        half_max_points = (half_max_crossing_indices[0], half_max_crossing_indices[-1])

        # Interpolation

        def simple_interp(x_start, ydata):
            x_init = x_start - 5
            x_pts = np.arange(x_init, x_init + 11)
            y_pts = ydata[x_pts]

            grad = (y_pts[-1] - y_pts[0]) / (x_pts[-1] - x_pts[0])

            x_true = x_start + (half_max - ydata[x_start]) / grad

            return x_true

        FWHM_pts = simple_interp(half_max_points[0], data), simple_interp(half_max_points[1], data)

        return FWHM_pts

    def get_slice_thickness(self, dcm):
        img = dcm.pixel_array
        res = dcm.PixelSpacing  # In-plane resolution from metadata
        cxy = self.ACR_obj.centre
        x_pts, y_pts = self.find_ramps(img, cxy, res)

        interp_factor = 5
        sample = np.arange(1, x_pts[1] - x_pts[0] + 2)
        new_sample = np.arange(1, x_pts[1] - x_pts[0] + (1 / interp_factor), (1 / interp_factor))
        offsets = np.arange(-3, 4)
        ramp_length = np.zeros((2, 7))

        line_store = []
        fwhm_store = []
        for i, offset in enumerate(offsets):
            lines = [skimage.measure.profile_line(img, (offset + y_pts[0], x_pts[0]), (offset + y_pts[0], x_pts[1]),
                                                  linewidth=2, mode='constant').flatten(),
                     skimage.measure.profile_line(img, (offset + y_pts[1], x_pts[0]), (offset + y_pts[1], x_pts[1]),
                                                  linewidth=2, mode='constant').flatten()]

            interp_lines = [scipy.interpolate.interp1d(sample, line)(new_sample) for line in lines]
            fwhm = [self.FWHM(interp_line) for interp_line in interp_lines]
            ramp_length[0, i] = (1 / interp_factor) * np.diff(fwhm[0]) * res[0]
            ramp_length[1, i] = (1 / interp_factor) * np.diff(fwhm[1]) * res[0]

            line_store.append(interp_lines)
            fwhm_store.append(fwhm)

        with np.errstate(divide='ignore', invalid='ignore'):
            dz = 0.2 * (np.prod(ramp_length, axis=0)) / np.sum(ramp_length, axis=0)

        dz = dz[~np.isnan(dz)]
        z_ind = np.argmin(np.abs(dcm.SliceThickness - dz))

        slice_thickness = dz[z_ind]

        if self.report:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(4, 1)
            fig.set_size_inches(8, 24)
            fig.tight_layout(pad=4)

            x_ramp = new_sample * res[0]
            x_extent = np.max(x_ramp)
            y_ramp = line_store[z_ind][1]
            y_extent = np.max(y_ramp)
            max_loc = np.argmax(y_ramp) * (1 / interp_factor) * res[0]

            axes[0].imshow(img)
            axes[0].scatter(cxy[0], cxy[1], c='red')
            axes[0].axis('off')
            axes[0].set_title('Centroid Location')

            axes[1].imshow(img)
            axes[1].plot([x_pts[0], x_pts[1]], offsets[z_ind] + [y_pts[0], y_pts[0]], 'b-')
            axes[1].plot([x_pts[0], x_pts[1]], offsets[z_ind] + [y_pts[1], y_pts[1]], 'r-')
            axes[1].axis('off')
            axes[1].set_title('Line Profiles')

            xmin = fwhm_store[z_ind][1][0] * (1 / interp_factor) * res[0] / x_extent
            xmax = fwhm_store[z_ind][1][1] * (1 / interp_factor) * res[0] / x_extent

            axes[2].plot(x_ramp, y_ramp, 'r', 
                        label=f'FWHM={np.round(ramp_length[1][z_ind], 2)}mm')
            axes[2].axhline(0.5 * y_extent, linestyle='dashdot', color='k',
                        xmin=xmin, xmax=xmax)
            axes[2].axvline(max_loc, linestyle='dashdot', color='k',
                        ymin=0, ymax=10 / 11)

            axes[2].set_xlabel('Relative Position (mm)')
            axes[2].set_xlim([0, x_extent])
            axes[2].set_ylim([0, y_extent * 1.1])
            axes[2].set_title('Upper Ramp')
            axes[2].grid()
            axes[2].legend(loc='best')

            xmin = fwhm_store[z_ind][0][0] * (1 / interp_factor) * res[0] / x_extent
            xmax = fwhm_store[z_ind][0][1] * (1 / interp_factor) * res[0] / x_extent
            x_ramp = new_sample * res[0]
            x_extent = np.max(x_ramp)
            y_ramp = line_store[z_ind][0]
            y_extent = np.max(y_ramp)
            max_loc = np.argmax(y_ramp) * (1 / interp_factor) * res[0]

            axes[3].plot(x_ramp, y_ramp, 'b', label=f'FWHM={np.round(ramp_length[0][z_ind], 2)}mm')
            axes[3].axhline(0.5 * y_extent, xmin=xmin, xmax=xmax, linestyle='dashdot',
                        color='k')
            axes[3].axvline(max_loc, ymin=0, ymax=10 / 11, linestyle='dashdot', color='k')

            axes[3].set_xlabel('Relative Position (mm)')
            axes[3].set_xlim([0, x_extent])
            axes[3].set_ylim([0, y_extent * 1.1])
            axes[3].set_title('Lower Ramp')
            axes[3].grid()
            axes[3].legend(loc='best')

            img_path = os.path.realpath(os.path.join(
                self.report_path, f'{self.img_desc(dcm)}_slice_thickness.png'))
            fig.savefig(img_path)
            self.report_files.append(img_path)

        return slice_thickness
