"""
ACR Slice Position

https://www.acraccreditation.org/-/media/acraccreditation/documents/mri/largephantomguidance.pdf

Calculates the bar length difference for slices 1 and 11 of the ACR phantom.

This script calculates the bar length difference in accordance with the ACR Guidance. Line profiles are drawn
vertically through the left and right wedges. The right wedge's line profile is shifted and wrapped round before being
subtracted from the left wedge's line profile, e.g.:

Right line profile: [1, 2, 3, 4, 5]
Right line profile wrapped round by 1: [2, 3, 4, 5, 1]

This wrapping process, from hereon referred to as circular shifting, is then used for subtractions.

The shift used to produce the minimum difference between the circularly shifted right line profile and the static left
one is used to determine the bar length difference, which is twice the slice position displacement.
The results are also visualised.

Created by Yassine Azma
yassine.azma@rmh.nhs.uk

28/12/2022
"""

import sys
import traceback
import scipy
import os
import numpy as np
import skimage.morphology
import skimage.measure

from hazenlib.HazenTask import HazenTask
from hazenlib.ACRObject import ACRObject


class ACRSlicePosition(HazenTask):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ACR_obj = ACRObject(self.dcm_list)

    def run(self) -> dict:
        # Identify relevant slices
        dcms = [self.ACR_obj.dcms[0], self.ACR_obj.dcms[-1]]

        # Initialise results dictionary
        results = self.init_result_dict()
        results['file'] = [self.img_desc(dcm) for dcm in dcms]
        results['measurement'] = {}

        for dcm in dcms:
            try:
                result = self.get_slice_position(dcm)
                results['measurement'][self.img_desc(dcm)] = {
                    "length difference": round(result, 2)
                    }
            except Exception as e:
                print(f"Could not calculate the bar length difference for {self.img_desc(dcm)} because of : {e}")
                traceback.print_exc(file=sys.stdout)
                continue
        # only return reports if requested
        if self.report:
            results['report_image'] = self.report_files

        return results

    def find_wedges(self, img, mask, res):
        # X COORDINATES
        x_investigate_region = np.ceil(35 / res[0]).astype(int)  # define width of region to test (comparable to wedges)

        if np.mod(x_investigate_region, 2) == 0:
            # we want an odd number to see -N to N points in the x direction
            x_investigate_region = x_investigate_region + 1

        w_point = np.argwhere(np.sum(mask, 0) > 0)[0].item()  # westmost point of object
        e_point = np.argwhere(np.sum(mask, 0) > 0)[-1].item()  # eastmost point of object
        n_point = np.argwhere(np.sum(mask, 1) > 0)[0].item()  # northmost point of object

        invest_x = []
        for k in range(x_investigate_region):
            y_loc = n_point + k  # add n_point to ensure in image's coordinate system
            t = mask[y_loc, np.arange(w_point, e_point + 1, 1)]  # mask for resultant line profile
            # line profile at varying y positions from west to east
            line_prof_x = skimage.measure.profile_line(img, (y_loc, w_point),
                                                       (y_loc, e_point), mode='constant').flatten()

            invest_x.append(t * line_prof_x)  # mask unwanted values out and append

        invest_x = np.array(invest_x).T  # transpose array
        mean_x_profile = np.mean(invest_x, 1)  # mean of horizontal projections of phantom
        abs_diff_x_profile = np.abs(np.diff(mean_x_profile))  # absolute first derivative of mean

        x_peaks, _ = self.ACR_obj.find_n_highest_peaks(abs_diff_x_profile, 2)  # find two highest peaks
        x_locs = w_point + x_peaks  # x coordinates of these peaks in image coordinate system(before diff operation)

        width_pts = [x_locs[0], x_locs[1]]  # width of wedges
        width = np.max(width_pts) - np.min(width_pts)  # width

        # rough midpoints of wedges
        x_pts = np.round([np.min(width_pts) + 0.25 * width, np.max(width_pts) - 0.25 * width]).astype(int)

        # Y COORDINATES
        # define height of region to test (comparable to wedges)
        y_investigate_region = int(np.ceil(20 / res[1]).item())

        # supposed distance from top of phantom to end of wedges
        end_point = n_point + np.round(50 / res[1]).astype(int)

        if np.mod(y_investigate_region, 2) == 0:
            # we want an odd number to see -N to N points in the y direction
            y_investigate_region = (y_investigate_region + 1)

        invest_y = []
        for m in range(y_investigate_region):
            x_loc = (m - np.floor(y_investigate_region / 2) + np.floor(np.mean(x_pts))).astype(int)
            c = mask[np.arange(n_point, end_point + 1, 1), x_loc]  # mask for resultant line profile
            line_prof_y = skimage.measure.profile_line(img, (n_point, x_loc), (end_point, x_loc),
                                                       mode='constant').flatten()
            invest_y.append(c * line_prof_y)

        invest_y = np.array(invest_y).T  # transpose array
        mean_y_profile = np.mean(invest_y, 1)  # mean of vertical projections of phantom
        abs_diff_y_profile = np.abs(np.diff(mean_y_profile))  # absolute first derivative of mean

        y_peaks, _ = self.ACR_obj.find_n_highest_peaks(abs_diff_y_profile, 2)  # find two highest peaks
        y_locs = w_point + y_peaks - 1  # y coordinates of these peaks in image coordinate system(before diff operation)

        if y_locs[1] - y_locs[0] < 5 / res[1]:
            y = [n_point + round(10 / res[1])]  # if peaks too close together, use phantom geometry
        else:
            y = np.round(np.min(y_locs) + 0.25 * np.abs(np.diff(y_locs)))  # define y coordinate

        dist_to_y = np.abs(n_point - y[0]) * res[1]  # distance to y from top of phantom
        y_pts = np.append(y, np.round(y[0] + (47 - dist_to_y) / res[1])).astype(
            int)  # place 2nd y point 47mm from top of phantom

        return x_pts, y_pts

    def get_slice_position(self, dcm):
        img = dcm.pixel_array
        res = dcm.PixelSpacing  # In-plane resolution from metadata
        mask = self.ACR_obj.mask_image
        x_pts, y_pts = self.find_wedges(img, mask, res)

        line_prof_L = skimage.measure.profile_line(img, (y_pts[0], x_pts[0]), (y_pts[1], x_pts[0]),
                                                   mode='constant').flatten()  # line profile through left wedge
        line_prof_R = skimage.measure.profile_line(img, (y_pts[0], x_pts[1]), (y_pts[1], x_pts[1]),
                                                   mode='constant').flatten()  # line profile through right wedge

        interp_factor = 5
        x = np.arange(1, len(line_prof_L) + 1)
        new_x = np.arange(1, len(line_prof_L) + (1 / interp_factor), (1 / interp_factor))

        interp_line_prof_L = scipy.interpolate.interp1d(x, line_prof_L)(new_x)  # interpolate left line profile
        interp_line_prof_R = scipy.interpolate.interp1d(x, line_prof_R)(new_x)  # interpolate right line profile

        delta = interp_line_prof_L - interp_line_prof_R  # difference of line profiles
        peaks, _ = ACRObject.find_n_highest_peaks(abs(delta), 2, 0.5 * np.max(abs(delta)))  # find two highest peaks

        if len(peaks) == 1:
            peaks = [peaks[0] - 50, peaks[0] + 50]  # if only one peak, set dummy range

        # set multiplier for right or left shift based on sign of peak
        pos = 1 if np.max(-delta[peaks[0]:peaks[1]]) < np.max(delta[peaks[0]:peaks[1]]) else -1

        # take line profiles in range of interest
        static_line_L = interp_line_prof_L[peaks[0]:peaks[1]]
        static_line_R = interp_line_prof_R[peaks[0]:peaks[1]]

        lag = np.linspace(-50, 50, 101, dtype=int)  # create array of lag values
        err = np.zeros(len(lag))  # initialise array of errors

        for k, lag_val in enumerate(lag):
            difference = static_line_R - np.roll(static_line_L, lag_val)  # difference of L and circularly shifted R
            # set wrapped values to nan
            if lag_val > 0:
                difference[:lag_val] = np.nan
            else:
                difference[lag_val:] = np.nan

            # filler value to suppress warning when trying to calculate mean of array filled with NaN otherwise
            # calculate difference
            err[k] = 1e10 if np.isnan(difference).all() else np.nanmean(difference)

        temp = np.argwhere(err == np.min(err[err > 0]))[0]  # find minimum non-zero error
        shift = -lag[temp][0] if pos == 1 else lag[temp][0]  # find shift corresponding to above error

        dL = pos * np.abs(shift) * (1 / interp_factor) * res[1]  # calculate bar length difference

        if self.report:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(4, 1)
            fig.set_size_inches(8, 32)
            fig.tight_layout(pad=4)

            axes[0].imshow(mask)
            axes[0].axis('off')
            axes[0].set_title('Thresholding Result')

            axes[1].imshow(img)
            axes[1].plot([x_pts[0], x_pts[0]], [y_pts[0], y_pts[1]], 'b')
            axes[1].plot([x_pts[1], x_pts[1]], [y_pts[0], y_pts[1]], 'r')
            axes[1].axis('off')
            axes[1].set_title('Line Profiles')

            axes[2].grid()
            axes[2].plot(
                (1 / interp_factor) * np.linspace(1, len(interp_line_prof_L), len(interp_line_prof_L)) * res[1],
                interp_line_prof_L, 'b')
            axes[2].plot(
                (1 / interp_factor) * np.linspace(1, len(interp_line_prof_R), len(interp_line_prof_R)) * res[1],
                interp_line_prof_R, 'r')
            axes[2].set_title('Original Line Profiles')
            axes[2].set_xlabel('Relative Pixel Position (mm)')

            axes[3].plot((1 / interp_factor) * np.linspace(1, len(interp_line_prof_L), len(interp_line_prof_L)) * res[1],
                     interp_line_prof_L, 'b')

            shift_line = np.roll(interp_line_prof_R, pos * shift)
            if shift < 0 and pos == -1:
                shift_line[0:np.abs(shift)] = np.nan
            elif shift < 0 and pos == 1:
                shift_line[pos * shift:] = np.nan
            elif shift > 0 and pos == -1:
                shift_line[pos * shift:] = np.nan
            else:
                shift_line[0:np.abs(pos) * shift] = np.nan

            axes[3].grid()
            axes[3].plot((1 / interp_factor) * np.linspace(1, len(interp_line_prof_L), len(interp_line_prof_L)) * res[1],
                     shift_line, 'r')
            axes[3].set_title('Shifted Line Profiles')
            axes[3].set_xlabel('Relative Pixel Position (mm)')

            img_path = os.path.realpath(os.path.join(
                self.report_path, f'{self.img_desc(dcm)}.png'))
            fig.savefig(img_path)
            self.report_files.append(img_path)

        return dL
