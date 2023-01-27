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


def find_n_peaks(data, n, height=1):
    peaks = scipy.signal.find_peaks(data, height)
    pk_heights = peaks[1]['peak_heights']
    pk_ind = peaks[0]
    highest_peaks = pk_ind[(-pk_heights).argsort()[:n]]  # find n highest peaks

    return np.sort(highest_peaks)


class ACRSlicePosition(HazenTask):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self) -> dict:
        results = {}
        z = []
        for dcm in self.data:
            z.append(dcm.ImagePositionPatient[2])

        idx_sort = np.argsort(z)

        for dcm in self.data:
            curr_z = dcm.ImagePositionPatient[2]
            if curr_z in (z[idx_sort[0]], z[idx_sort[10]]):
                try:
                    result = self.get_slice_position(dcm)
                except Exception as e:
                    print(f"Could not calculate the bar length difference for {self.key(dcm)} because of : {e}")
                    traceback.print_exc(file=sys.stdout)
                    continue

                results[self.key(dcm)] = result

        results['reports'] = {'images': self.report_files}

        return results

    def centroid_com(self, dcm):
        # Calculate centroid of object using a centre-of-mass calculation
        thresh_img = dcm > 0.25 * np.max(dcm)
        open_img = skimage.morphology.area_opening(thresh_img, area_threshold=500)
        bhull = skimage.morphology.convex_hull_image(open_img)
        coords = np.nonzero(bhull)  # row major - first array is columns

        sum_x = np.sum(coords[1])
        sum_y = np.sum(coords[0])
        cxy = sum_x / coords[0].shape, sum_y / coords[1].shape

        cxy = [cxy[0].astype(int), cxy[1].astype(int)]
        return bhull, cxy

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

        x_peaks = find_n_peaks(abs_diff_x_profile, 2)  # find two highest peaks
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

        y_peaks = find_n_peaks(abs_diff_y_profile, 2)  # find two highest peaks
        y_locs = w_point + y_peaks - 1  # y coordinates of these peaks in image coordinate system(before diff operation)

        if y_locs[1] - y_locs[0] < 5 / res[1]:
            y = n_point + round(10 / res[1])  # if peaks too close together, use phantom geometry
        else:
            y = np.round(np.min(y_locs) + 0.25 * np.abs(np.diff(y_locs)))  # define y coordinate

        dist_to_y = np.abs(n_point - y[0]) * res[1]  # distance to y from top of phantom
        y_pts = np.append(y, np.round(y[0] + (47 - dist_to_y) / res[1])).astype(int)  # place 2nd y point 47mm from top of phantom

        return x_pts, y_pts

    def get_slice_position(self, dcm):
        img = dcm.pixel_array
        res = dcm.PixelSpacing  # In-plane resolution from metadata
        mask, cxy = self.centroid_com(img)
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
        peaks = find_n_peaks(abs(delta), 2, 0.5 * np.max(abs(delta)))  # find two highest peaks

        if len(peaks) == 1:
            peaks = [peaks[0] - 50, peaks[0] + 50]  # if only one peak, set dummy range

        # set multiplier for right or left shift based on sign of peak
        pos = 1 if np.max(-delta[peaks[0]:peaks[1]]) < np.max(delta[peaks[0]:peaks[1]]) else -1

        # take line profiles in range of interest
        static_line_L = interp_line_prof_L[peaks[0]:peaks[1]]
        static_line_R = interp_line_prof_R[peaks[0]:peaks[1]]

        lag = np.linspace(-50, 50, 101, dtype=int)  # create array of lag values
        err = np.zeros(len(lag))  # initialise array of errors

        for k in range(len(lag)):
            temp_lag = lag[k]  # set a shift value
            difference = static_line_R - np.roll(static_line_L, temp_lag)  # difference of L and circularly shifted R
            # set wrapped values to nan
            if temp_lag > 0:
                difference[:temp_lag] = np.nan
            else:
                difference[temp_lag:] = np.nan

            if np.isnan(difference).all():
                err[k] = 1e10  # filler value to suppress warning when trying to calculate mean of array filled with NaN
            else:
                err[k] = np.nanmean(difference)  # calculate mean difference ignoring nan values

        temp = np.argwhere(err == np.min(err[err > 0]))[0]  # find minimum non-zero error
        shift = -lag[temp][0] if pos == 1 else lag[temp][0]  # find shift corresponding to above error

        dL = np.round(pos * np.abs(shift) * (1 / interp_factor) * res[1], 2)  # calculate bar length difference

        if self.report:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            plt.suptitle('Bar Length Difference = ' + str(np.round(dL, 2)) + 'mm', x=0.5, ha='center')
            fig.set_size_inches(8, 8)

            plt.subplot(2, 2, (1, 3))
            plt.imshow(img)
            plt.plot([x_pts[0], x_pts[0]], [y_pts[0], y_pts[1]], 'b')
            plt.plot([x_pts[1], x_pts[1]], [y_pts[0], y_pts[1]], 'r')
            plt.axis('off')
            plt.tight_layout()

            plt.subplot(2, 2, 2)
            plt.grid()
            plt.plot((1 / interp_factor) * np.linspace(1, len(interp_line_prof_L), len(interp_line_prof_L)) * res[1],
                     interp_line_prof_L, 'b')
            plt.plot((1 / interp_factor) * np.linspace(1, len(interp_line_prof_R), len(interp_line_prof_R)) * res[1],
                     interp_line_prof_R, 'r')
            plt.title('Original Line Profiles')
            plt.xlabel('Relative Pixel Position (mm)')
            plt.tight_layout()

            plt.subplot(2, 2, 4)
            plt.grid()
            plt.plot((1 / interp_factor) * np.linspace(1, len(interp_line_prof_L), len(interp_line_prof_L)) * res[1],
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

            plt.plot((1 / interp_factor) * np.linspace(1, len(interp_line_prof_L), len(interp_line_prof_L)) * res[1],
                     shift_line, 'r')
            plt.title('Shifted Line Profiles')
            plt.xlabel('Relative Pixel Position (mm)')
            plt.tight_layout()

            img_path = os.path.realpath(os.path.join(self.report_path, f'{self.key(dcm)}.png'))
            fig.savefig(img_path)
            self.report_files.append(img_path)

        return dL
