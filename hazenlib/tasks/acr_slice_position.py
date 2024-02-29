"""
ACR Slice Position

https://www.acraccreditation.org/-/media/acraccreditation/documents/mri/largephantomguidance.pdf

Calculates the bar length difference for slices 1 and 11 of the ACR phantom.

This script calculates the bar length difference in accordance with the ACR Guidance. Line profiles are drawn
vertically through the left and right wedges. The right wedge's line profile is shifted and wrapped round before being
subtracted from the left wedge's line profile, e.g.:

Right line profile: [1, 2, 3, 4, 5] \n
Right line profile wrapped round by 1: [2, 3, 4, 5, 1]

This wrapping process, from hereon referred to as 'circular shifting', is then used for subtractions.

The shift used to produce the minimum difference between the circularly shifted right line profile and the static left
one is used to determine the bar length difference, which is twice the slice position displacement. \n
The results are also visualised.

Created by Yassine Azma
yassine.azma@rmh.nhs.uk

28/12/2022
"""

import os
import sys
import traceback
import numpy as np

import scipy
import skimage.morphology
import skimage.measure

from hazenlib.HazenTask import HazenTask
from hazenlib.ACRObject import ACRObject


class ACRSlicePosition(HazenTask):
    """Slice position measurement class for DICOM images of the ACR phantom."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialise ACR object
        self.ACR_obj = ACRObject(self.dcm_list)

    def run(self) -> dict:
        """Main function for performing slice position measurement
        using the first and last slices from the ACR phantom image set.

        Returns:
            dict: results are returned in a standardised dictionary structure specifying the task name, input DICOM Series Description + SeriesNumber + InstanceNumber, task measurement key-value pairs, optionally path to the generated images for visualisation
        """
        # Identify relevant slices
        dcms = [self.ACR_obj.slice_stack[0], self.ACR_obj.slice_stack[-1]]

        # Initialise results dictionary
        results = self.init_result_dict()
        results["file"] = [self.img_desc(dcm) for dcm in dcms]
        results["measurement"] = {}

        for dcm in dcms:
            try:
                result = self.get_slice_position(dcm)
                results["measurement"][self.img_desc(dcm)] = {
                    "length difference": round(result, 2)
                }
            except Exception as e:
                print(
                    f"Could not calculate the bar length difference for {self.img_desc(dcm)} because of : {e}"
                )
                traceback.print_exc(file=sys.stdout)
                continue

        # only return reports if requested
        if self.report:
            results["report_image"] = self.report_files

        return results

    def find_wedges(self, img, mask):
        """Find wedges in the pixel array. \n
        Investigates the top half of the phantom to locate where the wedges
        pass through the slice, and calculates the co-ordinates of these locations.

        Args:
            img (np.ndarray): dcm.pixel_array
            mask (np.ndarray): dcm.pixel_array of the image mask

        Returns:
            tuple of tuples: of x and y coordinates of wedges.
        """
        # X COORDINATES
        x_investigate_region = np.ceil(35 / self.ACR_obj.dx).astype(
            int
        )  # define width of region to test (comparable to wedges)

        if np.mod(x_investigate_region, 2) == 0:
            # we want an odd number to see -N to N points in the x direction
            x_investigate_region = x_investigate_region + 1

        # westmost point of object
        w_point = np.argwhere(np.sum(mask, 0) > 0)[0].item()
        # eastmost point of object
        e_point = np.argwhere(np.sum(mask, 0) > 0)[-1].item()
        # northmost point of object
        n_point = np.argwhere(np.sum(mask, 1) > 0)[0].item()

        invest_x = []
        for k in range(x_investigate_region):
            # add n_point to ensure in image's coordinate system
            y_loc = n_point + k
            # mask for resultant line profile
            t = mask[y_loc, np.arange(w_point, e_point + 1, 1)]
            # line profile at varying y positions from west to east
            line_prof_x = skimage.measure.profile_line(
                img, (y_loc, w_point), (y_loc, e_point), mode="constant"
            ).flatten()

            # mask unwanted values out and append
            invest_x.append(t * line_prof_x)

        # transpose array
        invest_x = np.array(invest_x).T
        # mean of horizontal projections of phantom
        mean_x_profile = np.mean(invest_x, 1)
        # absolute first derivative of mean
        abs_diff_x_profile = np.abs(np.diff(mean_x_profile))

        # find two highest peaks
        x_peaks, _ = self.ACR_obj.find_n_highest_peaks(abs_diff_x_profile, 2)
        # x coordinates of these peaks in image coordinate system(before diff operation)
        x_locs = w_point + x_peaks

        # width of wedges
        width_pts = [x_locs[0], x_locs[1]]
        # width
        width = np.max(width_pts) - np.min(width_pts)

        # rough midpoints of wedges
        x_pts_left = round(np.min(width_pts) + 0.25 * width)
        x_pts_right = round(np.max(width_pts) - 0.25 * width)

        # Y COORDINATES
        # define height of region to test (comparable to wedges)
        y_investigate_region = int(np.ceil(20 / self.ACR_obj.dy).item())

        # supposed distance from top of phantom to end of wedges
        end_point = n_point + np.round(50 / self.ACR_obj.dy).astype(int)

        if np.mod(y_investigate_region, 2) == 0:
            # we want an odd number to see -N to N points in the y direction
            y_investigate_region = y_investigate_region + 1

        invest_y = []
        for m in range(y_investigate_region):
            x_loc = (
                m
                - np.floor(y_investigate_region / 2)
                + np.floor(np.mean([x_pts_left, x_pts_right]))
            ).astype(int)
            # mask for resultant line profile
            c = mask[np.arange(n_point, end_point + 1, 1), x_loc]
            line_prof_y = skimage.measure.profile_line(
                img, (n_point, x_loc), (end_point, x_loc), mode="constant"
            ).flatten()
            invest_y.append(c * line_prof_y)

        # transpose array
        invest_y = np.array(invest_y).T
        # mean of vertical projections of phantom
        mean_y_profile = np.mean(invest_y, 1)
        # absolute first derivative of mean
        abs_diff_y_profile = np.abs(np.diff(mean_y_profile))

        # find two highest peaks
        y_peaks, _ = self.ACR_obj.find_n_highest_peaks(abs_diff_y_profile, 2)
        # y coordinates of these peaks in image coordinate system(before diff operation)
        y_locs = w_point + y_peaks - 1

        if y_locs[1] - y_locs[0] < 5 / self.ACR_obj.dy:
            # if peaks too close together, use phantom geometry
            y = [n_point + round(10 / self.ACR_obj.dy)]
        else:
            # define y coordinate
            y = np.round(np.min(y_locs) + 0.25 * np.abs(np.diff(y_locs))).flatten()

        # distance to y from top of phantom
        dist_to_y = np.abs(n_point - y[0]) * self.ACR_obj.dy
        # place 2nd y point 47mm from top of phantom
        y_pt = round(y[0] + (47 - dist_to_y) / self.ACR_obj.dy)

        return [x_pts_left, x_pts_right], [y[0], y_pt]

    def get_slice_position(self, dcm):
        """Measure slice position. \n
        Locates the two opposing wedges and calculates the height difference.

        Args:
            dcm (pydicom.Dataset): DICOM image object.

        Returns:
            float: bar length difference.
        """
        img = dcm.pixel_array
        mask = self.ACR_obj.get_mask_image(img)
        x_pts, y_pts = self.find_wedges(img, mask)

        # line profile through left wedge
        line_prof_L = skimage.measure.profile_line(
            img, (y_pts[0], x_pts[0]), (y_pts[1], x_pts[0]), mode="constant"
        ).flatten()
        # line profile through right wedge
        line_prof_R = skimage.measure.profile_line(
            img, (y_pts[0], x_pts[1]), (y_pts[1], x_pts[1]), mode="constant"
        ).flatten()

        # interpolation
        interp_factor = 1 / 5
        x = np.arange(1, len(line_prof_L) + 1)
        new_x = np.arange(1, len(line_prof_L) + interp_factor, interp_factor)

        # interpolate left line profile
        interp_line_prof_L = scipy.interpolate.interp1d(x, line_prof_L)(new_x)

        # interpolate right line profile
        interp_line_prof_R = scipy.interpolate.interp1d(x, line_prof_R)(new_x)

        # difference of line profiles
        delta = interp_line_prof_L - interp_line_prof_R
        # find two highest peaks
        peaks, _ = ACRObject.find_n_highest_peaks(
            abs(delta), 2, 0.5 * np.max(abs(delta))
        )

        # if only one peak, set dummy range
        if len(peaks) == 1:
            peaks = [peaks[0] - 50, peaks[0] + 50]

        # set multiplier for right or left shift based on sign of peak
        pos = (
            1
            if np.max(-delta[peaks[0] : peaks[1]]) < np.max(delta[peaks[0] : peaks[1]])
            else -1
        )

        # take line profiles in range of interest
        static_line_L = interp_line_prof_L[peaks[0] : peaks[1]]
        static_line_R = interp_line_prof_R[peaks[0] : peaks[1]]

        # create array of lag values
        lag = np.linspace(-50, 50, 101, dtype=int)

        # initialise array of errors
        err = np.zeros(len(lag))

        for k, lag_val in enumerate(lag):
            # difference of L and circularly shifted R
            difference = static_line_R - np.roll(static_line_L, lag_val)
            # set wrapped values to nan
            if lag_val > 0:
                difference[:lag_val] = np.nan
            else:
                difference[lag_val:] = np.nan

            # filler value to suppress warning when trying to calculate mean of array filled with NaN otherwise
            # calculate difference
            err[k] = 1e10 if np.isnan(difference).all() else np.nanmean(difference)

        # find minimum non-zero error
        temp = np.argwhere(err == np.min(err[err > 0]))[0]

        # find shift corresponding to above error
        shift = -lag[temp][0] if pos == 1 else lag[temp][0]

        # calculate bar length difference
        dL = pos * np.abs(shift) * interp_factor * self.ACR_obj.dy

        if self.report:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(4, 1)
            fig.set_size_inches(8, 32)
            fig.tight_layout(pad=4)

            axes[0].imshow(mask)
            axes[0].axis("off")
            axes[0].set_title("Thresholding Result")

            axes[1].imshow(img)
            axes[1].plot([x_pts[0], x_pts[0]], [y_pts[0], y_pts[1]], "b")
            axes[1].plot([x_pts[1], x_pts[1]], [y_pts[0], y_pts[1]], "r")
            axes[1].axis("off")
            axes[1].set_title("Line Profiles")

            axes[2].grid()
            axes[2].plot(
                interp_factor
                * np.linspace(1, len(interp_line_prof_L), len(interp_line_prof_L))
                * self.ACR_obj.dy,
                interp_line_prof_L,
                "b",
            )
            axes[2].plot(
                interp_factor
                * np.linspace(1, len(interp_line_prof_R), len(interp_line_prof_R))
                * self.ACR_obj.dy,
                interp_line_prof_R,
                "r",
            )
            axes[2].set_title("Original Line Profiles")
            axes[2].set_xlabel("Relative Pixel Position (mm)")

            axes[3].plot(
                interp_factor
                * np.linspace(1, len(interp_line_prof_L), len(interp_line_prof_L))
                * self.ACR_obj.dy,
                interp_line_prof_L,
                "b",
            )

            shift_line = np.roll(interp_line_prof_R, pos * shift)
            if shift < 0 and pos == -1:
                shift_line[0 : np.abs(shift)] = np.nan
            elif shift < 0 and pos == 1:
                shift_line[pos * shift :] = np.nan
            elif shift > 0 and pos == -1:
                shift_line[pos * shift :] = np.nan
            else:
                shift_line[0 : np.abs(pos) * shift] = np.nan

            axes[3].grid()
            axes[3].plot(
                interp_factor
                * np.linspace(1, len(interp_line_prof_L), len(interp_line_prof_L))
                * self.ACR_obj.dy,
                shift_line,
                "r",
            )
            axes[3].set_title("Shifted Line Profiles")
            axes[3].set_xlabel("Relative Pixel Position (mm)")

            img_path = os.path.realpath(
                os.path.join(self.report_path, f"{self.img_desc(dcm)}.png")
            )
            fig.savefig(img_path)
            self.report_files.append(img_path)

        return dL
