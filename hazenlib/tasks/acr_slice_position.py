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

Refactored by Luis M. Santos
luis.santos2@nih.gov

07/02/2025
"""

import os
import sys
import traceback

import numpy as np
import scipy
import skimage.measure
import skimage.morphology
from hazenlib.ACRObject import ACRObject
from hazenlib.HazenTask import HazenTask
from hazenlib.logger import logger
from hazenlib.types import Measurement, Result


class ACRSlicePosition(HazenTask):
    """Slice position measurement class for DICOM images of the ACR phantom."""

    def __init__(self, **kwargs):
        if kwargs.pop("verbose", None) is not None:
            logger.warning(
                "verbose is not a supported argument for %s",
                type(self).__name__,
            )
        super().__init__(**kwargs)
        # Initialise ACR object
        self.ACR_obj = ACRObject(self.dcm_list)
        self.Y_WEDGE_OFFSET = int(20 / self.ACR_obj.dx)
        self.WEDGE_CENTER_Y = int(
            45 / self.ACR_obj.dx
        )  # Center region where it is safe to ray trace along x coordinate
        self.X_WEDGE_OFFSET = int(3 / self.ACR_obj.dx)
        self.GAUSSIAN_SIGMA = 2.5 / self.ACR_obj.dx
        self.MINIMUM_Y_PEAK_THRESHOLD = int(20 / self.ACR_obj.dx)
        self.INTERPOLATION_FACTOR = 0.2

    def run(self) -> Result:
        """Main function for performing slice position measurement
        using the first and last slices from the ACR phantom image set.

        Returns:
            dict: results are returned in a standardised dictionary structure specifying the task name, input DICOM Series Description + SeriesNumber + InstanceNumber, task measurement key-value pairs, optionally path to the generated images for visualisation
        """
        # Identify relevant slices
        dcms = [self.ACR_obj.slice_stack[0], self.ACR_obj.slice_stack[-1]]

        # Initialise results dictionary
        results = self.init_result_dict(desc=self.ACR_obj.acquisition_type())
        results.files = [self.img_desc(dcm) for dcm in dcms]

        for dcm in dcms:
            try:
                result = self.get_slice_position(dcm)
                results.add_measurement(
                    Measurement(
                        name="SlicePosition",
                        type="measured",
                        subtype="length difference",
                        description=self.img_desc(dcm),
                        value=round(result, 2),
                    ),
                )

            except Exception as e:
                logger.exception(
                    "Could not calculate the bar length difference for %s"
                    " because of : %s",
                    self.img_desc(dcm),
                    e,
                )
                traceback.print_exc(file=sys.stdout)
                continue

        # only return reports if requested
        if self.report:
            results.add_report_image(self.report_files)

        return results

    def write_report(
        self,
        dcm,
        img,
        center,
        x_pts,
        y_pts,
        interp_line_prof_L,
        interp_line_prof_R,
        interp_factor,
        pos,
        shift,
    ):
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1)
        fig.set_size_inches(8, 32)
        fig.tight_layout(pad=4)

        axes[0].imshow(img)
        axes[0].scatter(center[0], center[1], c="red")
        axes[0].plot([x_pts[0], x_pts[0]], [y_pts[0], y_pts[1]], "b")
        axes[0].plot([x_pts[1], x_pts[1]], [y_pts[0], y_pts[1]], "r")
        axes[0].axis("off")
        axes[0].set_title("Line Profiles")

        axes[1].grid()
        axes[1].plot(
            interp_factor
            * np.linspace(1, len(interp_line_prof_L), len(interp_line_prof_L))
            * self.ACR_obj.dy,
            interp_line_prof_L,
            "b",
        )
        axes[1].plot(
            interp_factor
            * np.linspace(1, len(interp_line_prof_R), len(interp_line_prof_R))
            * self.ACR_obj.dy,
            interp_line_prof_R,
            "r",
        )
        axes[1].set_title("Original Line Profiles")
        axes[1].set_xlabel("Relative Pixel Position (mm)")

        axes[2].plot(
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

        axes[2].grid()
        axes[2].plot(
            interp_factor
            * np.linspace(1, len(interp_line_prof_L), len(interp_line_prof_L))
            * self.ACR_obj.dy,
            shift_line,
            "r",
        )
        axes[2].set_title("Shifted Line Profiles")
        axes[2].set_xlabel("Relative Pixel Position (mm)")

        img_path = os.path.realpath(
            os.path.join(self.report_path, f"{self.img_desc(dcm)}.png")
        )
        fig.savefig(img_path)
        self.report_files.append(img_path)

        return img_path

    def find_wedges(self, img, center):
        """Find wedges in the pixel array. \n
        Investigates the top half of the phantom to locate where the wedges
        pass through the slice, and calculates the co-ordinates of these locations.

        Args:
            img (np.ndarray): dcm.pixel_array
            center (tuple[int]): dcm.pixel_array of the image mask

        Returns:
            tuple of tuples: of x and y coordinates of wedges.
        """
        # Raytrace upward to find transition zone of wedge
        yray = skimage.measure.profile_line(
            img,
            (0, center[0]),
            (center[1], center[0]),
            linewidth=int(1 / self.ACR_obj.dx),
            mode="constant",
            reduce_func=np.mean,
        ).flatten()
        abs_diff_y_profile = np.abs(np.diff(yray))
        smoothed_y_profile = scipy.ndimage.gaussian_filter1d(
            abs_diff_y_profile, self.GAUSSIAN_SIGMA
        )

        ypeaks = self.ACR_obj.find_n_highest_peaks(smoothed_y_profile, 5)
        # A properly centered phantom will not have any signal in upper region (0 < y <~20)
        # Very small hack to filter out any "peak" signal in that region.
        # This is to make sure that phantoms with a well demarcated upper edge of the wedge do not bias the line profile
        # placement towards the upper portion of the wedge box.
        ypeaks = [p for p in ypeaks[0] if p > self.MINIMUM_Y_PEAK_THRESHOLD]
        # The line profile should yield a minimum of 4 peaks.
        # The first peak denotes the start of the phantom.
        # The second peak denotes the end of the wedge region.
        # The third and fourth peaks are phantom specific since
        # they imply the edges of elements around the center
        y_center = ypeaks[1]

        # Raytrace sideways assuming static general center of wedge to detect side transitions of the wedge box.
        # The more miscentered the phantom, the more likely to get the wrong x coordinates of the wedges, but
        # the static center was selected such that other relative phantom features do not interfere unless
        # the phantom is in a really bad location.
        # Small jitters should not have as much of an impact as relative computation of this center experienced.
        # As an added benefit, we should still get a decently accurate result even if we capture the upper bound of
        # the wedges with line profiles because the delta is based on the bottom differences.
        xray = skimage.measure.profile_line(
            img,
            (self.WEDGE_CENTER_Y, 0),
            (self.WEDGE_CENTER_Y, img.shape[0]),
            linewidth=int(1 / self.ACR_obj.dx),
            mode="constant",
            reduce_func=np.mean,
        ).flatten()
        abs_diff_x_profile = np.abs(np.diff(xray))
        smoothed_x_profile = scipy.ndimage.gaussian_filter1d(
            abs_diff_x_profile, self.GAUSSIAN_SIGMA
        )

        xpeaks = self.ACR_obj.find_n_highest_peaks(smoothed_x_profile, 4)
        # The line profile should yield exactly 4 peaks unless we are off-centered.
        # The middle two peaks correspond to the edges of the wedge region on the horizontal plane.
        # The other two peaks denote the edges of the Phantom intersected on the horizontal plane.
        # A simple average of the middle 2 peaks should give the exact x center coordinate of the wedge region.
        x_center = int(np.sum(xpeaks[0][1:3]) // 2)

        logger.info("Wedge Bottom => {}, {}".format(x_center, y_center))

        return [
            int(x_center - self.X_WEDGE_OFFSET),
            int(x_center + self.X_WEDGE_OFFSET),
        ], [
            int(y_center - self.Y_WEDGE_OFFSET),
            int(y_center + self.Y_WEDGE_OFFSET),
        ]

    def get_slice_position(self, dcm):
        """Measure slice position. \n
        Locates the two opposing wedges and calculates the height difference.

        Args:
            dcm (pydicom.Dataset): DICOM image object.

        Returns:
            float: bar length difference.
        """
        logger.info(
            f"Computing slice position for slice #{dcm.InstanceNumber}"
        )
        img, rescaled, presentation = self.ACR_obj.get_presentation_pixels(dcm)
        cxy, _ = self.ACR_obj.find_phantom_center(
            rescaled, self.ACR_obj.dx, self.ACR_obj.dy
        )
        x_pts, y_pts = self.find_wedges(rescaled, cxy)
        logger.info(
            "Wedge Locations => \n{}\n{}".format(
                (x_pts[0], y_pts[0]), (x_pts[1], y_pts[1])
            )
        )

        # line profile through left wedge
        line_prof_L = skimage.measure.profile_line(
            rescaled,
            (y_pts[0], x_pts[0]),
            (y_pts[1], x_pts[0]),
            mode="constant",
        ).flatten()
        # line profile through right wedge
        line_prof_R = skimage.measure.profile_line(
            rescaled,
            (y_pts[0], x_pts[1]),
            (y_pts[1], x_pts[1]),
            mode="constant",
        ).flatten()

        # interpolation
        x = np.arange(1, len(line_prof_L) + 1)
        new_x = np.arange(
            1,
            len(line_prof_L) + self.INTERPOLATION_FACTOR,
            self.INTERPOLATION_FACTOR,
        )

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
        logger.info(peaks)

        # if only one peak, set dummy range
        if len(peaks) == 1:
            peaks = [peaks[0] - 50, peaks[0] + 50]

        # set multiplier for right or left shift based on sign of peak
        pos = (
            1
            if np.max(-delta[peaks[0] : peaks[1]])
            < np.max(delta[peaks[0] : peaks[1]])
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
            err[k] = (
                1e10 if np.isnan(difference).all() else np.nanmean(difference)
            )

        # find minimum non-zero error
        temp = np.argwhere(err == np.min(err[err > 0]))[0]

        # find shift corresponding to above error
        shift = -lag[temp][0] if pos == 1 else lag[temp][0]

        # calculate bar length difference
        dL = pos * np.abs(shift) * self.INTERPOLATION_FACTOR * self.ACR_obj.dy

        if self.report:
            self.write_report(
                dcm,
                rescaled,
                cxy,
                x_pts,
                y_pts,
                interp_line_prof_L,
                interp_line_prof_R,
                self.INTERPOLATION_FACTOR,
                pos,
                shift,
            )

        return dL
