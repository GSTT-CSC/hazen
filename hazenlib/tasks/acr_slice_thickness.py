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

import os
import sys
import traceback
import numpy as np

import scipy
import skimage.morphology
import skimage.measure

from hazenlib.HazenTask import HazenTask
from hazenlib.ACRObject import ACRObject
from hazenlib.utils import get_image_orientation



class ACRSliceThickness(HazenTask):
    """Slice width measurement class for DICOM images of the ACR phantom."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialise ACR object
        self.ACR_obj = ACRObject(self.dcm_list)

    def run(self) -> dict:
        """Main function for performing slice width measurement
        using slice 1 from the ACR phantom image set.

        Returns:
            dict: results are returned in a standardised dictionary structure specifying the task name, input DICOM Series Description + SeriesNumber + InstanceNumber, task measurement key-value pairs, optionally path to the generated images for visualisation.
        """
        # Identify relevant slice
        slice_thickness_dcm = self.ACR_obj.slice_stack[0]
        # TODO image may be 90 degrees cw or acw, could use code to identify which or could be added as extra arg

        ori = get_image_orientation(slice_thickness_dcm)
        if ori == 'Sagittal':
            # Get the pixel array from the DICOM file
            img = slice_thickness_dcm.pixel_array

            # Rotate the image 90 degrees clockwise

            rotated_img = np.rot90(img, k=-1)  # k=-1 for 90 degrees clockwise

            # Update the pixel array in the DICOM object
            slice_thickness_dcm.PixelData = rotated_img.tobytes()

        # Initialise results dictionary
        results = self.init_result_dict()
        results["file"] = self.img_desc(slice_thickness_dcm)

        try:
            result = self.get_slice_thickness(slice_thickness_dcm)
            results["measurement"] = {"slice width mm": round(result, 2)}
        except Exception as e:
            print(
                f"Could not calculate the slice thickness for {self.img_desc(slice_thickness_dcm)} because of : {e}"
            )
            traceback.print_exc(file=sys.stdout)

        # only return reports if requested
        if self.report:
            results["report_image"] = self.report_files

        return results

    def find_ramps(self, img, centre):
        """Find ramps in the pixel array and return the co-ordinates of their location.

        Args:
            img (np.ndarray): dcm.pixel_array
            centre (list): x,y coordinates of the phantom centre

        Returns:
            tuple: x and y coordinates of ramp.
        """
        # X
        investigate_region = int(np.ceil(5.5 / self.ACR_obj.dy).item())

        if np.mod(investigate_region, 2) == 0:
            investigate_region = investigate_region + 1

        # Line profiles around the central row
        invest_x = [
            skimage.measure.profile_line(
                img, (centre[1] + k, 1), (centre[1] + k, img.shape[1]), mode="constant"
            )
            for k in range(investigate_region)
        ]

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
        c = skimage.measure.profile_line(
            img,
            (centre[1] - 2 * investigate_region, centre[0]),
            (centre[1] + 2 * investigate_region, centre[0]),
            mode="constant",
        ).flatten()

        abs_diff_y_profile = np.absolute(np.diff(c))

        y_peaks, _ = self.ACR_obj.find_n_highest_peaks(abs_diff_y_profile, 2)
        y_locs = centre[1] - 2 * investigate_region + 1 + y_peaks
        height = np.max(y_locs) - np.min(y_locs)

        y = np.round([np.max(y_locs) - 0.25 * height, np.min(y_locs) + 0.25 * height])

        return x, y

    def FWHM(self, data):
        """Calculate full width at half maximum of the line profile.

        Args:
            data (np.ndarray): slice profile curve.

        Returns:
            tuple: co-ordinates of the half-maximum points on the line profile.
        """
        baseline = np.min(data)
        data -= baseline
        # TODO create separate variable so that data value isn't being overwritten
        half_max = np.max(data) * 0.5

        # Naive attempt
        half_max_crossing_indices = np.argwhere(
            np.diff(np.sign(data - half_max))
        ).flatten()

        # Interpolation
        def simple_interp(x_start, ydata):
            """Simple interpolation - obtaining more accurate x co-ordinates.

            Args:
                x_start (int or float): x coordinate of the half maximum.
                ydata (np.ndarray): y coordinates.

            Returns:
                float: true x coordinate of the half maximum.
            """
            x_points = np.arange(x_start - 5, x_start + 6)
            # Check if expected x_pts (indices) will be out of range ( >= len(ydata))
            inrange = np.where(x_points == len(ydata))[0]
            if np.size(inrange) > 0:
                # locate index of where ydata ends within x_pts
                # crop x_pts until len(ydata)
                x_pts = x_points[: inrange.flatten()[0]]
            else:
                x_pts = x_points

            y_pts = ydata[x_pts]

            grad = (y_pts[-1] - y_pts[0]) / (x_pts[-1] - x_pts[0])

            x_true = x_start + (half_max - ydata[x_start]) / grad

            return x_true

        FWHM_pts = simple_interp(half_max_crossing_indices[0], data), simple_interp(
            half_max_crossing_indices[-1], data
        )
        return FWHM_pts

    def get_slice_thickness(self, dcm):
        """Measure slice thickness. \n
        Identify the ramps, measure the line profile, measure the FWHM, and use this to calculate the slice thickness.

        Args:
            dcm (pydicom.Dataset): DICOM image object.

        Returns:
            float: measured slice thickness.
        """
        img = dcm.pixel_array
        cxy, _ = self.ACR_obj.find_phantom_center(img, self.ACR_obj.dx, self.ACR_obj.dy)
        x_pts, y_pts = self.find_ramps(img, cxy)

        interp_factor = 1 / 5
        interp_factor_dx = interp_factor * self.ACR_obj.dx
        sample = np.arange(1, x_pts[1] - x_pts[0] + 2)
        new_sample = np.arange(1, x_pts[1] - x_pts[0] + interp_factor, interp_factor)
        offsets = np.arange(-3, 4)
        ramp_length = np.zeros((2, 7))

        line_store = []
        fwhm_store = []
        for i, offset in enumerate(offsets):
            lines = [
                skimage.measure.profile_line(
                    img,
                    (offset + y_pts[0], x_pts[0]),
                    (offset + y_pts[0], x_pts[1]),
                    linewidth=2,
                    mode="constant",
                ).flatten(),
                skimage.measure.profile_line(
                    img,
                    (offset + y_pts[1], x_pts[0]),
                    (offset + y_pts[1], x_pts[1]),
                    linewidth=2,
                    mode="constant",
                ).flatten(),
            ]

            interp_lines = [
                scipy.interpolate.interp1d(sample, line)(new_sample) for line in lines
            ]
            fwhm = [self.FWHM(interp_line) for interp_line in interp_lines]
            ramp_length[0, i] = interp_factor_dx * np.diff(fwhm[0])
            ramp_length[1, i] = interp_factor_dx * np.diff(fwhm[1])

            line_store.append(interp_lines)
            fwhm_store.append(fwhm)

        with np.errstate(divide="ignore", invalid="ignore"):
            dz = 0.2 * (np.prod(ramp_length, axis=0)) / np.sum(ramp_length, axis=0)

        dz = dz[~np.isnan(dz)]
        # TODO check this - if it's taking the value closest to the DICOM slice thickness this is potentially not accurate?
        z_ind = np.argmin(np.abs(dcm.SliceThickness - dz))

        slice_thickness = dz[z_ind]

        if self.report:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(4, 1)
            fig.set_size_inches(8, 24)
            fig.tight_layout(pad=4)

            x_ramp = new_sample * self.ACR_obj.dx
            x_extent = np.max(x_ramp)
            y_ramp = line_store[z_ind][1]
            y_extent = np.max(y_ramp)
            max_loc = np.argmax(y_ramp) * interp_factor_dx

            axes[0].imshow(img)
            axes[0].scatter(cxy[0], cxy[1], c="red")
            axes[0].axis("off")
            axes[0].set_title("Centroid Location")

            axes[1].imshow(img)
            axes[1].plot(
                [x_pts[0], x_pts[1]], offsets[z_ind] + [y_pts[0], y_pts[0]], "b-"
            )
            axes[1].plot(
                [x_pts[0], x_pts[1]], offsets[z_ind] + [y_pts[1], y_pts[1]], "r-"
            )
            axes[1].axis("off")
            axes[1].set_title("Line Profiles")

            xmin = fwhm_store[z_ind][1][0] * interp_factor_dx / x_extent
            xmax = fwhm_store[z_ind][1][1] * interp_factor_dx / x_extent

            axes[2].plot(
                x_ramp,
                y_ramp,
                "r",
                label=f"FWHM={np.round(ramp_length[1][z_ind], 2)}mm",
            )
            axes[2].axhline(
                0.5 * y_extent, linestyle="dashdot", color="k", xmin=xmin, xmax=xmax
            )
            axes[2].axvline(
                max_loc, linestyle="dashdot", color="k", ymin=0, ymax=10 / 11
            )

            axes[2].set_xlabel("Relative Position (mm)")
            axes[2].set_xlim([0, x_extent])
            axes[2].set_ylim([0, y_extent * 1.1])
            axes[2].set_title("Upper Ramp")
            axes[2].grid()
            axes[2].legend(loc="best")

            xmin = fwhm_store[z_ind][0][0] * interp_factor_dx / x_extent
            xmax = fwhm_store[z_ind][0][1] * interp_factor_dx / x_extent
            x_ramp = new_sample * self.ACR_obj.dx
            x_extent = np.max(x_ramp)
            y_ramp = line_store[z_ind][0]
            y_extent = np.max(y_ramp)
            max_loc = np.argmax(y_ramp) * interp_factor_dx

            axes[3].plot(
                x_ramp,
                y_ramp,
                "b",
                label=f"FWHM={np.round(ramp_length[0][z_ind], 2)}mm",
            )
            axes[3].axhline(
                0.5 * y_extent, xmin=xmin, xmax=xmax, linestyle="dashdot", color="k"
            )
            axes[3].axvline(
                max_loc, ymin=0, ymax=10 / 11, linestyle="dashdot", color="k"
            )

            axes[3].set_xlabel("Relative Position (mm)")
            axes[3].set_xlim([0, x_extent])
            axes[3].set_ylim([0, y_extent * 1.1])
            axes[3].set_title("Lower Ramp")
            axes[3].grid()
            axes[3].legend(loc="best")

            img_path = os.path.realpath(
                os.path.join(
                    self.report_path, f"{self.img_desc(dcm)}_slice_thickness.png"
                )
            )
            fig.savefig(img_path)
            self.report_files.append(img_path)

        return slice_thickness
