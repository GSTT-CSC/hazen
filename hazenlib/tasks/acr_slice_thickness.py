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

sys.path.append(r"R:\Users Public\Students\Nathan Crossley\MRI\Hazen Project\hazen")
import traceback
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog

import scipy
from scipy.signal import find_peaks, medfilt
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
import skimage.morphology
import skimage.measure


from hazenlib.HazenTask import HazenTask
from hazenlib.ACRObject import ACRObject
from hazenlib.utils import get_image_orientation, get_dicom_files
from hazenlib.utils import Point, Line, XY


class ACRSliceThickness(HazenTask):
    """Slice width measurement class for DICOM images of the ACR phantom."""

    class SignalLine(Line):
        """Subclass of Line to implement functionality related to the ACR phantom's ramps"""

        def get_FWHM(self):

            if not hasattr(self, "signal"):
                raise ValueError("Signal across line has not been computed!")

            fitted = self._fit_piecewise_sigmoid()
            pass

        def _fit_piecewise_sigmoid(self) -> XY:

            smoothed = self.signal.copy()
            k = round(len(smoothed.y)/20)
            if k % 2 == 0:
                k += 1
            smoothed.y = medfilt(smoothed.y, k)
            smoothed.y = gaussian_filter1d(smoothed.y, round(k/2.5))

            peaks, props = find_peaks(smoothed.y, height=0, prominence=np.max(smoothed.y).item()/4)
            heights = props["peak_heights"]
            peak = peaks[np.argmax(heights)]

            def get_specific_sigmoid(wholeData: XY, fitStart: int, fitEnd:int) -> XY:
                fitData = wholeData[:, fitStart: fitEnd]
                A = np.max(fitData.y) - np.min(fitData.y)
                b = np.min(fitData.y)

                dy = np.diff(fitData.y)
                dx = np.diff(fitData.x)
                dx[dx == 0] = 1e-6
                absDeriv = np.abs(dy/dx)
                absGradMax = np.max(absDeriv)
                k = np.sign(fitData.y[-1] - fitData.y[0]) * absGradMax * 0.5
                x0 = fitData.x[np.argmax(absDeriv)]

                p0 = [A, k, x0, b]

                def sigmoid(x, A, k, x0, b):
                    exp_term = np.exp(-k*(x-x0))
                    return A/(1+exp_term) + b

                popt, _ = curve_fit(sigmoid, fitData.x, fitData.y, p0=p0)

                def specific_sigmoid(x):
                    return sigmoid(x, *popt)

                return specific_sigmoid

            sigmoidL_func = get_specific_sigmoid(smoothed, 0, peak)
            sigmoidR_func = get_specific_sigmoid(smoothed, peak, len(smoothed.x))

            sigmoidL = XY(smoothed.x, sigmoidL_func(smoothed.x))
            sigmoidR = XY(smoothed.x, sigmoidR_func(smoothed.x))

            def blending_weight(x, transition_x, transition_width):
                return 1/(1+np.exp(-(x-transition_x)/transition_width))

            W = blending_weight(smoothed.x, peak, 1/20 * peak + 1/20 * (len(smoothed.x) - peak))
            fitted = XY(smoothed.x, (1 - W) * sigmoidL.y + W * sigmoidR.y)

            return fitted

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

    def get_slice_thickness(self, dcm):
        """Measure slice thickness. \n
        Identify the ramps, measure the line profile, measure the FWHM, and use this to calculate the slice thickness.

        Args:
            dcm (pydicom.Dataset): DICOM image object.

        Returns:
            float: measured slice thickness.
        """
        img = dcm.pixel_array
        lines = self.place_lines(img)
        for line in lines:
            line.get_FWHM()
            pass

        return slice_thickness


    def place_lines(self, img: np.ndarray) -> list["Line"]:
        """Places line on image within ramps insert.
        Works for a rotated phantom.

        Args:
            img (np.ndarray): Pixel array from DICOM image.

        Returns:
            finalLines (list): A list of the two lines as Line objects.
        """
        # Normalize to uint8, enhance contast and binarize using otsu thresh

        img_uint8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        contrastEnhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3, 3)).apply(img_uint8)
        _, img_binary = cv2.threshold(contrastEnhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find contour by x-span sort
        contours, _ = cv2.findContours(
            img_binary.astype(np.uint8), mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE
        )
        contours_sorted = sorted(
            contours,
            key=lambda cont: abs(np.max(cont[:, 0, 0]) - np.min(cont[:, 0, 0])),
            reverse=True,
        )
        insertContour = contours_sorted[1]

        # Create list of Point objects for the four corners of the contour
        insertCorners = cv2.boxPoints(cv2.minAreaRect(insertContour))
        corners = [Point(*p) for p in insertCorners]

        # Define short sides of contours by list of line objects
        corners = sorted(corners, key=lambda point: corners[0].get_distance_to(point))
        shortSides = [Line(*corners[:2]), Line(*corners[2:])]

        # Get sublines of short sides and force start point to be higher in y
        sublines = [line.get_subline(perc=30) for line in shortSides]
        for line in sublines:
            if line.start.y < line.end.y:
                line.point_swap()

        # Define connecting lines
        connectingLines = [
            self.SignalLine(sublines[0].start, sublines[1].start),
            self.SignalLine(sublines[0].end, sublines[1].end),
        ]

        # Final lines are sublines of connecting lines
        finalLines = [line.get_subline(perc=95) for line in connectingLines]
        for line in finalLines: line.get_signal(img)

        return finalLines

    # def find_ramps(self, img, centre):
    #     """Find ramps in the pixel array and return the co-ordinates of their location.

    #     Args:
    #         img (np.ndarray): dcm.pixel_array
    #         centre (list): x,y coordinates of the phantom centre

    #     Returns:
    #         tuple: x and y coordinates of ramp.
    #     """
    #     # X
    #     investigate_region = int(np.ceil(5.5 / self.ACR_obj.dy).item())

    #     if np.mod(investigate_region, 2) == 0:
    #         investigate_region = investigate_region + 1

    #     # Line profiles around the central row
    #     invest_x = [
    #         skimage.measure.profile_line(
    #             img, (centre[1] + k, 1), (centre[1] + k, img.shape[1]), mode="constant"
    #         )
    #         for k in range(investigate_region)
    #     ]

    #     invest_x = np.array(invest_x).T
    #     mean_x_profile = np.mean(invest_x, 1)
    #     abs_diff_x_profile = np.absolute(np.diff(mean_x_profile))

    #     # find the points corresponding to the transition between:
    #     # [0] - background and the hyperintense phantom
    #     # [1] - hyperintense phantom and hypointense region with ramps
    #     # [2] - hypointense region with ramps and hyperintense phantom
    #     # [3] - hyperintense phantom and background

    #     x_peaks, _ = self.ACR_obj.find_n_highest_peaks(abs_diff_x_profile, 4)
    #     x_locs = np.sort(x_peaks) - 1

    #     width_pts = [x_locs[1], x_locs[2]]
    #     width = np.max(width_pts) - np.min(width_pts)

    #     # take rough estimate of x points for later line profiles
    #     x = np.round([np.min(width_pts) + 0.2 * width, np.max(width_pts) - 0.2 * width])

    #     # Y
    #     c = skimage.measure.profile_line(
    #         img,
    #         (centre[1] - 2 * investigate_region, centre[0]),
    #         (centre[1] + 2 * investigate_region, centre[0]),
    #         mode="constant",
    #     ).flatten()

    #     abs_diff_y_profile = np.absolute(np.diff(c))

    #     y_peaks, _ = self.ACR_obj.find_n_highest_peaks(abs_diff_y_profile, 2)
    #     y_locs = centre[1] - 2 * investigate_region + 1 + y_peaks
    #     height = np.max(y_locs) - np.min(y_locs)

    #     y = np.round([np.max(y_locs) - 0.25 * height, np.min(y_locs) + 0.25 * height])

    #     return x, y

    # def FWHM(self, data):
    #     """Calculate full width at half maximum of the line profile.

    #     Args:
    #         data (np.ndarray): slice profile curve.

    #     Returns:
    #         tuple: co-ordinates of the half-maximum points on the line profile.
    #     """
    #     baseline = np.min(data)
    #     data -= baseline
    #     # TODO create separate variable so that data value isn't being overwritten
    #     half_max = np.max(data) * 0.5

    #     # Naive attempt
    #     half_max_crossing_indices = np.argwhere(
    #         np.diff(np.sign(data - half_max))
    #     ).flatten()

    #     # Interpolation
    #     def simple_interp(x_start, ydata):
    #         """Simple interpolation - obtaining more accurate x co-ordinates.

    #         Args:
    #             x_start (int or float): x coordinate of the half maximum.
    #             ydata (np.ndarray): y coordinates.

    #         Returns:
    #             float: true x coordinate of the half maximum.
    #         """
    #         x_points = np.arange(x_start - 5, x_start + 6)
    #         # Check if expected x_pts (indices) will be out of range ( >= len(ydata))
    #         inrange = np.where(x_points == len(ydata))[0]
    #         if np.size(inrange) > 0:
    #             # locate index of where ydata ends within x_pts
    #             # crop x_pts until len(ydata)
    #             x_pts = x_points[: inrange.flatten()[0]]
    #         else:
    #             x_pts = x_points

    #         y_pts = ydata[x_pts]

    #         grad = (y_pts[-1] - y_pts[0]) / (x_pts[-1] - x_pts[0])

    #         x_true = x_start + (half_max - ydata[x_start]) / grad

    #         return x_true

    #     FWHM_pts = simple_interp(half_max_crossing_indices[0], data), simple_interp(
    #         half_max_crossing_indices[-1], data
    #     )
    #     return FWHM_pts



"""
import matplotlib
matplotlib.use("inline")
root = Tk()
root.withdraw()
file_path = filedialog.askdirectory()
task = ACRSliceThickness(input_data=get_dicom_files(file_path))
task.run()
"""
