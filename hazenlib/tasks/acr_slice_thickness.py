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
from scipy.interpolate import interp1d
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
            """Calculates the FWHM by fitting a piecewise sigmoid to signal across line"""
            if not hasattr(self, "signal"):
                raise ValueError("Signal across line has not been computed!")

            fitted = self._fit_piecewise_sigmoid()

            peaks, props = find_peaks(fitted.y, height=0, prominence=np.max(fitted.y).item() / 4)
            peak_height = np.max(props["peak_heights"])

            backgroundL = fitted.y[0]
            backgroundR = fitted.y[-1]

            halfMaxL = (peak_height - backgroundL) / 2 + backgroundL
            halfMaxR = (peak_height - backgroundR) / 2 + backgroundR

            def simple_interpolate(targetY: float, signal: XY) -> float:
                crossing_index = np.where(signal.y > targetY)[0][0]
                x1, x2 = signal.x[crossing_index - 1], signal.x[crossing_index]
                y1, y2 = signal.y[crossing_index - 1], signal.y[crossing_index]
                targetX = x1 + (targetY - y1) * (x2 - x1) / (y2 - y1)
                return targetX

            xL = simple_interpolate(halfMaxL, fitted)
            xR = simple_interpolate(halfMaxR, fitted[:, ::-1])

            self.FWHM = xR - xL
            self.fitted = fitted

        def _fit_piecewise_sigmoid(self) -> XY:

            smoothed = self.signal.copy()
            k = round(len(smoothed.y) / 20)
            if k % 2 == 0:
                k += 1
            smoothed.y = medfilt(smoothed.y, k)
            smoothed.y = gaussian_filter1d(smoothed.y, round(k / 2.5))

            peaks, props = find_peaks(
                smoothed.y, height=0, prominence=np.max(smoothed.y).item() / 4
            )
            heights = props["peak_heights"]
            peak = peaks[np.argmax(heights)]

            def get_specific_sigmoid(wholeData: XY, fitStart: int, fitEnd: int) -> XY:
                fitData = wholeData[:, fitStart:fitEnd]
                A = np.max(fitData.y) - np.min(fitData.y)
                b = np.min(fitData.y)

                dy = np.diff(fitData.y)
                dx = np.diff(fitData.x)
                dx[dx == 0] = 1e-6
                absDeriv = np.abs(dy / dx)
                absGradMax = np.max(absDeriv)
                k = np.sign(fitData.y[-1] - fitData.y[0]) * absGradMax * 0.5
                x0 = fitData.x[np.argmax(absDeriv)]

                p0 = [A, k, x0, b]

                def sigmoid(x, A, k, x0, b):
                    exp_term = np.exp(-k * (x - x0))
                    return A / (1 + exp_term) + b

                popt, _ = curve_fit(sigmoid, fitData.x, fitData.y, p0=p0)

                def specific_sigmoid(x):
                    return sigmoid(x, *popt)

                return specific_sigmoid

            sigmoidL_func = get_specific_sigmoid(smoothed, 0, peak)
            sigmoidR_func = get_specific_sigmoid(smoothed, peak, len(smoothed.x))

            sigmoidL = XY(smoothed.x, sigmoidL_func(smoothed.x))
            sigmoidR = XY(smoothed.x, sigmoidR_func(smoothed.x))

            def blending_weight(x, transition_x, transition_width):
                return 1 / (1 + np.exp(-(x - transition_x) / transition_width))

            W = blending_weight(smoothed.x, peak, 1 / 20 * peak + 1 / 20 * (len(smoothed.x) - peak))
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
        if ori == "Sagittal":
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
        slice_thickness = 0.2 * (lines[0].FWHM * lines[1].FWHM) / (lines[0].FWHM + lines[1].FWHM)

        if self.report:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 3, figsize=(16, 8))
            axes[0].imshow(img)
            for i, line in enumerate(lines):
                axes[0].plot([line.start.x, line.end.x], [line.start.y, line.end.y])
                axes[i + 1].plot(
                    line.signal.x, line.signal.y, label="Raw signal", alpha=0.25, color=f"C{i}"
                )
                axes[i + 1].plot(
                    line.fitted.x, line.fitted.y, label="Fitted piecewise sigmoid", color=f"C{i}"
                )
                axes[i + 1].legend(loc="lower right", bbox_to_anchor=(1, -0.2))

            axes[0].axis("off")

            axes[0].set_title("Plot showing placement of profile lines.")
            axes[1].set_title("Pixel profile across blue line.")
            axes[1].set_xlabel("Distance along blue line (pixels)")
            axes[1].set_ylabel("Pixel value")

            axes[2].set_title("Pixel profile across orange line.")
            axes[2].set_xlabel("Distance along orange line (pixels)")
            axes[2].set_ylabel("Pixel value")
            plt.tight_layout()

            img_path = os.path.realpath(
                os.path.join(self.report_path, f"{self.img_desc(dcm)}_slice_thickness.png")
            )

            fig.savefig(img_path, dpi=600)
            self.report_files.append(img_path)

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
        for line in finalLines:
            line.get_signal(img)

        return finalLines


import matplotlib

matplotlib.use("inline")
root = Tk()
root.withdraw()
file_path = filedialog.askdirectory()
task = ACRSliceThickness(input_data=get_dicom_files(file_path), report=True)
stResult = task.run()
print(stResult)
pass
