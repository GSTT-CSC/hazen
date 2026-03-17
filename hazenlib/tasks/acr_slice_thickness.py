"""
ACR Slice Thickness
___________________

Reference
_________

`ACR Large Phantom Guidance PDF <https://accreditationsupport.acr.org/helpdesk/attachments/11093487417>`_

Intro
_____

The slice thickness accuracy test assesses the accuracy with which a slice of specified thickness is achieved.
The prescribed slice thickness is compared with the measured slice thickness.

The ramps appear in a structure called the slice thickness insert. Figure 10 shows an image of slice 1 with
the slice thickness insert and signal ramps identified. The two ramps are crossed: one has a negative slope
and the other a positive slope with respect to the plane of slice 1. They are produced by cutting 1 mm wide
slots in a block of plastic. The slots are open to the interior of the phantom and are filled with the same
solution that fills the bulk of the phantom.

The signal ramps have a slope of 10 to 1 with respect to the plane of slice 1, which is an angle of about 5.71°.
Therefore, the signal ramps will appear in the image of slice 1 with a length that is 10 times the thickness of
the slice. If the phantom is misaligned from right-left, one ramp will appear longer than the other. The
crossed ramps allow for correction of the error introduced by right-left misalignment, and the slice thickness
formula takes that into account.

ACR Guidelines
______________

ACR Algorithm
+++++++++++++

    #. Display slice 1, and magnify the image by a factor of 2 to 4, keeping the slice thickness insert fully
        visible on the screen.
    #. Adjust the display level so that the signal ramps are well visualized.
        *. The ramp signal is much lower than that of surrounding water, so usually it will be necessary
            to lower the display level substantially and narrow the window.
    #. Place a rectangular ROI at the middle of each ramp as shown below in Figure 11.
        *. Note the mean signal values for each of these two ROIs and then average those values.
        *. The result is a number approximating the mean signal in the middle of the ramps.
        *. An elliptical ROI may be used if a rectangular one is unavailable.
    #. Lower the display level to half of the average ramp signal calculated in step 3.
        *. Leave the display window set to its minimum.
    #. Use the on-screen distance measurement tool to measure the lengths of the top and bottom ramps.
        This is illustrated below in Figure 12. Record these lengths and compare to the action limits.

Our Approximation
+++++++++++++++++

    #. Find the phantom center.
    #. Zoom input x4.
    #. Crop insert around initial center.
    #. Apply Window Width and Window Level based on cropped region.
    #. Identify the Y coordinates by sampling a line profile at the sample crop center and finding highest 2 peaks.
    #. Identify the X coordinates by sampling line profiles in the horizontal direction going through the Y points.
    #. Place rectangular ROIs at those centers with a fixed width.
    #. Apply WL based on ROI averages.
    #. Identify widths.
    #. Use ACR formula for results.

ACR Scoring Rubric
++++++++++++++++++

The slice thickness is calculated using the following formula:

    Slice thickness = 0.2 x (top x bottom)/(top + bottom)

In the formula above, the `top` and `bottom` are the measured lengths of the top and bottom signal ramps.

**Note:** 0.2 is a unitless factor that corrects for rotation of the phantom about the vertical (y) axis.

For example, if the top signal ramp were 59.5mm long and the bottom ramp were 47.2mm long, then the
calculated slice thickness would be:

    Slice thickness = 0.2 x (59.5mm x 47.2mm)/(59.5mm + 47.2mm) = 5.26 mm.


Notes
_____

..note::

    A failure of this test means that the scanner is producing slices of substantially different thickness from the
    prescribed thickness. This problem will generally not occur in isolation since the scanner deficiencies that
    can cause it may also cause other image problems. Therefore, the implications of a failure are not just that
    the slices are too thick or thin, but can also result in poor image contrast and low SNR.

..warning::

    When making these measurements, **be careful to fully cover the widths of the ramp with the
    ROIs** in the top-bottom direction, but not to allow the ROIs to stray outside the ramps into adjacent
    high- or low-signal regions. If there is a large difference,(that is, more than 20%), between the signal
    values obtained for the ROIs, it is often due to one or both of the ROIs including regions outside the
    ramps.

Documented by Luis M. Santos, M.D.
luis.santos2@nih.gov


"""

# Python Imports
import os
import sys
import traceback

# Module Imports
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, medfilt

# Local Imports
from hazenlib.ACRObject import ACRObject
from hazenlib.HazenTask import HazenTask
from hazenlib.logger import logger
from hazenlib.types import Measurement, Result
from hazenlib.utils import XY, Line, Point, get_image_orientation


class ACRSliceThickness(HazenTask):
    """Slice width measurement class for DICOM images of the ACR phantom."""

    class SignalLine(Line):
        """Subclass of Line to implement functionality related to the ACR phantom's ramps"""

        def get_FWHM(self):
            """Calculates the FWHM by fitting a piecewise sigmoid to signal across line"""
            if not hasattr(self, "signal"):
                raise ValueError("Signal across line has not been computed!")

            fitted = self._fit_piecewise_sigmoid()

            peaks, props = find_peaks(
                fitted.y, height=0, prominence=np.max(fitted.y).item() / 4
            )
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

            def get_specific_sigmoid(
                wholeData: XY, fitStart: int, fitEnd: int
            ) -> XY:
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
            sigmoidR_func = get_specific_sigmoid(
                smoothed, peak, len(smoothed.x)
            )

            sigmoidL = XY(smoothed.x, sigmoidL_func(smoothed.x))
            sigmoidR = XY(smoothed.x, sigmoidR_func(smoothed.x))

            def blending_weight(x, transition_x, transition_width):
                return 1 / (1 + np.exp(-(x - transition_x) / transition_width))

            W = blending_weight(
                smoothed.x,
                peak,
                1 / 20 * peak + 1 / 20 * (len(smoothed.x) - peak),
            )
            fitted = XY(smoothed.x, (1 - W) * sigmoidL.y + W * sigmoidR.y)

            return fitted

    def __init__(self, **kwargs):
        if kwargs.pop("verbose", None) is not None:
            logger.warning(
                "verbose is not a supported argument for %s",
                type(self).__name__,
            )
        super().__init__(**kwargs)
        # Initialise ACR object
        self.ACR_obj = ACRObject(self.dcm_list)
        self.SAMPLING_LINE_WIDTH = (
            4 / self.ACR_obj.dx
        )  # How many pixel lines to use in the sampling during ramp line profiling.
        self.RAMP_HEIGHT = (
            4.5 / self.ACR_obj.dx
        )  # I measured the ramp height to be about 5mm on PACS, but testing shows it might be slightly less??
        self.RAMP_Y_OFFSET = (
            1 / self.ACR_obj.dx
        )  # 1mm adjustment off center to grab the bottom ramp. There's technically a 2mm gap between slots.
        self.RAMP_X_OFFSET = (
            10 / self.ACR_obj.dx
        )  # This is extra padding added to the resolved width of ramp to allow the FWHM have more samples than necessary in the event we underestimated the true length of the ramp.
        self.INSERT_ROI_HEIGHT = (
            10 / self.ACR_obj.dx
        )  # Allow just enough space for slots but exclude insert boundaries
        self.INSERT_ROI_WIDTH = (
            150 / self.ACR_obj.dx
        )  # Allow enough space to capture the slots which might be R-L offsetted.
        self.CROPPED_ROI_WIDTH = (
            150 / self.ACR_obj.dx
        )  # Allow enough space to capture the slots which might be R-L offsetted.
        self.CROPPED_ROI_HEIGHT = (
            20 / self.ACR_obj.dx
        )  # Capture slots plus some surrounding areas to help visualization in report.
        self.WINDOW_ROI_WIDTH = (
            10 / self.ACR_obj.dx
        )  # Rectangle that captures enough of a population at the center to determine proper mean signal of slots.
        self.WINDOW_ROI_HEIGHT = (
            5 / self.ACR_obj.dx
        )  # Rectangle that captures enough of a population at the center to determine proper mean signal of slots.
        self.RAMP_PROFILE_SMOOTHING = (
            5 / self.ACR_obj.dx
        )  # Smoothing to apply on sampled line profile to remove local minimas within the slot.

    def run(self) -> Result:
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
        results = self.init_result_dict(desc=self.ACR_obj.acquisition_type())
        results.files = self.img_desc(slice_thickness_dcm)

        try:
            thickness_results = self.get_slice_thickness(slice_thickness_dcm)

            results.add_measurement(
                Measurement(
                    name="SliceWidth",
                    type="measured",
                    subtype="slice width",
                    unit="mm",
                    value=round(thickness_results["thickness"], 2),
                ),
            )

            for ramp in thickness_results["ramps"]:
                results.add_measurement(
                    Measurement(
                        name="SliceWidth",
                        type="measured",
                        subtype="Ramps",
                        unit="mm",
                        value=thickness_results["ramps"][ramp]["width"],
                        visibility="intermediate",
                    ),
                )

        except Exception as e:
            logger.exception(
                "Could not calculate the slice thickness for %s"
                " because of : %s",
                self.img_desc(slice_thickness_dcm),
                e,
            )
            traceback.print_exc(file=sys.stdout)

        # only return reports if requested
        if self.report:
            results.add_report_image(self.report_files)

        return results

    def get_slice_thickness(self, dcm):
        """Measure slice thickness. \n
        Identify the ramps, measure the line profile, measure the FWHM, and use this to calculate the slice thickness.

        Returns:
            dict: Dictionary containing:
                - ``"thickness"`` (float): Calculated slice thickness in mm.
                - ``"ramps"`` (list[float]): FWHM values (in pixels) for each ramp line used in the calculation.
        """
        img = dcm.pixel_array
        lines = self.place_lines(img)
        for line in lines:
            line.get_FWHM()
        slice_thickness = (
            0.2
            * (lines[0].FWHM * lines[1].FWHM)
            / (lines[0].FWHM + lines[1].FWHM)
        )

        if self.report:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 3, figsize=(16, 8))
            axes[0].imshow(img)
            for i, line in enumerate(lines):
                axes[0].plot(
                    [line.start.x, line.end.x], [line.start.y, line.end.y]
                )
                axes[i + 1].plot(
                    line.signal.x,
                    line.signal.y,
                    label="Raw signal",
                    alpha=0.25,
                    color=f"C{i}",
                )
                axes[i + 1].plot(
                    line.fitted.x,
                    line.fitted.y,
                    label="Fitted piecewise sigmoid",
                    color=f"C{i}",
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
                os.path.join(
                    self.report_path,
                    f"{self.img_desc(dcm)}_slice_thickness.png",
                )
            )

            fig.savefig(img_path, dpi=600)
            self.report_files.append(img_path)

        return {
            "thickness": slice_thickness,
            "ramps": [line.FWHM for line in lines],
        }

    def place_lines(self, img: np.ndarray) -> list["Line"]:
        """Places line on image within ramps insert.
        Works for a rotated phantom.

        Args:
            img (np.ndarray): Pixel array from DICOM image.

        Returns:
            finalLines (list): A list of the two lines as Line objects.
        """
        # Normalize to uint8, enhance contast and binarize using otsu thresh

        img_uint8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(
            np.uint8
        )
        contrastEnhanced = cv2.createCLAHE(
            clipLimit=2.0, tileGridSize=(3, 3)
        ).apply(img_uint8)
        _, img_binary = cv2.threshold(
            contrastEnhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Find contour by x-span sort
        contours, _ = cv2.findContours(
            img_binary.astype(np.uint8),
            mode=cv2.RETR_TREE,
            method=cv2.CHAIN_APPROX_NONE,
        )

        def get_aspect_ratio(contour):
            _, (width, height), _ = cv2.minAreaRect(contour)
            return min(width, height) / max(width, height)

        # filter out tiny contours from noise
        threshArea = 15 * 15
        contours = [
            cont for cont in contours if cv2.contourArea(cont) >= threshArea
        ]
        # select most elongated insert contour (heuristic for central insert)
        contours_sorted = sorted(
            contours,
            key=lambda c: get_aspect_ratio(c),
        )
        insertContour = contours_sorted[0]

        # Create list of Point objects for the four corners of the contour
        insertCorners = cv2.boxPoints(cv2.minAreaRect(insertContour))
        corners = [Point(*p) for p in insertCorners]

        # Define short sides of contours by list of line objects
        corners = sorted(
            corners, key=lambda point: corners[0].get_distance_to(point)
        )
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
