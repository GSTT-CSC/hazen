"""ACR Slice Thickness.

___________________

Reference
_________

`ACR Large Phantom Guidance PDF <https://accreditationsupport.acr.org/helpdesk/attachments/11093487417>`_

Intro
_____

The slice thickness accuracy test assesses the accuracy with which a
 slice of specified thickness is achieved.  The prescribed slice
 thickness is compared with the measured slice thickness.

The ramps appear in a structure called the slice thickness
insert. Figure 10 shows an image of slice 1 with the slice thickness
insert and signal ramps identified. The two ramps are crossed: one has
a negative slope and the other a positive slope with respect to the
plane of slice 1. They are produced by cutting 1 mm wide slots in a
block of plastic. The slots are open to the interior of the phantom
and are filled with the same solution that fills the bulk of the
phantom.

The signal ramps have a slope of 10 to 1 with respect to the plane of
slice 1, which is an angle of about 5.71 deg.  Therefore, the signal
ramps will appear in the image of slice 1 with a length that is 10
times the thickness of the slice. If the phantom is misaligned from
right-left, one ramp will appear longer than the other. The crossed
ramps allow for correction of the error introduced by right-left
misalignment, and the slice thickness formula takes that into account.

ACR Guidelines
______________

ACR Algorithm
+++++++++++++

    #. Display slice 1, and magnify the image by a factor of 2 to 4,
        keeping the slice thickness insert fully visible on the
        screen.
    #. Adjust the display level so that the signal ramps are well
        visualized.
        *. The ramp signal is much lower than that of surrounding
            water, so usually it will be necessary to lower the
            display level substantially and narrow the window.
    #. Place a rectangular ROI at the middle of each ramp as shown
        below in Figure 11.
        *. Note the mean signal values for each of these two ROIs and
         then average those values.
        *. The result is a number approximating the mean signal in the
         middle of the ramps.
        *. An elliptical ROI may be used if a rectangular one is
         unavailable.
    #. Lower the display level to half of the average ramp signal
        calculated in step 3.
        *. Leave the display window set to its minimum.
    #. Use the on-screen distance measurement tool to measure the
        lengths of the top and bottom ramps.  This is illustrated
        below in Figure 12. Record these lengths and compare to the
        action limits.

Computer Vision Methodology
+++++++++++++++++++++++++++

This automated implementation locates the slice thickness insert using
computer vision and calculates ramp widths via curve fitting:

    #. Pre-process image with CLAHE contrast enhancement and Otsu
        binarization to isolate the insert.
    #. Detect contours and identify the slice thickness insert using
       aspect ratio heuristics (the insert is the most elongated
       rectangular structure).
    #. Calculate the oriented bounding box corners of the insert to
        determine ramp geometry.
    #. Construct profile lines across the ramps using geometric
        sublines (spanning 95% of ramp length).
    #. Extract pixel intensity profiles along each line (averaging
        over 4-pixel width).
    #. Smooth profiles with median/Gaussian filters and fit piecewise
        sigmoid functions:
       - Detect the signal peak to separate left and right segments
       - Fit separate sigmoids to each segment using least-squares optimization
       - Blend the sigmoids with a smooth transition weighting
         function at the peak
    #. Calculate FWHM from fitted curves using linear interpolation at
        half-maximum points.
    #. Apply ACR formula: thickness = 0.2 * (L1 * L2) / (L1 + L2),
        accounting for phantom rotation.

Sagittal images are automatically detected and rotated 90 degrees
clockwise prior to analysis.

ACR Scoring Rubric
++++++++++++++++++

The slice thickness is calculated using the following formula:

    Slice thickness = 0.2 x (top x bottom)/(top + bottom)

In the formula above, the `top` and `bottom` are the measured lengths
of the top and bottom signal ramps.

**Note:** 0.2 is a unitless factor that corrects for rotation of the
  phantom about the vertical (y) axis.

For example, if the top signal ramp were 59.5mm long and the bottom
ramp were 47.2mm long, then the calculated slice thickness would be:

    Slice thickness = 0.2 x (59.5mm x 47.2mm)/(59.5mm + 47.2mm) = 5.26 mm.


Notes
-----
..note::

    A failure of this test means that the scanner is producing slices
    of substantially different thickness from the prescribed
    thickness. This problem will generally not occur in isolation
    since the scanner deficiencies that can cause it may also cause
    other image problems. Therefore, the implications of a failure are
    not just that the slices are too thick or thin, but can also
    result in poor image contrast and low SNR.

..warning::

    The automated line placement relies on the insert contour being
    clearly distinguishable from background.  Poor image contrast or
    artifacts may cause the aspect ratio heuristic to fail. The
    fitting algorithm assumes the ramps produce distinct signal peaks;
    very thin slices (<2mm) may challenge the piecewise sigmoid fit.

Documented by Luis M. Santos, M.D.
luis.santos2@nih.gov

and

Alex Drysdale
alexander.drysdale@wales.nhs.uk

"""

# Python Imports
import os
import sys
import traceback
from pathlib import Path

# Module Imports
import cv2
import numpy as np
import pydicom
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
    """Slice width measurement class for DICOM images of the ACR phantom.

    Automatically detects the slice thickness insert using computer vision,
    places measurement lines across the crossed ramps, and calculates slice
    thickness using piecewise sigmoid fitting to determine FWHM. Handles
    arbitrary phantom rotation and sagittal acquisitions.
    """

    class SignalLine(Line):
        """Subclass of Line to related to the ACR phantom's ramps.

        Extends the Line class with signal profiling and curve fitting
        capabilities specifically designed for measuring the FWHM of
        the characteristic ramp signal (low signal slots in high
        signal background).
        """

        def get_fwhm(self) -> None:
            """Calculate the FWHM by fitting a piecewise sigmoid.

            Fits a smooth piecewise sigmoid curve to the signal
            profile, then calculates the Full Width at Half Maximum
            (FWHM) using linear interpolation at the half-maximum
            crossing points. The fitted curve and FWHM value are
            stored as instance attributes.

            Raises:
                ValueError: If signal has not been computed via
                        get_signal() before calling.

            Sets Attributes:
                fwhm (float): Full width at half maximum in pixels.
                fitted (XY): Fitted curve as XY coordinates.

            """
            if not hasattr(self, "signal"):
                msg = "Signal across line has not been computed!"
                raise ValueError(msg)

            fitted = self._fit_piecewise_sigmoid()

            _, props = find_peaks(
                fitted.y,
                height=0,
                prominence=np.max(fitted.y).item() / 4,
            )
            peak_height = np.max(props["peak_heights"])

            background_l = fitted.y[0]
            background_r = fitted.y[-1]

            half_max_l = (peak_height - background_l) / 2 + background_l
            half_max_r = (peak_height - background_r) / 2 + background_r

            def simple_interpolate(target_y: float, signal: XY) -> float:
                crossing_index = np.where(signal.y > target_y)[0][0]
                x1, x2 = signal.x[crossing_index - 1], signal.x[crossing_index]
                y1, y2 = signal.y[crossing_index - 1], signal.y[crossing_index]
                return x1 + (target_y - y1) * (x2 - x1) / (y2 - y1)

            x_l = simple_interpolate(half_max_l, fitted)
            x_r = simple_interpolate(half_max_r, fitted[:, ::-1])

            self.fwhm = x_r - x_l
            self.fitted = fitted

        def _fit_piecewise_sigmoid(self) -> XY:
            smoothed = self.signal.copy()
            k = round(len(smoothed.y) / 20)
            if k % 2 == 0:
                k += 1
            smoothed.y = medfilt(smoothed.y, k)
            smoothed.y = gaussian_filter1d(smoothed.y, round(k / 2.5))

            peaks, props = find_peaks(
                smoothed.y,
                height=0,
                prominence=np.max(smoothed.y).item() / 4,
            )
            heights = props["peak_heights"]
            peak = peaks[np.argmax(heights)]

            def get_specific_sigmoid(
                whole_data: XY,
                fit_start: int,
                fit_end: int,
            ) -> XY:
                fit_data = whole_data[:, fit_start:fit_end]
                a = np.max(fit_data.y) - np.min(fit_data.y)
                b = np.min(fit_data.y)

                dy = np.diff(fit_data.y)
                dx = np.diff(fit_data.x)
                dx[dx == 0] = 1e-6
                abs_deriv = np.abs(dy / dx)
                abs_grad_max = np.max(abs_deriv)
                k = (
                    np.sign(fit_data.y[-1] - fit_data.y[0])
                    * abs_grad_max
                    * 0.5
                )
                x0 = fit_data.x[np.argmax(abs_deriv)]

                p0 = [a, k, x0, b]

                def sigmoid(
                    x: float | np.ndarray,
                    a: float,
                    k: float,
                    x0: float,
                    b: float,
                ) -> float:
                    exp_term = np.exp(-k * (x - x0))
                    return a / (1 + exp_term) + b

                popt, _ = curve_fit(sigmoid, fit_data.x, fit_data.y, p0=p0)

                def specific_sigmoid(x: float | np.ndarray) -> float:
                    return sigmoid(x, *popt)

                return specific_sigmoid

            sigmoid_l_func = get_specific_sigmoid(smoothed, 0, peak)
            sigmoid_r_func = get_specific_sigmoid(
                smoothed,
                peak,
                len(smoothed.x),
            )

            sigmoid_l = XY(smoothed.x, sigmoid_l_func(smoothed.x))
            sigmoid_r = XY(smoothed.x, sigmoid_r_func(smoothed.x))

            def blending_weight(
                x: np.ndarray,
                transition_x: np.ndarray,
                transition_width: np.ndarray,
            ) -> np.ndarray:
                return 1 / (1 + np.exp(-(x - transition_x) / transition_width))

            w = blending_weight(
                smoothed.x,
                peak,
                1 / 20 * peak + 1 / 20 * (len(smoothed.x) - peak),
            )
            return XY(smoothed.x, (1 - w) * sigmoid_l.y + w * sigmoid_r.y)

    def __init__(self, **kwargs) -> None:
        """Initialise the ACRSliceThickness object."""
        if kwargs.pop("verbose", None) is not None:
            logger.warning(
                "verbose is not a supported argument for %s",
                type(self).__name__,
            )
        super().__init__(**kwargs)
        # Initialise ACR object
        self.ACR_obj = ACRObject(self.dcm_list)

    def run(self) -> Result:
        """Perform slice width measurement.

        Using slice 1 from the ACR phantom image set.

        Returns:
            dict: results are returned in a standardised dictionary
                structure specifying the task name, input DICOM Series
                Description + SeriesNumber + InstanceNumber, task
                measurement key-value pairs, optionally path to the
                generated images for visualisation.

        """
        # Identify relevant slice
        slice_thickness_dcm = self.ACR_obj.slice_stack[0]
        # TODO: image may be 90 degrees cw or acw, could use code to
        # identify which or could be added as extra arg

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

    def get_slice_thickness(self, dcm: pydicom.Dataset) -> dict:
        """Measure slice thickness.

        Identify the ramps, measure the line profile, measure the
        FWHM, and use this to calculate the slice thickness.

        Returns:
            dict: Dictionary containing:
                - ``"thickness"`` (float): Calculated slice thickness in mm.
                - ``"ramps"`` (list[float]): FWHM values (in pixels)
                  for each ramp line used in the calculation.

        """
        img = dcm.pixel_array
        lines = self.place_lines(img)
        for line in lines:
            line.get_fwhm()
        slice_thickness = (
            0.2
            * (lines[0].fwhm * lines[1].fwhm)
            / (lines[0].fwhm + lines[1].fwhm)
        )

        if self.report:
            import matplotlib.pyplot as plt  # noqa: PLC0415 I001

            fig, axes = plt.subplots(1, 3, figsize=(16, 8))
            axes[0].imshow(img)
            for i, line in enumerate(lines):
                axes[0].plot(
                    [line.start.x, line.end.x],
                    [line.start.y, line.end.y],
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
                Path(self.report_path)
                / f"{self.img_desc(dcm)}_slice_thickness.png",
            )

            fig.savefig(img_path, dpi=600)
            self.report_files.append(img_path)

        return {
            "thickness": slice_thickness,
            "ramps": {
                pos: {"width": line.fwhm}
                for pos, line in zip(("top", "bottom"), lines, strict=True)
            },
        }

    def place_lines(self, img: np.ndarray) -> list["Line"]:
        """Places line on image within ramps insert.

        Works for a rotated phantom.

        Args:
            img (np.ndarray): Pixel array from DICOM image.

        Returns:
            final_lines (list): A list of the two lines as Line objects.

        """
        # Normalize to uint8, enhance contast and binarize using otsu thresh

        img_uint8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(
            np.uint8,
        )
        contrast_enhanced = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(3, 3),
        ).apply(img_uint8)
        _, img_binary = cv2.threshold(
            contrast_enhanced,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

        # Find contour by x-span sort
        contours, _ = cv2.findContours(
            img_binary.astype(np.uint8),
            mode=cv2.RETR_TREE,
            method=cv2.CHAIN_APPROX_NONE,
        )

        def get_aspect_ratio(contour: np.ndarray) -> float:
            _, (width, height), _ = cv2.minAreaRect(contour)
            return min(width, height) / max(width, height)

        # filter out tiny contours from noise
        thresh_area = 15 * 15
        contours = [
            cont for cont in contours if cv2.contourArea(cont) >= thresh_area
        ]
        # select most elongated insert contour (heuristic for central insert)
        contours_sorted = sorted(
            contours,
            key=lambda c: get_aspect_ratio(c),
        )
        insert_contour = contours_sorted[0]

        # Create list of Point objects for the four corners of the contour
        insert_corners = cv2.boxPoints(cv2.minAreaRect(insert_contour))
        corners = [Point(*p) for p in insert_corners]

        # Define short sides of contours by list of line objects
        corners = sorted(
            corners,
            key=lambda point: corners[0].get_distance_to(point),
        )
        short_sides = [Line(*corners[:2]), Line(*corners[2:])]

        # Get sublines of short sides and force start point to be higher in y
        sublines = [line.get_subline(perc=30) for line in short_sides]
        for line in sublines:
            if line.start.y < line.end.y:
                line.point_swap()

        # Define connecting lines
        connecting_lines = [
            self.SignalLine(sublines[0].start, sublines[1].start),
            self.SignalLine(sublines[0].end, sublines[1].end),
        ]

        # Final lines are sublines of connecting lines
        final_lines = [line.get_subline(perc=95) for line in connecting_lines]
        for line in final_lines:
            line.get_signal(img)

        return final_lines
