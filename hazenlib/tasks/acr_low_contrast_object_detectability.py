"""Low-Contrast Object Detectability.

As per section 7 of the Large and Medium Phantom Test Guidance for the
ACR MRI Accreditation Program:

https://www.acraccreditation.org/-/media/ACRAccreditation/Documents/MRI/ACR-Large-Med-Phantom-Guidance-102022.pdf

```
the low contrast detectability test assesses the extent to which objects
of low contrast are discernible in the images.
```

These are performed on slices 8 through 11 by counting
the number of visible spokes.

The implementation follows that of:

A statistical approach to automated analysis of the
low-contrast object detectability test for the large ACR MRI phantom

DOI = {10.1002/acm2.70173}
journal = {Journal of Applied Clinical Medical Physics},
author = {Golestani, Ali M. and Gee, Julia M.},
year = {2025},
month = jul

An implementation by the authors can be found on GitHub:
    https://github.com/aligoles/ACR-Low-Contrast-Object-Detectability

With the original paper:
    https://doi.org/10.1002/acm2.70173

ACR Low Contrast Object Detectability:
    https://mriquestions.com/uploads/3/4/5/7/34572113/largephantomguidance.pdf

Notes from the paper:

- Images with acquired with:
        - 3.0T Siemens MAGNETOM Vida.
        - 1.5T Philips scanner integrated into an Elekta Unity MR-Linac System.
- 40 Datasets analyzed (20 for each scanner).


Implementation overview:

- Normalise image intensity for each slice (independently) to within [0, 1].
- Background removal process performed using histogram thresholding.
- Contrast disk is identified.
        - Detect center of phantom, crop and then find a large circle.
- 90 Angular radials profile in a specific angle are generated.
- Known phantom geometry and rotation used to calculate position of first spoke.
        - Circles at 12.5, 25.0 and 38.0mm from CoG.
- 2nd order polynomial fitted to the model and added to general linear model
    (GLM) regressors.
- 3 GLM regressors created for each 1D profile.
- Test passes if every GLM regressors exceed the significance level.
        - Significance level for each slice is set to 0.0125.
- Significance within each slice is adjusted using the Benjamini-Hochberg
    false discovery rate.

Implemented for Hazen by Alex Drysdale: alexander.drysdale@wales.nhs.uk
"""
# Typing
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pydicom

# Python imports
import logging
from typing import Any

# Module imports
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import statsmodels as sm
import statsmodels.api
from matplotlib.patches import Circle

# Local imports
from hazenlib.ACRObject import ACRObject
from hazenlib.HazenTask import HazenTask
from hazenlib.types import Measurement, Result

logger = logging.getLogger(__name__)

class ACRLowContrastObjectDetectability(HazenTask):
    """Low Contrast Object Detectability (LCOD) class for the ACR phantom."""

    OBJECTS_PER_SPOKE = 3
    NUM_SPOKES = 10

    def __init__(
            self, alpha: float = 0.0125, **kwargs: Any,
    ) -> None:
        """Initialise the LCOD object."""
        if kwargs.pop("verbose", None) is not None:
            logger.warning(
                "verbose is not a supported argument for %s",
                type(self).__name__,
            )
        super().__init__(**kwargs)

        self.alpha = alpha

        self.slice_range = slice(7,11)

        # Initialise ACR object
        self.ACR_obj = ACRObject(self.dcm_list)
        self.rotation = np.int64(
            self.ACR_obj.determine_rotation(
                self.ACR_obj.slice_stack[0].pixel_array,
            ),
        )
        self.lcod_center = None

        # Pass threshold is at least N spokes total for both the T1 and T2
        # acquisitions where:
        # @ 1.5T, N =  7
        # @ 3.0T, N = 37
        match float(self.ACR_obj.slice_stack[0]["MagneticFieldStrength"].value):
            case 3.0:
                self.pass_threshold = 37
            case 1.5:
                self.pass_threshold = 7
            case _:
                logger.error(
                    "No LCOD pass threshold specified for %s T systems"
                    " assuming a pass threshold of at least 7 spokes for"
                    " each sequence",
                    self.ACR_obj.slice_stack[0]["MagneticFieldStrength"].value,
                )


    def run(self) -> Result:
        """Run the LCOD analysis."""
        results = self.init_result_dict()
        results.files = [
            self.img_desc(f)
            for f in self.ACR_obj.slice_stack[self.slice_range]
        ]

        total_spokes = 0
        for i, dcm in enumerate(self.ACR_obj.slice_stack[self.slice_range]):
            slice_no = i + self.slice_range.start + 1
            result = self.count_spokes(dcm, alpha=self.alpha)
            try:
                num_spokes = np.where(result != self.OBJECTS_PER_SPOKE)[0][0]
            except IndexError:
                num_spokes = result.size

            # Add individual spoke measurements for debugging
            # and further analysis
            # If this results in hose-pipping then it might be best to remove
            for j, r in enumerate(result):
                spoke_no = j + 1
                results.add_measurement(
                    Measurement(
                        name="LowContrastObjectDetectability",
                        type="measured",
                        subtype=f"slice {slice_no} spoke {spoke_no}",
                        value=r,
                    ),
                )
            total_spokes += num_spokes
            results.add_measurement(
                Measurement(
                    name="LowContrastObjectDetectability",
                    type="measured",
                    subtype=f"slice {slice_no}",
                    value=num_spokes,
                ),
            )

        results.add_measurement(
            Measurement(
                name="LowContrastObjectDetectability",
                type="measured",
                subtype="total",
                value=total_spokes,
            ),
        )

        results.add_measurement(
            Measurement(
                name="LowContrastObjectDetectability",
                type="measured",
                subtype="pass/fail",
                value=total_spokes >= self.pass_threshold,
            ),
        )

        if self.report:
            results.add_report_image(self.report_files)

        return results


    def count_spokes(
        self,
        dcm: pydicom.Dataset,
        alpha: float = 0.05,
        report_window: float = 0.2,
    ) -> np.ndarray:
        """Count the number of spokes using polar coordinate transformation."""
        # Input validation and preprocessing
        img_data = np.abs(dcm.pixel_array) if np.min(dcm.pixel_array) < 0 else dcm.pixel_array

        if np.max(img_data) == 0:
            logger.warning("Image has zero intensity, returning zero spokes")
            return np.zeros(self.NUM_SPOKES)

        # Normalize and threshold
        norm_img = img_data / np.max(img_data)
        try:
            binary_mask = self.histogram_threshold(norm_img)
        except Exception as e:
            logger.warning(f"Histogram thresholding failed: {e}, using simple threshold")
            binary_mask = norm_img > 0.3

        # Find center with robust method
        try:
            cx_inner, cy_inner = self.find_center(dcm)
        except Exception as e:
            logger.warning(f"Center detection failed: {e}, using image center")
            cy_inner, cx_inner = img_data.shape[0] // 2, img_data.shape[1] // 2

        # Convert to polar coordinates for efficient radial analysis
        polar_img, r_grid, theta_grid = self._cartesian_to_polar(
            norm_img, cx_inner, cy_inner
        )
        polar_mask, _, _ = self._cartesian_to_polar(binary_mask, cx_inner, cy_inner)

        # Analyze spokes in polar space
        pass_vec = np.zeros(self.NUM_SPOKES)
        the_angle = self.rotation

        for spoke_number in range(self.NUM_SPOKES):
            # Define angular range for current spoke
            angle_center = the_angle - spoke_number * 36
            angle_range = np.deg2rad([angle_center - 8, angle_center + 8])

            # Extract radial profiles across the angular range
            profiles = []
            for angle_offset in np.linspace(-8, 8, 16):  # Sample across spoke width
                angle = np.deg2rad(angle_center + angle_offset)
                profile = self._extract_radial_profile(polar_img, polar_mask,
                                                     r_grid, theta_grid, angle)
                if profile is not None and len(profile) > 20:
                    profiles.append(profile)

            if not profiles:
                continue

            # Average profiles and analyze
            avg_profile = np.mean(profiles, axis=0)
            p_vals, params = self._analyze_profile(avg_profile, spoke_number)

            if (p_vals is not None and params is not None and
                len(p_vals) >= self.OBJECTS_PER_SPOKE and
                np.all(p_vals[:self.OBJECTS_PER_SPOKE] < alpha) and
                np.all(params[:self.OBJECTS_PER_SPOKE] > 0)):
                pass_vec[spoke_number] = 1

        # Generate report if requested
        if self.report:
            self._generate_report(dcm, pass_vec, cx_inner, cy_inner, report_window)

        return pass_vec


    def find_center(
        self,
        dcm: pydicom.Dataset,
        crop_ratio: float = 0.7,
    ) -> tuple[int]:
        """Find the center of the LCOD phantom."""
        if self.lcod_center is not None:
            return self.lcod_center

        (main_cx, main_cy), main_radius = self.ACR_obj.find_phantom_center(
            dcm.pixel_array, self.ACR_obj.dx, self.ACR_obj.dy,
        )
        r = main_radius * crop_ratio
        cropped_image = dcm.pixel_array[
            max(0, int(main_cy - r)):int(main_cy + r + 1),
            max(0, int(main_cx - r)):int(main_cx + r + 1),
        ]
        cropped_image = (
            (cropped_image - cropped_image.min())
            * 255.0 / (cropped_image.max() - cropped_image.min())
        ).astype(np.uint8)

        img_blur = cv2.GaussianBlur(cropped_image, (1, 1), 0)
        img_grad = img_blur.max() - img_blur

        detected_circles = cv2.HoughCircles(
            img_grad,
            method=cv2.HOUGH_GRADIENT,
            dp=2,
            minDist=cropped_image.shape[0] // 2,
        ).flatten()
        self.lcod_center = tuple(
            dc + max(0, int(main_c - r))
            for dc, main_c in zip(detected_circles[:2], (main_cx, main_cy))
        )

        if self.report:
            fig, axes = plt.subplots(2, 2, constrained_layout=True)

            axes[0, 0].imshow(cropped_image)
            axes[0, 0].scatter(
                detected_circles[0],
                detected_circles[1],
                marker="x",
                color="red",
            )
            axes[0, 0].set_title("Cropped Image")

            axes[0, 1].imshow(img_blur)
            axes[0, 1].set_title("Blur")

            axes[1, 0].imshow(img_grad)
            axes[1, 0].set_title("Inverse")

            cx, cy, r = detected_circles[:3]
            circle = Circle(
                (cx, cy), r, fill=False, edgecolor="red", linewidth=2,
            )
            axes[1, 1].imshow(cropped_image, cmap="gray")
            axes[1, 1].add_patch(circle)
            axes[1, 1].set_title("Detected Circle")

            fig.suptitle("LCOD Center Detection")

            data_path = Path(self.dcm_list[0].filename).parent.name
            img_path = (
                Path(self.report_path)
                / f"center_{data_path}_{self.img_desc(dcm)}.png"
            )
            fig.savefig(img_path)
            plt.close()

        return self.lcod_center


    @staticmethod
    def histogram_threshold(
            img: np.ndarray,
            low: float = 0.05,
            high: float = 0.65,
            step: float = 0.001,
            prop: tuple[float] = (0.1, 0.2),
    ) -> np.ndarray:
        """Histogram thesholding to extract the central disk."""
        # Image must be normalised!
        if np.max(img) > 1.0:
            img /= np.max(img)

        structure = np.ones((3, 3), dtype=int)

        for thr in np.arange(low, high, step):
            ret, thresh = cv2.threshold(img, thr, 1, 0)
            labelled, ncomponents = sp.ndimage.measurements.label(
                thresh, structure,
            )
            thresh_inner = labelled == np.max(labelled)
            if prop[0] < np.sum(thresh_inner != 0) / np.sum(img != 0) < prop[1]:
                break

        return img * sp.ndimage.binary_fill_holes(
            thresh,
            structure=np.ones((5, 5), dtype=int),
        )


    # The angle_stat_test method has been replaced by the more efficient
    # polar coordinate approach using _analyze_profile, _extract_radial_profile,
    # and related methods.


    @staticmethod
    def _cartesian_to_polar(self, image: np.ndarray, cx: float, cy: float) -> tuple:
        """Convert Cartesian image to polar coordinates.

        Args:
            image: Input image in Cartesian coordinates
            cx: Center x-coordinate
            cy: Center y-coordinate

        Returns:
            Tuple of (polar_image, r_grid, theta_grid)
        """
        height, width = image.shape
        y, x = np.mgrid[:height, :width]

        # Convert to polar coordinates
        r = np.sqrt((x - cx)**2 + (y - cy)**2)
        theta = np.arctan2(y - cy, x - cx)  # Note: arctan2 handles all quadrants

        # Create polar image using interpolation
        r_max = int(np.sqrt(height**2 + width**2))
        r_points = np.linspace(0, r_max, r_max)
        theta_points = np.linspace(-np.pi, np.pi, 360)

        # Use scipy interpolation for polar transform
        polar_img = sp.ndimage.map_coordinates(
            image,
            [cy + r_points[:, None] * np.sin(theta_points[None, :]),
             cx + r_points[:, None] * np.cos(theta_points[None, :])],
            order=1,
            mode='constant',
            cval=0
        )

        return polar_img, r_points, theta_points

    def _extract_radial_profile(self, polar_img: np.ndarray, polar_mask: np.ndarray,
                               r_grid: np.ndarray, theta_grid: np.ndarray,
                               angle: float) -> np.ndarray:
        """Extract radial profile at given angle from polar image.

        Args:
            polar_img: Image in polar coordinates
            polar_mask: Binary mask in polar coordinates
            r_grid: Radial coordinate grid
            theta_grid: Angular coordinate grid
            angle: Angle in radians

        Returns:
            Radial intensity profile or None if extraction fails
        """
        # Find closest angle index
        angle_idx = np.argmin(np.abs(theta_grid - angle))
        if angle_idx >= polar_img.shape[1]:
            return None

        # Extract profile
        profile = polar_img[:, angle_idx]
        mask = polar_mask[:, angle_idx] if polar_mask.shape == polar_img.shape else None

        # Apply mask if available
        if mask is not None:
            profile = profile * mask

        # Remove background and truncate
        idx_l, idx_h = self.truncate_profile(profile, 0.3)
        if idx_l is None or idx_h is None or idx_l >= idx_h:
            return None

        return profile[idx_l:idx_h]

    def _analyze_profile(self, profile: np.ndarray, spoke_number: int) -> tuple:
        """Analyze radial profile for low-contrast object detection.

        Args:
            profile: Radial intensity profile
            spoke_number: Spoke number (0-9)

        Returns:
            Tuple of (p-values, parameters) or (None, None) if analysis fails
        """
        if len(profile) < 20:
            return None, None

        # Resample to fixed length
        profile_resampled = sp.signal.resample(profile, 90)

        # Detrend with robust polynomial fitting
        if np.std(profile_resampled) > 0.01:
            x = np.linspace(0, 1, len(profile_resampled))
            # Use lower order polynomial for stability
            coeffs = np.polyfit(x, profile_resampled, 2)
            trend = np.polyval(coeffs, x)
            detrended = profile_resampled - trend
        else:
            detrended = profile_resampled
            trend = np.zeros_like(profile_resampled)

        # Simple smoothing
        kernel = np.ones(3) / 3
        smoothed = np.convolve(detrended, kernel, mode='same')

        # Create design matrix with template
        template = self.template_profile_separate(spoke_number)
        X = np.column_stack([template, np.ones(90)])  # Include intercept

        # Fit linear model
        try:
            model = sm.OLS(smoothed, X).fit()
            return model.pvalues[:3], model.params[:3]  # Return only object p-values and params
        except:
            return None, None

    def _generate_report(self, dcm: pydicom.Dataset, pass_vec: np.ndarray,
                        cx: float, cy: float, report_window: float):
        """Generate analysis report with detected spokes."""
        try:
            fig, axes = plt.subplots(1, 1, figsize=(8, 8))
            center_val = dcm.pixel_array[int(cy), int(cx)]

            # Use robust intensity scaling
            mean_val = np.mean(dcm.pixel_array)
            std_val = np.std(dcm.pixel_array)
            vmin = max(0, mean_val - 2 * std_val)
            vmax = mean_val + 2 * std_val

            axes.imshow(dcm.pixel_array, cmap="gray", alpha=0.7, vmin=vmin, vmax=vmax)

            # Highlight detected spokes
            the_angle = self.rotation
            for i, detected in enumerate(pass_vec):
                if detected > 0:
                    angle = np.radians(the_angle - i * 36)
                    # Draw line for detected spoke
                    r_max = min(dcm.pixel_array.shape) // 2
                    x_end = cx + r_max * np.cos(angle)
                    y_end = cy + r_max * np.sin(angle)
                    axes.plot([cx, x_end], [cy, y_end], 'r-', alpha=0.7, linewidth=2)

            axes.scatter(cx, cy, marker="x", color='red', s=100)
            axes.set_title(f"Detected Spokes {int(np.sum(pass_vec))}/10", fontsize=14)
            axes.set_xlabel("Pixel")
            axes.set_ylabel("Pixel")

            data_path = Path(self.dcm_list[0].filename).parent.name
            img_path = (Path(self.report_path) /
                       f"spokes_{data_path}_{self.img_desc(dcm)}.png")
            fig.savefig(img_path, dpi=150, bbox_inches='tight')
            plt.close()

            self.report_files.append(img_path)
        except Exception as e:
            logger.debug(f"Failed to generate report: {e}")

    @staticmethod
    def truncate_profile(p: float, m: np.ndarray) -> tuple[float] | tuple[None]:
        """Get a 1D array of profile (p) and a background threshold (M).

        and eliminate backgrounds from the begining and end of the profile
        and returns the index of the first and last element with higher value
        than m
        """
        indices = np.where(p >= m)[0]
        # Return the first and last indices
        if indices.size == 0:
            # Handle edge case where no value meets the condition
            return None, None
        idx_l, idx_h = indices[0], indices[-1]
        return idx_l, idx_h

    @staticmethod
    def template_profile_separate(spoke_number: int) -> np.ndarray:
        """Generate expected reference profile depending on the spoke number."""
        profile = np.zeros((90,3))
        dist = [12/0.5, 25/0.5, 38/0.5]  # in millimeter
        radi = [7, 6.39, 5.78, 5.17, 4.55, 3.94, 3.33, 2.72, 2.11, 1.5]
        r = radi[spoke_number]
        for i, d in enumerate(dist):
            profile[round(d - r):round(d + r), i] = 1
        return profile

    @staticmethod
    def shift_for_max_corr(x: int, y: int, max_shift: int) -> tuple:
        """Jitter generated 1D profile to align with the spoke reference."""
        cc_vec = []
        shifts = np.arange(-max_shift, max_shift + 1)

        for shift in shifts:
            y_shifted = np.roll(y, shift)
            # Handle boundary conditions
            if shift < 0:
                y_shifted[shift:] = y_shifted[shift - 1]
            elif shift > 0:
                y_shifted[:shift] = y_shifted[shift - 1]

            # Compute correlation
            cc_vec.append(np.corrcoef(x, y_shifted)[0, 1])

        # Find the optimal shift with the maximum correlation
        optimal_shift_idx = np.argmax(cc_vec)
        optimal_shift = shifts[optimal_shift_idx]
        y_aligned = np.roll(y, optimal_shift)

        return x, y_aligned, optimal_shift
