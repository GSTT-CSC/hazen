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

Notes from the paper:

- Images with acquired with:
        - 3.0T Siemens MAGNETOM Vida.
        - 1.5T Philips scanner integrated into an Elekta Unity MR-Linac System.
- 40 Datasets analyzed (20 for each scanner).


Implementation overview:

- Normalise image intensity for each slice (independently) to within [0, 1].
- Background removal process performed using histogram thresholding.
- Contrast disk is identified by detecting and labelling connected components.
- Center of Gravity (CoG) method used to detect center of circle.
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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pydicom

# Python imports
import logging
from typing import Any

# Module imports
import cv2
import numpy as np
import pandas as pd
import scipy as sp
import statsmodels as sm
import statsmodels.api

# Local imports
from hazenlib.ACRObject import ACRObject
from hazenlib.HazenTask import HazenTask
from hazenlib.types import Measurement, Result

logger = logging.getLogger(__name__)

class ACRLowContrastObjectDetectability(HazenTask):
    """Low Contrast Object Detectability (LCOD) class for the ACR phantom."""

    def __init__(self, alpha: float = 0.0125, **kwargs: Any) -> None:
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
            slice_no = i + self.slice_range.start
            result = self.count_spokes(dcm, alpha=self.alpha)
            try:
                num_spokes = np.where(result != 3)[0][0]
            except IndexError:
                num_spokes = result.size

            # Add individual spoke measurements for debugging
            # and further analysis
            # If this results in hose-pipping then it might be best to remove
            for j, r in enumerate(result):
                results.add_measurement(
                    Measurement(
                        name="LowContrastObjectDetectability",
                        type="measured",
                        subtype=f"slice {slice_no} spoke {j + 1}",
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


    def count_spokes(self, dcm: pydicom.Dataset, alpha: float = 0.0125) -> int:
        """Count the number of spokes."""
        if np.min(dcm.pixel_array) < 0:
            msg = "Pixel data should be positive"
            logger.critical(
                "%s but got minimum = %f", msg, np.min(dcm.pixel_array),
            )
            raise ValueError(msg)
        norm_img = dcm.pixel_array / np.max(dcm.pixel_array)
        image_threshold = self.histogram_threshold(norm_img)
        cX_inner, cY_inner = self.find_center(image_threshold)

        img_all = np.zeros_like(dcm.pixel_array)
        pass_vec = np.zeros(10)
        p_vals_vec = []
        params_vec = []
        angle_vec = []
        spoke_vec = []
        the_angle = np.int64(self.ACR_obj.determine_rotation(dcm.pixel_array))

        for spoke_number, alpha_degree_initial in enumerate(
                range(the_angle, -360, -36),
        ):
            if spoke_number > 9:
                break
            for angle in np.arange(
                    alpha_degree_initial - 8,
                    alpha_degree_initial + 8,
                    1 / (1 + spoke_number),
            ):
                p_vals, params = self.angle_stat_test(
                    image_threshold,
                    angle,
                    spoke_number + 1,
                    cX_inner,
                    cY_inner,
                )
                if p_vals is not None and params is not None:
                    params_vec.extend(params[0:3])
                    p_vals_vec.extend(p_vals[0:3])
                    spoke_vec.append(spoke_number)
                    angle_vec.append(angle)

        p_vals_vec_fdr = sm.stats.multitest.fdrcorrection(
            p_vals_vec, alpha=alpha, method="indep", is_sorted=False,
        )
        pvals_fdr = p_vals_vec_fdr[0].reshape(-1, 3)
        params_vec_fdr = np.array(params_vec).reshape(-1, 3)

        # check if all three p-values of a spoke pass significant threshold
        for i, g in enumerate(pvals_fdr):
            if np.sum(g) == 3 and np.sum(params_vec_fdr[i, :] > 0) == 3:
                img = self.angle_image(
                    angle_vec[i],
                    cX_inner,
                    cY_inner,
                    img_all.shape[1],
                    img_all.shape[0],
                )
                img_all = np.logical_or(img_all, img)
                pass_vec[spoke_vec[i]] += 1

        return pass_vec


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


    @staticmethod
    def find_center(thresh_inner: np.ndarray) -> tuple[int]:
        """Find the center of the LCOD circle."""
        M_inner = cv2.moments(np.uint8(thresh_inner))
        cY_inner = int(M_inner["m10"] / M_inner["m00"])
        cX_inner = int(M_inner["m01"] / M_inner["m00"])
        return (cX_inner, cY_inner)


    @staticmethod
    def angle_stat_test(
            image_thresholded: np.ndarray,
            angle: float,
            spoke_number: int,
            cX: int,
            cY: int,
    ) -> tuple:
        """Perform statistical test for s specific angle."""
        # Normalize the image
        image_min, image_max = np.min(image_thresholded), np.max(image_thresholded)
        image_normalized = (image_thresholded - image_min) / (image_max - image_min)
        [X, Y] = image_thresholded.shape

        # Generate angle image and radial profile
        Im = ACRLowContrastObjectDetectability.angle_image(angle, cX, cY,X,Y)
        df = ACRLowContrastObjectDetectability.angle_profile_radial(
            image_normalized, Im, cX, cY,
        )
        p = np.array(df["profile"])

        # Truncate the profile
        idx_l, idx_h = ACRLowContrastObjectDetectability.truncate_profile(
            p, 0.5,
        )
        if idx_l is None or idx_h is None or idx_l >= idx_h:
            return None, None  # Handle invalid profiles gracefully

        # Process the truncated profile
        part = p[idx_l:idx_h]
        part_resampled = sp.signal.resample(part, 90)
        part_short = part_resampled[:-5]

        # Polynomial detrending
        num = len(part_short)
        x_short = np.linspace(0, num, num)
        model = np.polyfit(x_short, part_short, 2)
        predicted = np.polyval(
            model,
            np.linspace(0, len(part_resampled), len(part_resampled)),
        )
        part_detrended = part_resampled - predicted

        # Thresholding
        mean_part = np.mean(part_detrended)
        std_part = np.std(part_detrended)
        l_thr, h_thr = mean_part - 2 * std_part, mean_part + 2 * std_part
        part_detrended[-5:] = np.where(
            (part_detrended[-5:] < l_thr) | (part_detrended[-5:] > h_thr),
            0,
            part_detrended[-5:],
        )

        # Smoothing
        kernel = np.ones(3) / 3  # Kernel size 3
        part_smoothed = np.convolve(
            part_detrended - mean_part,
            kernel,
            mode="same",
        )

        # Cross-correlation for alignment
        profile_all = np.sum(
            ACRLowContrastObjectDetectability.template_profile_separate(
                spoke_number,
            ),
            axis=1,
        )
        _, part_smoothed, _ = ACRLowContrastObjectDetectability.shift_for_max_corr(
            profile_all, part_smoothed, 5,
        )

        # Prepare GLM
        y = part_smoothed.reshape(90, 1)
        predicted = predicted.reshape(90, 1)
        profiles_with_bias = np.append(
            ACRLowContrastObjectDetectability.template_profile_separate(
                spoke_number,
            ),
            np.ones((90, 1)),
            axis=1,
        )

        # Fit the GLM
        glm_model = sm.api.GLM(y, profiles_with_bias)
        glm_results = glm_model.fit()

        # Return statistical results
        return glm_results.pvalues, glm_results.params


    @staticmethod
    def angle_image(
        alpha_degree: float,
        cX: int,
        cY: int,
        X: int,
        Y: int,
    ) -> np.ndarray:
        """Get an angle and generate a 1D profile of the image along the angle."""
        Im = np.zeros((X, Y))
        alpha = np.radians(alpha_degree)

        if alpha_degree == 90:
            Im[cX, cY:Y] = 1  # Direct assignment for vertical line
        elif 45 <= abs(alpha_degree) < 135:
            y_range = np.arange(cY, Y)
            x_values = cX + (y_range - cY) / np.tan(alpha)
            valid_indices = (x_values >= 0) & (x_values < X-1)
            Im[np.round(x_values[valid_indices]).astype(int), y_range[valid_indices]] = 1
        elif 135 <= abs(alpha_degree) < 225:
            x_range = np.arange(cX, X)
            y_values = cY + np.tan(alpha) * (x_range - cX)
            valid_indices = (y_values >= 0) & (y_values < Y-1)
            Im[x_range[valid_indices], np.round(y_values[valid_indices]).astype(int)] = 1
        elif 225 <= abs(alpha_degree) < 315:
            y_range = np.arange(0, cY)
            x_values = cX + (y_range - cY) / np.tan(alpha)
            valid_indices = (x_values >= 0) & (x_values < X-1)
            Im[np.round(x_values[valid_indices]).astype(int), y_range[valid_indices]] = 1
        else:
            x_range = np.arange(0, cX)
            y_values = cY + np.tan(alpha) * (x_range - cX)
            valid_indices = (y_values >= 0) & (y_values < Y-1)
            Im[x_range[valid_indices], np.round(y_values[valid_indices]).astype(int)] = 1

        return Im


    @staticmethod
    def angle_profile_radial(
        image: np.ndarray,
        image_profile: np.ndarray,
        cX: np.ndarray,
        cY: np.ndarray,
    ) -> pd.DataFrame:
        """Get the angular profile as a function of distance."""
        # Get indices where the profile image is not zero
        coords = np.argwhere(image_profile == 1)

        # Calculate the profile, distances, and store results
        X, Y = coords[:, 0], coords[:, 1]
        profile = image[X, Y]
        distances = np.sqrt((cX - X) ** 2 + (cY - Y) ** 2)

        # Create the DataFrame
        df = pd.DataFrame({
            "profile": profile,
            "X": X,
            "Y": Y,
            "distance": distances,
        })

        # Sort the DataFrame by distance
        df.sort_values(
            by="distance", inplace=True, ascending=True, ignore_index=True,
        )
        return df

    @staticmethod
    def truncate_profile(p: float, M: np.ndarray) -> tuple[float] | tuple[None]:
        """Get a 1D array of profile (p) and a background threshold (M).

        and eliminate backgrounds from the begining and end of the profile
        and returns the index of the first and last element with higher value
        than M
        """
        indices = np.where(p >= M)[0]
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
        pixel_size = 0.5
        radi = [12/0.5, 25/0.5, 38/0.5] ## in milimeter
        D = [7, 6.39, 5.78, 5.17, 4.55, 3.94, 3.33, 2.72, 2.11, 1.5]
        d = D[spoke_number-1]
        for i,r in enumerate(radi):
            profile[round(r-d):round(r+d),i] = 1
        return profile


    def shift_for_max_corr(x: int,y: int ,max_shift: int) -> tuple:
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
