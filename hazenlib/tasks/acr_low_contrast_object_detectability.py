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

import copy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pydicom

# Python imports
import logging
from concurrent import futures
from pathlib import Path

# Module imports
import cv2
import matplotlib.pyplot as plt
import nevergrad as ng
import numpy as np
import scipy as sp
import statsmodels
import statsmodels.api as sm
from hazenlib.ACRObject import ACRObject
from hazenlib.HazenTask import HazenTask
from hazenlib.types import LCODTemplate, Measurement, P_HazenTask, Result
from hazenlib.utils import get_pixel_size
from matplotlib.patches import Circle

logger = logging.getLogger(__name__)

class ACRLowContrastObjectDetectability(HazenTask):
    """Low Contrast Object Detectability (LCOD) class for the ACR phantom."""

    def __init__(
            self, alpha: float = 0.0125, **kwargs: P_HazenTask.kwargs,
    ) -> None:
        """Initialise the LCOD object."""
        if kwargs.pop("verbose", None) is not None:
            logger.warning(
                "verbose is not a supported argument for %s",
                type(self).__name__,
            )
        super().__init__(**kwargs)

        self.alpha = alpha

        # Start at last slice (highest contrast) and work backwards
        self.slice_range = slice(10, 6, -1)

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

        # Only used in reporting
        self.fig = None
        self.axes = None


    def run(self) -> Result:
        """Run the LCOD analysis."""
        results = self.init_result_dict()
        results.files = [
            self.img_desc(f)
            for f in self.ACR_obj.slice_stack[self.slice_range]
        ]

        total_spokes = 0
        for i, dcm in enumerate(self.ACR_obj.slice_stack[self.slice_range]):
            slice_no = self.slice_range.step * i + self.slice_range.start
            result = self.count_spokes(dcm, slice_no=slice_no, alpha=self.alpha)
            try:
                num_spokes = min(i for i, r in enumerate(result) if not r)
            except (IndexError):
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
        raw: pydicom.Dataset,
        slice_no: int = -1,
        alpha: float = 0.05,
    ) -> np.ndarray:
        """Count the number of spokes using polar coordinate transformation."""
        # TODO(@abdrysdale): Apply smoothing before spoke detection
        # https://github.com/sbu-physics-mri/hazen-wales/issues/18
        dcm = self._preprocess(raw)

        # Find the position of each spoke
        # (with the pixel coordinates of the each of the object centers)
        template = self.find_spokes(dcm)
        spokes  = template.spokes
        cx, cy = (template.cx, template.cy)
        dx, dy = get_pixel_size(dcm)
        theta = template.theta

        # Pre-process

        p_vals_all = []
        params_all = []
        for spoke in spokes:
            profile, (x_coords, y_coords), object_mask = spoke.profile(
                self._preprocess(dcm),
                size=90,
                return_coords=True,
                return_object_mask=True,
            )
            p_vals, params = self._analyze_profile(profile, object_mask)

            p_vals_all += list(p_vals)
            params_all += list(params)

        p_vals_fdr = statsmodels.stats.multitest.fdrcorrection(
            p_vals_all,
            alpha=alpha,
            method="indep",
            is_sorted=False,
        )[0].reshape(-1, len(p_vals))
        params_fdr = np.array(params_all).reshape(-1, len(params))

        # Check if the p-values pass the significance test
        spoke_can_pass = True
        for spoke_number, spoke in enumerate(spokes):

            for i, obj in enumerate(spoke):
                obj.detected = (
                    p_vals_fdr[spoke_number, i]
                    and params_fdr[spoke_number, i] > 0
                )
            spoke.passed = all(obj.detected for obj in spoke) and spoke_can_pass
            spoke_can_pass = spoke.passed

            if self.report:
                # Figure and axes should be obtained from analyze profile
                profile, (x_coords, y_coords), object_mask = spoke.profile(
                    dcm,
                    size=90,
                    return_coords=True,
                    return_object_mask=True,
                )
                _ = self._analyze_profile(profile, object_mask)

                self.fig.suptitle(
                    f"Slice {slice_no}, Spoke {spoke_number},"
                    f" Passed={spoke.passed}",
                )

                # Low contrast slice (no mask)
                vmin, vmax = self._window(raw)
                self.axes[0, 0].imshow(
                    raw.pixel_array, cmap="gray", vmin=vmin, vmax=vmax,
                )
                self.axes[0, 0].scatter(cx / dx, cy / dy, marker="o", c="y")
                self.axes[0, 0].scatter(
                    spoke.cx / dx, spoke.cy / dy, marker="o", c="r", s=0.05,
                )
                self.axes[0, 0].scatter(
                    x_coords, y_coords, marker="x", c="r", s=0.01,
                )


                vmin, vmax = self._window(dcm)
                self.axes[1, 0].imshow(
                    dcm.pixel_array, cmap="gray", vmin=vmin, vmax=vmax,
                )
                self.axes[1, 0].imshow(
                    template.mask(dcm),
                    alpha=template.mask(dcm) * 0.1,
                )
                self.axes[1, 0].scatter(cx / dx, cy / dy, marker="o", c="y")
                self.axes[1, 0].scatter(
                    spoke.cx / dx, spoke.cy / dy, marker="o", c="r", s=0.05,
                )
                self.axes[1, 0].scatter(
                    x_coords, y_coords, marker="x", c="r", s=0.01,
                )

                data_path = Path(self.dcm_list[0].filename).parent.name
                img_path = (
                    Path(self.report_path) /
                    (
                        f"spokes_{data_path}_slice_{str(slice_no).zfill(2)}"
                        f"_spoke_{str(spoke_number).zfill(2)}"
                        f"_{self.img_desc(dcm)}.png"
                    )
                )
                self.fig.savefig(img_path, dpi=150)
                self.report_files.append(img_path)

                self.fig = None
                self.axes = None
                plt.close()

        # Generate report if requested
        if self.report:
            fig, axes = plt.subplots(1, 1, figsize=(8, 8))

            # Use intensity scaling
            vmin, vmax = self._window(dcm)

            axes.imshow(
                dcm.pixel_array, cmap="gray", alpha=1, vmin=vmin, vmax=vmax,
            )

            template = LCODTemplate(cx, cy, theta)

            # Highlight detected spokes
            mask = template.mask(
                dcm,
                subset="passed",
                warn_if_object_out_of_bounds=True,
            )
            axes.imshow(mask, alpha=0.3 * mask, cmap="Greens")

            # Highlight undetected spokes
            mask = template.mask(
                dcm,
                subset="failed",
                warn_if_object_out_of_bounds=True,
            )
            axes.imshow(mask, alpha=0.3 * mask, cmap="Reds")

            axes.scatter(cx / dx, cy / dy, marker="x", color="red", s=100)
            axes.set_title(
                f"Slice {slice_no}"
                f" Detected Spokes {int(np.sum(s.passed for s in spokes))}/10",
                fontsize=14,
            )
            axes.set_xlabel("Pixel")
            axes.set_ylabel("Pixel")

            data_path = Path(self.dcm_list[0].filename).parent.name
            img_path = (
                Path(self.report_path) /
                f"spokes_{data_path}_{self.img_desc(dcm)}_slice_{slice_no}.png"
            )
            fig.savefig(img_path, dpi=150)
            plt.close()

            self.report_files.append(img_path)

        return [s.passed for s in spokes]


    def find_center(
        self,
        crop_ratio: float = 0.7,
    ) -> tuple[float]:
        """Find the center of the LCOD phantom."""
        if self.lcod_center is not None:
            return self.lcod_center

        dcm = self.ACR_obj.slice_stack[0]
        (main_cx, main_cy), main_radius = self.ACR_obj.find_phantom_center(
            dcm.pixel_array, self.ACR_obj.dx, self.ACR_obj.dy,
        )

        dcm = self.ACR_obj.slice_stack[-1]
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

        lcod_center = tuple(
            (dc + max(0, int(main_c - r))) * dv
            for dc, main_c, dv in zip(
                detected_circles[:2],
                (main_cx, main_cy),
                (self.ACR_obj.dx, self.ACR_obj.dy),
            )
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
            self.report_files.append(img_path)

        return lcod_center



    def find_spokes(
        self,
        dcm: pydicom.Dataset,
        center_search_tolerance: float = 0.05,
        *,
        random_state: np.random.RandomState | None = None,
    ) -> LCODTemplate:
        """Find the position of the spokes within the LCOD disk."""
        # Optimisation parameters that are hard-coded
        # to ensure standardisation.
        optimiser: str = "MultiSQPPlus"
        budget: int = 1000
        initial_rotation_offset: float = 25

        def minimiser(cx: float, cy: float, theta: float) -> float:
            template = LCODTemplate(cx, cy, theta)
            return - np.sum(template.mask(dcm) * dcm.pixel_array)

        theta_0 = self.rotation + initial_rotation_offset
        theta_p = ng.p.Scalar(
            init=float(theta_0),
            lower=theta_0 - 18,
            upper=theta_0 + 18,
        )

        if self.lcod_center is None:
            cx_0, cy_0 = self.find_center()

            parametrization = ng.p.Instrumentation(
                cx=ng.p.Scalar(
                init=float(cx_0),
                lower=float(cx_0) * (1 - center_search_tolerance),
                upper=float(cx_0) * (1 + center_search_tolerance),
                ),
                cy=ng.p.Scalar(
                    init=float(cy_0),
                    lower=float(cy_0) * (1 - center_search_tolerance),
                    upper=float(cy_0) * (1 + center_search_tolerance),
                ),
                theta=theta_p,
            )

        else:
            cx, cy = self.lcod_center
            parametrization = ng.p.Instrumentation(
                cx=cx, cy=cy, theta=theta_p,
            )


        if random_state is not None:
            parametrization.random_state = random_state

        opt = ng.optimizers.registry[optimiser](
            parametrization=parametrization,
            budget=budget,
            num_workers=4,
        )

        with futures.ThreadPoolExecutor(max_workers=opt.num_workers) as executor:
            recommendation = opt.minimize(
                minimiser, executor=executor, batch_mode=False,
            )

        _, values = recommendation.value
        if self.lcod_center is not None:
            values["cx"] = self.lcod_center[0]
            values["cy"] = self.lcod_center[1]
        else:
            self.lcod_center = (values["cx"], values["cy"])

        return LCODTemplate(**values)


    def _analyze_profile(
            self,
            profile: np.ndarray,
            object_mask: np.ndarray,
            *,
            std_tol: float = 0.01,
    ) -> tuple:
        """Analyze radial profile for low-contrast object detection.

        Args:
            profile: Radial intensity profile
            object_mask : A 3 x N array containing boolean mask of each object.
            std_tol: Tolerance for the standard deviation for
                detecting polynomial coefficients.

        Returns:
            Tuple of (p-values, parameters).

        """
        # De-trend with robust polynomial fitting
        if np.std(profile) > std_tol:
            x = np.linspace(0, 1, len(profile))
            # Use lower order polynomial for stability
            coeffs = np.polyfit(x, profile, 2)
            trend = np.polyval(coeffs, x)
        else:
            trend = np.zeros_like(profile)

        detrended = profile - trend

        # Simple smoothing
        kernel = np.ones(3) / 3
        smoothed = np.convolve(
            detrended - np.mean(detrended), kernel, mode="same",
        ).reshape((profile.size, 1))

        # Prepare GLM
        data = np.column_stack((object_mask, np.ones_like(profile)))

        # Fit GLM
        model = sm.GLM(smoothed, data).fit()

        # Reporting
        if self.report:
            plt.clf()
            self.fig, self.axes = plt.subplots(2, 2, figsize=(16, 16))

            x = np.arange(profile.size)

            self.axes[0, 1].plot(x, profile, label="Profile")

            for i in range(3):
                self.axes[0, 1].plot(
                    x[object_mask[:, i]],
                    profile[object_mask[:, i]],
                    label=f"Object {i+1}",
                )
            self.axes[0, 1].plot(x, trend, label="Trend", linestyle="dashed")
            self.axes[0, 1].legend()
            self.axes[0, 1].set_title("Radial Profile")

            self.axes[1, 1].plot(x, smoothed, label="De-trended")
            for i in range(3):
                obj_indexes = object_mask[:, i]
                self.axes[1, 1].plot(
                    x[obj_indexes],
                    smoothed[obj_indexes],
                    label=f"Object {i+1}",
                )

                idx_max = np.argmax(smoothed[obj_indexes])
                y_max = smoothed[obj_indexes][idx_max] * 0.95
                x_max = x[obj_indexes][idx_max]
                x_text = x_max - (np.max(x) - np.min(x)) * 0.05
                y_text = (
                    (np.min(smoothed) + np.min(smoothed[obj_indexes])) / 2
                )
                self.axes[1, 1].annotate(
                    f"p = {model.pvalues[i]:.4f}",
                    (x_max, y_max),    # Annotation
                    (x_text, y_text),   # Text
                    arrowprops={
                        "arrowstyle": "->", "connectionstyle": "arc",
                    },
                )
            self.axes[1, 1].legend()
            self.axes[1, 1].set_title("De-trended profile")

        return model.pvalues[:3], model.params[:3]


    @staticmethod
    def _window(dcm: pydicom.FileDataset) -> tuple[float]:
        """Return vmin, vmax values based on simple window method."""
        mean_val = np.mean(dcm.pixel_array)
        std_val = np.std(dcm.pixel_array)
        vmin = max(0, mean_val - 2 * std_val)
        vmax = mean_val + 2 * std_val
        return (vmin, vmax)


    @staticmethod
    def _preprocess(
        dcm: pydicom.FileDataset,
        threshold_min: float = 0.05,
        threshold_max: float = 0.65,
        threshold_step: float = 0.001,
        lower: float = 0.1,
        upper: float = 0.2,
    ) -> pydicom.FileDataset:
        """Preprocess the DICOM."""
        processed = copy.deepcopy(dcm)
        data = processed.pixel_array

        fdata = data / np.max(data)    # Normalise

        # Threshold
        structure = np.ones((3, 3), dtype=int)
        for thr in np.arange(threshold_min, threshold_max, threshold_step):
            ret, thresh = cv2.threshold(fdata, thr, 1, 0)
            labelled, ncomponents = sp.ndimage.measurements.label(
                thresh, structure,
            )
            thresh_inner = labelled == np.max(labelled)
            if lower < np.sum(thresh_inner != 0) / np.sum(fdata != 0) < upper:
                break
        data *= thresh_inner

        processed.set_pixel_data(
            data,
            dcm[(0x0028,0x0004)].value, # Photometric Interpretation
            dcm[(0x0028,0x0101)].value, # Bits Stored
        )
        return processed
