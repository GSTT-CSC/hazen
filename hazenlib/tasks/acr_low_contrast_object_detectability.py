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

import functools
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pydicom

# Python imports
import logging
from concurrent import futures
from dataclasses import dataclass, field
from pathlib import Path

# Module imports
import cv2
import matplotlib.pyplot as plt
import nevergrad as ng
import numpy as np
import scipy as sp
import statsmodels as sm
import statsmodels.api
from hazenlib.ACRObject import ACRObject
from hazenlib.HazenTask import HazenTask
from hazenlib.types import Measurement, P_HazenTask, Result
from hazenlib.utils import get_pixel_size
from matplotlib.patches import Circle

logger = logging.getLogger(__name__)

@dataclass
class LowContrastObject:
    """Class for the Low Contrast Present within each spoke."""

    x: int
    y: int
    diameter: float
    value: float | None = None


@dataclass
class Spoke:
    """Spoke radiating from the center to the edge of the LCOD disk.

    Also contains the LowContrastObjects along the spoke.
    """

    cx: float
    cy: float
    theta: float        # Degrees

    diameter: float

    # Distance from the center (mm)
    dist: tuple[float] = (12.5, 25, 38.0)

    passed: bool = False

    objects: Sequence[LowContrastObject] = field(init=False)

    def __post_init__(self) -> None:
        """Initialise the objects within the spoke."""
        # y increases as the you down the image
        # hence the minus sign is so that theta = 0
        # corresponds to the object being visually
        # above the center rather than below.
        logger.debug("Creating objects along spoke angle: %f", self.theta)
        self.objects = tuple(
            LowContrastObject(
                x=self.cx + d * np.sin(np.deg2rad(self.theta)),
                y=self.cy - d * np.cos(np.deg2rad(self.theta)),
                diameter=self.diameter,
            )
            for d in self.dist
        )


    def __len__(self) -> int:
        """Objects within the spoke."""
        return len(self.objects)

    def __iter__(self) -> LowContrastObject:
        """Iterate over the low contrast objects."""
        return iter(self.objects)

    def profile(
        self,
        dcm: pydicom.FileDataset,
        size: int = 128,
        offset: tuple[float] = (0.0, 0.0),
    ) -> np.ndarray:
        """Return the 1D profile of the DICOM for the spoke."""
        px_x, px_y = get_pixel_size(dcm)
        if px_x != px_y:
            msg = "Only square pixels are supported"
            logger.critcal("%s but got (%f, %f)", msg, px_x, px_y)
            raise ValueError(msg)
        r_coords = np.linspace(
            0, self.diameter * px_x, size, endpoint=False,
        )
        theta_coords = np.zeros_like(r_coords) + self.theta

        x_coords = (
            r_coords * np.sin(np.deg2rad(theta_coords))
            + (self.cx + offset[1]) * px_x
        )
        y_coords = (
            r_coords * np.cos(np.deg2rad(theta_coords))
            + (self.cy + offset[0]) * px_y
        )

        return sp.ndimage.map_coordinates(
            dcm.pixel_array, [y_coords.ravel(), x_coords.ravel()], order=1,
        )


@dataclass
class LCODTemplate:
    """Template for the LCOD object."""

    # Position of the center of the LCOD on the image.
    cx: float
    cy: float

    # Angle of rotation for the 1st spoke (degrees)
    theta: float

    # Radius of each low contrast object (mm)
    # Starting with the largest spoke and moving clockwise.
    radi: tuple[float]  = (
        7.00, 6.39, 5.78, 5.17, 4.55, 3.94, 3.33, 2.72, 2.11, 1.50,
    )


    @functools.cached_property
    def spokes(self) -> Sequence[Spoke]:
        """Position of each of the low contrast object spokes."""
        return self._calc_spokes(
            self.cx, self.cy, self.theta, self.radi,
        )

    @staticmethod
    @functools.lru_cache(maxsize=10)
    def _calc_spokes(
        cx: float,
        cy: float,
        theta: float,
        radi: tuple[float],
    ) -> tuple[Spoke]:
        # Spokes start with the largest and decrease in size clockwise
        return tuple(
            Spoke(cx, cy, theta + i * (360 / len(radi)), 2 * r)
            for i, r in enumerate(radi)
        )

    def mask(
        self,
        dcm: pydicom.FileDataset,
        offset: tuple[float] = (0.0, 0.0),
        *,
        warn_if_object_out_of_bounds: bool = False,
    ) -> np.ndarray:
        """Mask the DICOM pixel array from the Spoke geometry.

        DICOM file is used for relating pixel geometry to
        the geometry of the acquisition.

        Similarly, offset (y, x) is used to account for cropped images.
        """
        mask = np.zeros_like(dcm.pixel_array, dtype=np.bool)

        # Convert from real coordinates to pixel coordinates
        dx, dy = get_pixel_size(dcm)
        y_grid, x_grid = np.meshgrid(
            *[
                np.linspace(v0, v0 + s * dv, num=s, endpoint=False)
                for v0, dv, s in zip(
                    offset, (dy, dx), mask.shape,
                )
            ],
            indexing="ij",
        )

        for sidx, spoke in enumerate(self.spokes):
            for oidx, obj in enumerate(spoke):
                # I genuinely have no idea why this correction factor is needed
                # but without it the objects in the mask have twice the radius
                # as the actual objects in the image.
                # TODO(@abdrysdale) : Remove the correction factor.
                cf = 2
                is_object = (
                    (y_grid - obj.y) ** 2 + (x_grid - obj.x) ** 2
                    <= (obj.diameter / (cf * 2)) ** 2
                )
                # Compare actual to measured area to check if object is on the
                # grid.
                mask_area = (cf ** 2) * np.sum(is_object) * (dx * dy)
                obj_area = np.pi * (obj.diameter / 2) ** 2
                if warn_if_object_out_of_bounds and not np.isclose(
                    mask_area, obj_area, rtol=1e-1, atol=dx * dy,
                ):
                    logger.warning(
                        "Object %d in spoke %d is out of bounds.\n"
                        "File: %s\n"
                        "Object area:\t%f\nMask area:\t%f\n"
                        "Object:\n\tCenter: (%f, %f)\n\tDiameter: %f\n"
                        "Image:\n\tPixel size: (%f, %f)\n\tShape: (%d, %d)",
                        oidx,
                        sidx,
                        dcm.filename,
                        obj_area, mask_area,
                        obj.x, obj.y, obj.diameter,
                        dx, dy, *mask.shape,
                    )
                mask |= is_object
        return mask


class ACRLowContrastObjectDetectability(HazenTask):
    """Low Contrast Object Detectability (LCOD) class for the ACR phantom."""

    OBJECTS_PER_SPOKE = 3
    NUM_SPOKES = 10

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
        self.slice_range = slice(11,7,-1)

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
        dcm: pydicom.Dataset,
        slice_no: int = -1,
        alpha: float = 0.05,
    ) -> np.ndarray:
        """Count the number of spokes using polar coordinate transformation."""
        # Input validation and preprocessing
        img_data = (
            np.abs(dcm.pixel_array)
            if np.min(dcm.pixel_array) < 0
            else dcm.pixel_array
        )

        if np.max(img_data) == 0:
            logger.warning("Image has zero intensity, returning zero spokes")
            return [0] * self.NUM_SPOKES

        # TODO(@abdrysdale): Apply smoothing before spoke detection
        # https://github.com/sbu-physics-mri/hazen-wales/issues/18

        # Find the position of each spoke
        # (with the pixel coordinates of the each of the object centers)
        spokes, (cx, cy), theta = self.find_spokes(
            dcm, return_center=True, return_theta=True,
        )

        for spoke_number, spoke in enumerate(spokes):
            p_vals, params = self._analyze_profile(
                spoke.profile(dcm, size=90),
                spoke_number,
            )

            if (p_vals is not None and params is not None and
                len(p_vals) >= len(spoke) and
                np.all(p_vals[:len(spoke)] < alpha) and
                np.all(params[:len(spoke)] > 0)
            ):
                spoke.passed = True

        # Generate report if requested
        if self.report:
            fig, axes = plt.subplots(1, 1, figsize=(8, 8))

            # Use robust intensity scaling
            mean_val = np.mean(dcm.pixel_array)
            std_val = np.std(dcm.pixel_array)
            vmin = max(0, mean_val - 2 * std_val)
            vmax = mean_val + 2 * std_val

            axes.imshow(dcm.pixel_array, cmap="gray", alpha=0.7, vmin=vmin, vmax=vmax)

            template = LCODTemplate(cx, cy, theta)

            # Highlight detected spokes
            mask = template.mask(dcm, warn_if_object_out_of_bounds=True)
            axes.imshow(mask, alpha=0.3 * mask)

            axes.scatter(cx, cy, marker="x", color="red", s=100)
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

        return lcod_center



    def find_spokes(
        self,
        dcm: pydicom.Dataset,
        center_search_tolerance: float = 0.05,
        *,
        random_state: np.random.RandomState | None = None,
        return_center: bool = False,
        return_theta: bool = False,
    ) -> Sequence[Spoke]:
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

        spokes = LCODTemplate(**values).spokes

        if any([return_center, return_theta]):
            return tuple(
                out
                for out, rtn in zip(
                        (
                            spokes,
                            (values["cx"], values["cy"]),
                            values["theta"],
                        ),
                        (True, return_center, return_theta),
                )
            )
        return spokes


    def _analyze_profile(
            self,
            profile:
            np.ndarray,
            spoke_number: int,
            *,
            std_tol: float = 0.01,
    ) -> tuple:
        """Analyze radial profile for low-contrast object detection.

        Args:
            profile: Radial intensity profile
            spoke_number: Spoke number (0-9)
            std_tol: Tolerance for the standard deviation for
                detecting polynomial coefficients.

        Returns:
            Tuple of (p-values, parameters) or (None, None) if analysis fails

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
        smoothed = np.convolve(detrended, kernel, mode="same")

        # Create design matrix with template
        template = self.template_profile_separate(spoke_number)
        # Include intercept
        data = np.column_stack([template, np.ones_like(profile)])

        # Fit linear model
        try:
            model = sm.OLS(smoothed, data).fit()
            return model.pvalues[:3], model.params[:3]
        except:
            return None, None


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
