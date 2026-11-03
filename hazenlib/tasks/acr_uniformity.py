"""
ACR Uniformity

https://www.acraccreditation.org/-/media/acraccreditation/documents/mri/largephantomguidance.pdf

Calculates the percentage integral uniformity for slice 7 of the ACR phantom.

This script calculates the percentage integral uniformity in accordance with
the ACR Guidance. A large circular ROI of 195 cm2 is placed in the phantom
and the mean of a 1 cm2 circular ROI is calculated at every position where it
lies fully inside the large ROI. The 1 cm2 ROIs with the lowest and highest
mean are used to calculate the integral uniformity:

    PIU = 100 * (1 - (max - min) / (max + min))

Method
------
0. The phantom orientation is determined from slice 1 (as in the geometric
   accuracy task) and slice 7 is rotated by a multiple of 90 degrees (and
   mirrored if needed) into the standard view, with the notch at the top.
   This matters because the large ROI is placed relative to the notch, and
   the report is then shown in the standard view.
1. Rescaled pixel values are used, so a DICOM rescale intercept cannot
   distort the ratio.
2. The phantom centre is found from its edges (as in the geometric accuracy
   task), to sub-pixel precision.
3. The large ROI is a 195 cm2 circle centred on the phantom (the lower end
   of the ACR 195-205 cm2 range). Its top is 78.8 mm above the centre, which
   is within about a millimetre of the bottom of the notch at 12 o'clock
   (79-80 mm above the centre on the test data), so the blurred edge of the
   notch can reach into it. Pixels within 2 mm of anything dark (below half
   the median signal in the ROI) are therefore left out of the large ROI;
   this removes the notch edge, and air bubbles if there are any.
4. The 1 cm2 ROI is a circle of radius 5.64 mm. All ROI geometry is in mm, so
   it is correct for any (including non-square) pixel size.
5. The mean of the small ROI is computed at every pixel by correlation with
   a normalised circular kernel; only positions where the whole small ROI
   lies inside the large ROI are considered.

Created by Yassine Azma
yassine.azma@rmh.nhs.uk

13/01/2022

90 degree orientation correction, ROI geometry in mm, accurate phantom
centre, notch-free large ROI and rescaled pixel values 08/10/2026
"""

import os
import sys
import traceback

import numpy as np
import scipy.ndimage as ndi
from hazenlib.ACRObject import ACRObject
from hazenlib.HazenTask import HazenTask
from hazenlib.logger import logger
from hazenlib.tasks.acr_geometric_accuracy import (
    describe_orientation,
    determine_phantom_orientation,
    measure_phantom,
    reorient,
    reorient_spacing,
)
from hazenlib.types import Measurement, Result


class ACRUniformity(HazenTask):
    """Uniformity measurement class for DICOM images of the ACR phantom."""

    SLICE_INDEX = 6  # slice 7
    LARGE_ROI_AREA = 19500.0  # mm2 (195 cm2)
    LARGE_ROI_OFFSET = 0.0  # mm below the phantom centre
    SMALL_ROI_AREA = 100.0  # mm2 (1 cm2)
    DARK_FRACTION = 0.5  # of the median signal; darker pixels are excluded
    DARK_MARGIN = 2.0  # mm excluded around dark pixels (the notch edge)

    def __init__(self, **kwargs):
        if kwargs.pop("verbose", None) is not None:
            logger.warning(
                "verbose is not a supported argument for %s",
                type(self).__name__,
            )
        super().__init__(**kwargs)

        # Initialise ACR object
        self.ACR_obj = ACRObject(self.dcm_list)
        self._orientation = None
        self.r_large = float(np.sqrt(self.LARGE_ROI_AREA / np.pi))  # mm
        self.r_small = float(np.sqrt(self.SMALL_ROI_AREA / np.pi))  # mm

    # ------------------------------------------------------------------
    # Orientation and spacing
    # ------------------------------------------------------------------
    @property
    def pixel_spacing(self):
        """(row, column) spacing in mm as stored in the DICOM, before any
        reorientation. ACRObject.dx/dy hold PixelSpacing[0]/[1]."""
        return (self.ACR_obj.dx, self.ACR_obj.dy)

    @property
    def orientation(self):
        """(k, flip) correction putting the phantom in its standard view.

        Determined from slice 1, which has strongly asymmetric features.
        """
        if self._orientation is None:
            self._orientation = determine_phantom_orientation(
                self.ACR_obj.slice_stack[0].pixel_array, self.pixel_spacing
            )
            if self._orientation == (0, False):
                logger.info("Phantom is in the standard orientation")
            else:
                logger.warning(
                    "Phantom orientation is non-standard; images %s for "
                    "analysis",
                    describe_orientation(*self._orientation),
                )
        return self._orientation

    @property
    def spacing(self):
        """(row, column) spacing in mm of the reoriented image."""
        return reorient_spacing(self.pixel_spacing, self.orientation[0])

    # ------------------------------------------------------------------
    # Task
    # ------------------------------------------------------------------
    def run(self) -> Result:
        """Main function for performing uniformity measurement using slice 7 from the ACR phantom image set.

        Returns:
            dict: results are returned in a standardised dictionary structure specifying the task name, input DICOM
                Series Description + SeriesNumber + InstanceNumber, task measurement key-value pairs, optionally path
                to the generated images for visualisation
        """
        dcm = self.ACR_obj.slice_stack[self.SLICE_INDEX]
        # Initialise results dictionary
        results = self.init_result_dict(desc=self.ACR_obj.acquisition_type())
        results.files = self.img_desc(dcm)

        try:
            result = self.get_integral_uniformity(dcm)
            results.add_measurement(
                Measurement(
                    name="Uniformity",
                    type="measured",
                    subtype="Integral uniformity",
                    unit="%",
                    value=round(result, 2),
                ),
            )
            for label, (_, _, value) in (
                ("minimum", self.min_roi),
                ("maximum", self.max_roi),
            ):
                results.add_measurement(
                    Measurement(
                        name="Uniformity",
                        type="measured",
                        subtype="ROI mean",
                        description=f"1 cm2 ROI with the {label} mean",
                        value=round(value, 2),
                        visibility="intermediate",
                    ),
                )
        except Exception as e:
            logger.exception(
                "Could not calculate the percent integral uniformity for %s"
                " because of : %s",
                self.img_desc(dcm),
                e,
            )
            traceback.print_exc(file=sys.stdout)

        # only return reports if requested
        if self.report:
            results.add_report_image(self.report_files)

        return results

    @staticmethod
    def calculate_uniformity(min_value, max_value):
        """Calculates the percent integral uniformity (PIU) using formula `PIU = 100 * (1 - (max - min) / (max + min))`.

        Args:
            min_value (float): minimum mean value.
            max_value (float): maximum mean value.

        Returns:
            float: value of integral uniformity.

        """
        return float(
            100 * (1 - (max_value - min_value) / (max_value + min_value))
        )

    # ------------------------------------------------------------------
    # ROIs
    # ------------------------------------------------------------------
    def large_roi_mask(self, img, centre):
        """Boolean mask of the large ROI.

        A 195 cm2 circle centred on the phantom, without the pixels within
        DARK_MARGIN mm of anything darker than DARK_FRACTION of the median
        signal in the circle (the edge of the notch, air bubbles).

        Args:
            img (np.ndarray): slice 7 in the standard view.
            centre (tuple): phantom centre (x, y) in pixels.
        """
        row_sp, col_sp = self.spacing
        rows, cols = np.ogrid[: img.shape[0], : img.shape[1]]
        x_mm = (cols - centre[0]) * col_sp
        y_mm = (rows - centre[1]) * row_sp - self.LARGE_ROI_OFFSET
        circle = x_mm**2 + y_mm**2 <= self.r_large**2
        dark = img < self.DARK_FRACTION * np.median(img[circle])
        near_dark = (
            ndi.distance_transform_edt(~dark, sampling=self.spacing)
            <= self.DARK_MARGIN
        )
        excluded = circle & near_dark
        if excluded.any():
            logger.info(
                "%.1f cm2 near dark features (the notch) left out of the"
                " large ROI",
                excluded.sum() * row_sp * col_sp / 100,
            )
        return circle & ~near_dark

    def small_roi_kernel(self):
        """Boolean kernel of the 1 cm2 circular ROI, in pixels.

        Pixels whose centres lie within 5.64 mm of the kernel centre; for
        non-square pixels the kernel is elliptical in pixel units.
        """
        row_sp, col_sp = self.spacing
        half_rows = int(np.ceil(self.r_small / row_sp))
        half_cols = int(np.ceil(self.r_small / col_sp))
        rows, cols = np.ogrid[
            -half_rows : half_rows + 1, -half_cols : half_cols + 1
        ]
        return (rows * row_sp) ** 2 + (cols * col_sp) ** 2 <= self.r_small**2

    def get_mean_roi_values(self, img, large_roi):
        """Find the 1 cm2 ROIs with the lowest and highest mean.

        Below is an excerpt from the ACR Large Phantom Guidance

        For each series, the measurements are made according to the following procedure:

            #. Display slice location 7.
            #. Place a large, circular region-of-interest (ROI) on the image as shown in Figure 15.
                *. The area of the ROI depends on whether the large (200 cm2) or medium (160 cm2) phantom
                    was scanned (Table 4).
                *. This large ROI defines the boundary of the region in which the image uniformity is measured.
                *. Although the mean pixel intensity inside this ROI is not needed for the uniformity test, it is
                    used in the percent signal ghosting test (section 6.0), so it should be noted.
            #. Set the display window to its minimum, and lower the level until the entire area inside the large ROI is
            white.
                *. The goal now is to raise the level slowly until a small, roughly 1 cm2 region of dark pixels
                    develops inside the ROI. This is the region of lowest signal in the large ROI.
                *. Sometimes more than one region of dark pixels will appear. In that case, focus attention on the
                    largest dark region.
                *. In some cases, rather than having a well-defined dark region, one or more wide, poorly defined
                    dark areas or areas of mixed black and white pixels are apparent.
                *. In that case, make a visual estimate of the location of the darkest 1 cm2 portion of the largest
                    dark area should be made.
            #. Place a 1 cm2 circular ROI on the low-signal region identified in step 3.
                *. If measuring in the Medium phantom be sure that this ROI does not include any of the notch at
                the top of the phantom.
                *. Figures 16a and 17a show what typical Large and Medium phantom images look like at this
                    point.
                *. Record the mean pixel value for this 1 cm2 ROI. This is the measured low-signal value.
                *. If there is uncertainty about where to place the ROI because there is no single obviously darkest
                location, try several locations and select the one having the lowest mean pixel value.
            #. Raise the level until all but a small, roughly 1 cm2 region of white pixels remains inside the large ROI.
                *. This is the region of highest signal.
                *. Sometimes more than 1 region of white pixels will remain.
                    In that case, focus attention on the largest white region. It can happen that rather than having a
                    well-defined white region, one ends up with 1 or more diffuse areas of mixed black and white pixels.
                    In that case, make a best estimate of the location of the brightest 1 cm2 portion of the largest
                    bright area.
            #. Place a 1 cm2 circular ROI on the high-signal region identified in step 5.
                *. Record the average pixel value for this 1 cm2 ROI. This is the measured high-signal value.
                *. If there is uncertainty about where to place the ROI because there is no single obviously
                    brightest location, try several locations and select the one having the highest mean pixel value.

        Args:
            img (np.ndarray): slice 7 in the standard view.
            large_roi (np.ndarray): boolean mask of the large ROI.

        Returns:
            tuple: x_min, y_min, min_mean, x_max, y_max, max_mean, with the
            positions (x, y) of the small ROI centres in pixels.
        """
        kernel = self.small_roi_kernel()
        # Positions where the whole small ROI lies inside the large ROI
        valid = ndi.binary_erosion(large_roi, structure=kernel)
        if not valid.any():
            msg = "The large ROI is too small to hold a 1 cm2 ROI"
            raise ValueError(msg)
        means = ndi.correlate(
            np.asarray(img, dtype=float),
            kernel / kernel.sum(),
            mode="constant",
        )
        values = np.where(valid, means, np.nan)
        i_min = np.unravel_index(np.nanargmin(values), values.shape)
        i_max = np.unravel_index(np.nanargmax(values), values.shape)
        return (
            int(i_min[1]),
            int(i_min[0]),
            float(values[i_min]),
            int(i_max[1]),
            int(i_max[0]),
            float(values[i_max]),
        )

    def get_integral_uniformity(self, dcm):
        """Calculates the percent integral uniformity (PIU) of a DICOM pixel array. \n
        Moves a 1 cm^2 ROI through the 195 cm^2 large ROI inside the phantom
        and calculates its mean at every position. \n
        The PIU is defined as: `PIU = 100 * (1 - (max - min) / (max + min))`, where \n
        'max' and 'min' represent the maximum and minimum of the means of the 1 cm^2 ROIs.

        Args:
            dcm (pydicom.Dataset): DICOM image object to calculate uniformity from.

        Returns:
            float: value of integral uniformity.
        """
        _, rescaled, _ = self.ACR_obj.get_presentation_pixels(dcm)
        img = np.ascontiguousarray(
            reorient(np.asarray(rescaled, dtype=float), *self.orientation)
        )
        centre, phantom_mask, _ = measure_phantom(
            img, self.spacing, ("Horizontal", "Vertical")
        )
        logger.info("Phantom centre (%.1f, %.1f)", *centre)

        large_roi = self.large_roi_mask(img, centre)
        if phantom_mask[large_roi].mean() < 0.99:
            logger.warning("The large ROI extends beyond the phantom")
        x_min, y_min, min_value, x_max, y_max, max_value = (
            self.get_mean_roi_values(img, large_roi)
        )
        self.min_roi = (x_min, y_min, min_value)
        self.max_roi = (x_max, y_max, max_value)
        self.large_roi_mean = float(img[large_roi].mean())
        self.large_roi = large_roi
        logger.info("Mean Min ROI => (%i, %i) = %.2f", x_min, y_min, min_value)
        logger.info("Mean Max ROI => (%i, %i) = %.2f", x_max, y_max, max_value)

        # Uniformity calculation
        piu = self.calculate_uniformity(min_value, max_value)
        logger.info(
            "PIU = 100 * (1 - (%.2f - %.2f) / (%.2f + %.2f)) = %.2f",
            max_value,
            min_value,
            max_value,
            min_value,
            piu,
        )

        if self.report:
            logger.info("Writing report ... ")
            self.write_report(img, centre, piu, dcm)

        return piu

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    def write_report(self, img, centre, piu, dcm):
        import matplotlib.pyplot as plt

        row_sp, col_sp = self.spacing
        cx, cy = centre
        theta = np.linspace(0, 2 * np.pi, 360)

        def circle(x_px, y_px, r_mm):
            return (
                x_px + r_mm * np.cos(theta) / col_sp,
                y_px + r_mm * np.sin(theta) / row_sp,
            )

        fig, axes = plt.subplots(1, 2, figsize=(16, 8.5), layout="constrained")
        x_min, y_min, min_value = self.min_roi
        x_max, y_max, max_value = self.max_roi
        windows = (
            (None, None, "Slice 7"),
            (
                min_value - 0.05 * (max_value - min_value),
                max_value + 0.05 * (max_value - min_value),
                "Windowed from the minimum to the maximum ROI mean",
            ),
        )
        for ax, (vmin, vmax, label) in zip(axes, windows):
            ax.imshow(
                img, cmap="gray", vmin=vmin, vmax=vmax, aspect=row_sp / col_sp
            )
            # outline of the large ROI (the notch edge is cut out of it)
            ax.contour(self.large_roi, [0.5], colors="tab:blue", linewidths=1)
            ax.scatter(cx, cy, c="tab:blue", s=10)
            for (x, y, value), name, colour in (
                (self.min_roi, "Min", "tab:red"),
                (self.max_roi, "Max", "tab:green"),
            ):
                ax.plot(*circle(x, y, self.r_small), c=colour, lw=1.5)
                ax.annotate(
                    f"{name} = {value:.1f}",
                    (x + (self.r_small + 2) / col_sp, y),
                    c=colour,
                    va="center",
                    bbox=dict(facecolor="white", alpha=0.7, lw=0),
                )
            ax.set_title(label)
            ax.axis("off")

        title = f"Percent Integral Uniformity = {piu:.2f}%"
        if self.orientation != (0, False):
            title += (
                f"\n(image {describe_orientation(*self.orientation)}"
                " to standard view)"
            )
        fig.suptitle(title)

        img_path = os.path.realpath(
            os.path.join(self.report_path, f"{self.img_desc(dcm)}.png")
        )
        fig.savefig(img_path, dpi=80)
        plt.close(fig)
        self.report_files.append(img_path)
        return img_path
