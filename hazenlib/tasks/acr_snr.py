"""
ACR SNR

Calculates the SNR for slice 7 (the uniformity slice) of the ACR phantom.

This script utilises the smoothed subtraction method described in McCann 2013 [1], and a standard subtraction SNR.

Method
------
0. The phantom orientation is determined from slice 1 (as in the geometric
   accuracy task) and slice 7 is rotated by a multiple of 90 degrees (and
   mirrored if needed) into the standard view. SNR itself does not depend on
   this (the ROI layout is symmetric), but the report is then shown in the
   standard view. For the subtraction method each series is reoriented
   using its own slice 1, so the two images line up even if only one of
   them was exported rotated.
1. Rescaled pixel values are used, so a DICOM rescale intercept cannot
   distort the signal.
2. The phantom centre is found from its edges (as in the geometric accuracy
   task).
3. Five 20 x 20 mm square ROIs are used: one at the centre and one in each
   quadrant, centred 40 mm from the centre horizontally and vertically. The
   geometry is in mm, so it is the same for any matrix or pixel size.
4. Smoothing method: the noise image is the image minus a 25 x 25 pixel
   moving average of it (computed in floating point). Subtraction method:
   the noise image is the difference of the two images, and its standard
   deviation is divided by sqrt(2).
5. SNR is the mean over the five ROIs of (mean signal / noise SD).
6. The normalised SNR divides out voxel volume, bandwidth, averages, TR and
   the number of phase encoding steps. The numbers of frequency and phase
   encoding steps are taken along the right image axes using the
   in-plane phase encoding direction.

Created by Neil Heraghty (Adapted by Yassine Azma, yassine.azma@rmh.nhs.uk)

09/01/2023

ROI geometry in mm, 90 degree orientation correction, rescaled pixel values
and phase encoding direction in the normalisation 08/10/2026

[1] McCann, A. J., Workman, A., & McGrath, C. (2013). A quick and robust
method for measurement of signal-to-noise ratio in MRI. Physics in Medicine
& Biology, 58(11), 3775.
"""

# Typing imports
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pydicom

# Python imports
import contextlib
import os
import sys
import traceback
from pathlib import Path
from typing import Any

# Module imports
import numpy as np
from scipy import ndimage

# Local imports
import hazenlib.utils
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
from hazenlib.utils import dcmread, get_dicom_files


class ACRSNR(HazenTask):
    """Signal-to-noise ratio measurement class for DICOM images of the ACR phantom."""

    SLICE_INDEX = 6  # slice 7
    ROI_SIZE = 20.0  # mm, side of each square ROI
    ROI_OFFSET = 40.0  # mm, offset of the outer ROIs from the centre
    FILTER_SIZE = 25  # pixels, moving average for the smoothing method

    def __init__(
        self,
        subtract: Path | str | None = None,
        measured_slice_width: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialise the Hazen ACR SNR Object."""
        if kwargs.pop("verbose", None) is not None:
            logger.warning(
                "verbose is not a supported argument for %s",
                type(self).__name__,
            )
        super().__init__(**kwargs)
        self.ACR_obj = ACRObject(self.dcm_list)
        self._orientation = None

        # measured slice width is expected to be a floating point number
        self.measured_slice_width = measured_slice_width
        with contextlib.suppress(TypeError):
            self.measured_slice_width = float(measured_slice_width)

        # subtract is expected to be a path to a folder
        try:
            self.subtract = Path(subtract)
        except TypeError:
            self.subtract = None
        else:
            if not self.subtract.is_dir():
                self.subtract = None

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
        """(k, flip) correction putting the phantom in its standard view."""
        if self._orientation is None:
            self._orientation = self.find_orientation(self.ACR_obj)
        return self._orientation

    @staticmethod
    def find_orientation(acr_obj):
        """Orientation correction for a series, from its slice 1."""
        orientation = determine_phantom_orientation(
            acr_obj.slice_stack[0].pixel_array, (acr_obj.dx, acr_obj.dy)
        )
        if orientation == (0, False):
            logger.info("Phantom is in the standard orientation")
        else:
            logger.warning(
                "Phantom orientation is non-standard; images %s for analysis",
                describe_orientation(*orientation),
            )
        return orientation

    @property
    def spacing(self):
        """(row, column) spacing in mm of the reoriented image."""
        return reorient_spacing(self.pixel_spacing, self.orientation[0])

    def standard_view(self, dcm, orientation=None):
        """Rescaled pixel values of a slice, in the standard view."""
        _, rescaled, _ = self.ACR_obj.get_presentation_pixels(dcm)
        k, flip = self.orientation if orientation is None else orientation
        return np.ascontiguousarray(
            reorient(np.asarray(rescaled, dtype=float), k, flip)
        )

    # ------------------------------------------------------------------
    # Task
    # ------------------------------------------------------------------
    def run(self) -> Result:
        """Main function for performing SNR measurement using slice 7 from the ACR phantom image set. Performs either
        smoothing or subtraction method depending on user-provided input.

        Notes:
            Uses the smoothing method by default or the subtraction method if a second set of images are provided (using the --subtract option with dataset in a separate folder).

        Returns:
            dict: results are returned in a standardised dictionary structure specifying the task name, input DICOM Series Description + SeriesNumber + InstanceNumber, task measurement key-value pairs, optionally path to the generated images for visualisation.
        """
        # Identify relevant slice
        snr_dcm = self.ACR_obj.slice_stack[self.SLICE_INDEX]
        # Initialise results dictionary
        results = self.init_result_dict(desc=self.ACR_obj.acquisition_type())

        # SINGLE METHOD (SMOOTHING)
        if self.subtract is None:
            logger.debug(
                "Calculating SNR on %s with the single (smoothing) method",
                self.img_desc(snr_dcm),
            )
            try:
                results.files = self.img_desc(snr_dcm)
                snr, normalised_snr = self.snr_by_smoothing(
                    snr_dcm, self.measured_slice_width
                )
                results.add_measurement(
                    Measurement(
                        name="SNR",
                        type="measured",
                        subtype="smoothing",
                        value=round(snr, 2),
                        visibility="intermediate",
                    ),
                )
                results.add_measurement(
                    Measurement(
                        name="SNR",
                        type="normalised",
                        subtype="smoothing",
                        value=round(normalised_snr, 2),
                    ),
                )

            except Exception as e:
                logger.exception(
                    "Could not calculate the SNR for %s because of : %s",
                    self.img_desc(snr_dcm),
                    e,
                )
                traceback.print_exc(file=sys.stdout)
        # SUBTRACTION METHOD
        else:
            # Use the shared helper to get all DICOM files from the directory provided
            filepaths = get_dicom_files(Path(self.subtract))
            acr_obj2 = ACRObject([dcmread(dicom) for dicom in filepaths])
            snr_dcm2 = acr_obj2.slice_stack[self.SLICE_INDEX]

            logger.debug(
                "Calculating SNR on %s and %s with the subtraction method",
                self.img_desc(snr_dcm),
                self.img_desc(snr_dcm2),
            )

            results.files = [self.img_desc(snr_dcm), self.img_desc(snr_dcm2)]
            try:
                snr, normalised_snr = self.snr_by_subtraction(
                    snr_dcm,
                    snr_dcm2,
                    self.measured_slice_width,
                    orientation2=self.find_orientation(acr_obj2),
                )

                results.add_measurement(
                    Measurement(
                        name="SNR",
                        type="measured",
                        subtype="subtraction",
                        value=round(snr, 2),
                        visibility="intermediate",
                    ),
                )
                results.add_measurement(
                    Measurement(
                        name="SNR",
                        type="normalised",
                        subtype="subtraction",
                        value=round(normalised_snr, 2),
                    ),
                )

            except Exception as e:
                logger.exception(
                    "Could not calculate the SNR for %s and %s"
                    " because of : %s",
                    self.img_desc(snr_dcm),
                    self.img_desc(snr_dcm2),
                    e,
                )
                traceback.print_exc(file=sys.stdout)

        # only return reports if requested
        if self.report:
            results.add_report_image(self.report_files)

        return results

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------
    @staticmethod
    def encoding_steps(dcm) -> tuple:
        """Numbers of (frequency, phase) encoding steps of the image matrix.

        Uses the in-plane phase encoding direction: "COL"/"COLUMN" means the
        phase is encoded along the columns (vertically), so there are Rows
        phase steps and Columns frequency samples; "ROW" is the other way
        round. If the direction is unknown, columns are assumed to be the
        frequency direction.
        """
        rows = hazenlib.utils.get_rows(dcm)
        columns = hazenlib.utils.get_columns(dcm)
        try:
            direction = str(hazenlib.utils.get_pe_direction(dcm)).upper()
        except (AttributeError, KeyError, IndexError):
            direction = ""
        if direction.startswith("COL"):
            return columns, rows
        if direction == "ROW":
            return rows, columns
        logger.warning(
            "Phase encoding direction not found; assuming frequency encoding"
            " along the columns for the SNR normalisation"
        )
        return columns, rows

    def get_normalised_snr_factor(
        self, dcm, measured_slice_width=None
    ) -> float:
        """Calculates the normalisation factor to be applied to the SNR in order to obtain the absolute SNR (ASNR). The
        normalisation factor depends on voxel size, bandwidth, number of averages and number of phase encoding steps.

        Args:
            dcm (pydicom.Dataset): DICOM image object
            measured_slice_width (float, optional): Provide the true slice width for the set of images. Defaults to None.

        Returns:
            float: normalisation factor.
        """
        # Pixel spacing as read by ACRObject (also covers enhanced DICOM and
        # does not depend on the Manufacturer tag being present)
        dx, dy = self.pixel_spacing
        bandwidth = hazenlib.utils.get_bandwidth(dcm)
        TR = hazenlib.utils.get_TR(dcm)
        n_frequency, n_phase = self.encoding_steps(dcm)

        if measured_slice_width:
            slice_thickness = measured_slice_width
        else:
            slice_thickness = hazenlib.utils.get_slice_thickness(dcm)

        averages = hazenlib.utils.get_average(dcm)
        bandwidth_factor = np.sqrt(
            (bandwidth * n_frequency / 2) / 1000
        ) / np.sqrt(30)
        voxel_factor = 1 / (0.001 * dx * dy * slice_thickness)

        normalised_snr_factor = (
            bandwidth_factor
            * voxel_factor
            * (1 / (np.sqrt(averages * n_phase * (TR / 1000))))
        )
        return normalised_snr_factor

    # ------------------------------------------------------------------
    # Noise images and ROIs
    # ------------------------------------------------------------------
    def filtered_image(self, img) -> np.ndarray:
        """Apply filtering to a pixel array (image), as per the single image SNR method outlined in McCann et al, 2013.

        Notes:
            Performs a 2D moving average (uniform_filter, SciPy) in floating
            point: on an integer array uniform_filter returns integers, which
            truncated the smoothed values.

        Args:
            img (np.ndarray or pydicom.Dataset): image, in any orientation.

        Returns:
            np.ndarray: pixel array of the filtered image.
        """
        data = img.pixel_array if hasattr(img, "pixel_array") else img
        # McCann 2013 recommends 9x9 for the head coil and 25x25 for the
        # body coil; 25x25 is used here as in the original MATLAB code.
        # TODO add coil options, same as with MagNet SNR
        return ndimage.uniform_filter(
            np.asarray(data, dtype=float), self.FILTER_SIZE, mode="constant"
        )

    def get_noise_image(self, img) -> np.ndarray:
        """Get a noise image when only one set of DICOM data is available.

        Notes:
            Separates the image noise by smoothing the pixel array and subtracting the smoothed pixel array from the
            original, leaving only the noise.

        Args:
            img (np.ndarray or pydicom.Dataset): image, in any orientation.

        Returns:
            np.ndarray: pixel array representing the image noise.
        """
        data = img.pixel_array if hasattr(img, "pixel_array") else img
        return np.asarray(data, dtype=float) - self.filtered_image(data)

    def roi_centres(self):
        """Centres (x, y) of the five ROIs in mm from the phantom centre."""
        o = self.ROI_OFFSET
        return [(0.0, 0.0), (-o, -o), (-o, o), (o, -o), (o, o)]

    def roi_masks(self, shape, centre):
        """Boolean masks of the five square ROIs.

        Args:
            shape (tuple): image shape (rows, columns).
            centre (tuple): phantom centre (x, y) in pixels.

        Returns:
            list of np.ndarray: one mask per ROI.
        """
        row_sp, col_sp = self.spacing
        rows, cols = np.ogrid[: shape[0], : shape[1]]
        x_mm = (cols - centre[0]) * col_sp
        y_mm = (rows - centre[1]) * row_sp
        half = self.ROI_SIZE / 2
        masks = []
        for rx, ry in self.roi_centres():
            mask = (np.abs(x_mm - rx) < half) & (np.abs(y_mm - ry) < half)
            if not mask.any():
                msg = "SNR ROI lies outside the image"
                raise ValueError(msg)
            masks.append(mask)
        return masks

    def get_roi_samples(self, ax, data, centre_col, centre_row) -> list:
        """Pixel values in each of the five ROIs. If 'ax' is provided, the ROIs are also drawn on it.

        Args:
            ax (matplotlib axis or None): axis for visualisation.
            data (np.ndarray): image in the standard view.
            centre_col (float): x coordinate of the phantom centre (pixels).
            centre_row (float): y coordinate of the phantom centre (pixels).

        Returns:
            list of np.ndarray: pixel values within each ROI.
        """
        masks = self.roi_masks(data.shape, (centre_col, centre_row))
        if ax is not None:
            from matplotlib.patches import Rectangle

            row_sp, col_sp = self.spacing
            half = self.ROI_SIZE / 2
            for rx, ry in self.roi_centres():
                ax.add_patch(
                    Rectangle(
                        (
                            centre_col + (rx - half) / col_sp - 0.5,
                            centre_row + (ry - half) / row_sp - 0.5,
                        ),
                        self.ROI_SIZE / col_sp,
                        self.ROI_SIZE / row_sp,
                        edgecolor="red",
                        facecolor="none",
                    )
                )
        return [data[m] for m in masks]

    def phantom_centre(self, img):
        """Phantom centre (x, y) in pixels, from its edges."""
        centre, _, _ = measure_phantom(
            img, self.spacing, ("Horizontal", "Vertical")
        )
        return centre

    # ------------------------------------------------------------------
    # SNR
    # ------------------------------------------------------------------
    def snr_by_smoothing(
        self, dcm: pydicom.Dataset, measured_slice_width=None
    ) -> tuple:
        """Obtains a noise image using the single-image smoothing technique. Generates ROIs within the phantom region
        of the pixel array. Then measures the mean signal within each ROI on the original pixel array, and the standard
        deviation within the ROI on the noise image. Calculates SNR using these values and multiplies the SNR by the
        normalisation factor.

        Args:
            dcm (pydicom.Dataset): DICOM image object.
            measured_slice_width (float, optional): Provide the true slice width for the set of images. Defaults to None.

        Returns:
            tuple of float: snr, normalised_snr.
        """
        img = self.standard_view(dcm)
        centre = self.phantom_centre(img)
        noise_img = self.get_noise_image(img)

        signal = [
            np.mean(roi) for roi in self.get_roi_samples(None, img, *centre)
        ]
        noise = [
            np.std(roi, ddof=1)
            for roi in self.get_roi_samples(None, noise_img, *centre)
        ]
        # note no root_2 factor in noise for smoothed subtraction (one image) method, replicating Matlab approach and
        # McCann 2013
        snr = float(np.mean(np.divide(signal, noise)))

        normalised_snr = snr * self.get_normalised_snr_factor(
            dcm, measured_slice_width
        )

        if self.report:
            self.write_report(
                dcm,
                img,
                noise_img,
                centre,
                signal,
                noise,
                snr,
                "Smoothed noise image (image - 25 x 25 moving average)",
                "_smoothing",
            )

        return snr, normalised_snr

    def snr_by_subtraction(
        self,
        dcm1: pydicom.Dataset,
        dcm2: pydicom.Dataset,
        measured_slice_width=None,
        orientation2=None,
    ) -> tuple:
        """Calculates signal to noise ratio using the two image subtraction method. Obtains a noise image by subtracting
        the two pixel arrays. Obtains ROIs within the phantom region of the pixel arrays. Calculates the mean within
        the ROI on one of the pixel arrays, and the standard deviation within the ROIs on the noise image.
        Calculates the SNR with these measurements and multiplies by the normalisation factor.

        Args:
            dcm1 (pydicom.Dataset): DICOM image object to calculate signal.
            dcm2 (pydicom.Dataset): DICOM image object to calculate noise.
            measured_slice_width (float, optional): Provide the true slice width for the set of images. Defaults to None.
            orientation2 (tuple, optional): orientation correction (k, flip)
                for the second series; defaults to that of the first.

        Returns:
            tuple of float: snr, normalised_snr.
        """
        img1 = self.standard_view(dcm1)
        img2 = self.standard_view(dcm2, orientation2)
        if img1.shape != img2.shape:
            msg = (
                "The two series have different matrix sizes"
                f" ({img1.shape} and {img2.shape}) and cannot be subtracted"
            )
            raise ValueError(msg)
        centre = self.phantom_centre(img1)
        difference = img1 - img2

        signal = [
            np.mean(roi) for roi in self.get_roi_samples(None, img1, *centre)
        ]
        noise = np.divide(
            [
                np.std(roi, ddof=1)
                for roi in self.get_roi_samples(None, difference, *centre)
            ],
            np.sqrt(2),
        )
        snr = float(np.mean(np.divide(signal, noise)))

        normalised_snr = snr * self.get_normalised_snr_factor(
            dcm1, measured_slice_width
        )

        if self.report:
            self.write_report(
                dcm1,
                img1,
                difference,
                centre,
                signal,
                noise,
                snr,
                "Difference image (noise SD divided by sqrt 2)",
                "_snr_subtraction",
            )

        return snr, normalised_snr

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    def write_report(
        self, dcm, img, noise_img, centre, signal, noise, snr, label, suffix
    ):
        import matplotlib.pyplot as plt

        row_sp, col_sp = self.spacing
        fig, axes = plt.subplots(1, 2, figsize=(16, 8.5), layout="constrained")

        axes[0].imshow(img, cmap="gray", aspect=row_sp / col_sp)
        axes[0].scatter(*centre, c="red", s=12)
        self.get_roi_samples(axes[0], img, *centre)
        for (rx, ry), s, n in zip(self.roi_centres(), signal, noise):
            axes[0].text(
                centre[0] + rx / col_sp,
                centre[1] + (ry + self.ROI_SIZE / 2 + 5) / row_sp,
                f"SNR {s / n:.1f}",
                c="red",
                ha="center",
                fontsize=9,
                bbox=dict(facecolor="white", alpha=0.7, lw=0),
            )
        title = "Slice 7 with SNR ROIs"
        if self.orientation != (0, False):
            title += (
                f"\n(image {describe_orientation(*self.orientation)}"
                " to standard view)"
            )
        axes[0].set_title(title)
        axes[0].axis("off")

        inside = np.zeros(noise_img.shape, dtype=bool)
        for m in self.roi_masks(noise_img.shape, centre):
            inside |= m
        spread = 4 * np.std(noise_img[inside])
        axes[1].imshow(
            noise_img,
            cmap="gray",
            vmin=-spread,
            vmax=spread,
            aspect=row_sp / col_sp,
        )
        self.get_roi_samples(axes[1], noise_img, *centre)
        axes[1].set_title(label)
        axes[1].axis("off")
        fig.suptitle(f"SNR = {snr:.2f}")

        img_path = os.path.realpath(
            os.path.join(self.report_path, f"{self.img_desc(dcm)}{suffix}.png")
        )
        fig.savefig(img_path, dpi=80)
        plt.close(fig)
        self.report_files.append(img_path)
        return img_path
