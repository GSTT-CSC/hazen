"""
ACR Ghosting

https://www.acraccreditation.org/-/media/acraccreditation/documents/mri/largephantomguidance.pdf

Calculates the percent-signal ghosting for slice 7 of the ACR phantom.

This script calculates the percentage signal ghosting in accordance with the
ACR Guidance: a large circular ROI of 200 cm2 is placed in the phantom and
four elliptical ROIs of 10 cm2 (axis ratio 4:1) are placed in the background
above, below, left and right of the phantom.

    PSG = 100 * | ((top + bottom) - (left + right)) / (2 * large) |

Method
------
0. The phantom orientation is determined from slice 1 (as in the geometric
   accuracy task) and slice 7 is rotated by a multiple of 90 degrees (and
   mirrored if needed) into the standard view. PSG itself is unchanged by
   this (the formula is symmetric), but "top", "bottom", "left" and "right"
   then refer to the phantom, and the report is shown in the standard view.
1. Rescaled pixel values are used, so a DICOM rescale intercept cannot
   distort the ratio.
2. The phantom is segmented (Otsu threshold, largest component, convex hull)
   and its edges are taken along the horizontal and vertical lines through
   its centre.
3. The reconstructed field of view is the bounding box of the rows and
   columns that are not entirely zero, so zero-filled borders are never
   sampled as background.
4. All ROI geometry is in mm, so it is correct for any (including
   non-square) pixel size:
   - large ROI: circle of 200 cm2, 5 mm below the phantom centre;
   - ghost ROIs: ellipses of 10 cm2 with a 4:1 axis ratio, long axis
     parallel to the nearby phantom edge, centred midway between the
     phantom edge and the edge of the field of view and kept at least 3 mm
     from both. If the gap is too narrow the ellipse is flattened at
     constant area, down to half its nominal width. If there is not enough
     room on one side (phantom close to the edge of the field of view), the
     opposite ROI of that pair is used for both and a warning is given,
     rather than returning NaN.

Created by Yassine Azma
yassine.azma@rmh.nhs.uk

14/11/2022

90 degree orientation correction, ROI geometry in mm and robust ROI
placement 08/10/2026
"""

import os
import sys
import traceback
from dataclasses import dataclass

import numpy as np
from hazenlib.ACRObject import ACRObject
from hazenlib.HazenTask import HazenTask
from hazenlib.logger import logger
from hazenlib.tasks.acr_geometric_accuracy import (
    describe_orientation,
    determine_phantom_orientation,
    reorient,
    reorient_spacing,
    segment_phantom,
)
from hazenlib.types import Measurement, Result


@dataclass
class EllipseROI:
    """An elliptical ROI, in mm from the phantom centre (x right, y down)."""

    name: str
    x: float  # centre
    y: float
    a: float  # semi-axis along x
    b: float  # semi-axis along y
    mean: float = np.nan

    @property
    def area(self) -> float:
        """Area in mm2."""
        return float(np.pi * self.a * self.b)


class ACRGhosting(HazenTask):
    """Ghosting measurement class for DICOM images of the ACR phantom."""

    LARGE_ROI_AREA = 20000.0  # mm2 (200 cm2)
    LARGE_ROI_OFFSET = 5.0  # mm below the phantom centre
    GHOST_ROI_AREA = 1000.0  # mm2 (10 cm2)
    GHOST_ROI_RATIO = 4.0  # long / short axis
    GHOST_ROI_MARGIN = 3.0  # mm kept from the phantom and from the FOV edge
    # A ghost ROI may be flattened (at constant area) down to this fraction
    # of its nominal short axis, i.e. to at most twice its nominal length;
    # if the gap is narrower still, that side is not used.
    MIN_SHORT_AXIS_FRACTION = 0.5
    SLICE_INDEX = 6  # slice 7

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
        """Main function for performing ghosting measurement using slice 7 from the ACR phantom image set.

        Returns:
            dict: results are returned in a standardised dictionary structure specifying the task name, input DICOM Series Description + SeriesNumber + InstanceNumber, task measurement key-value pairs, optionally path to the generated images for visualisation.
        """
        dcm = self.ACR_obj.slice_stack[self.SLICE_INDEX]
        # Initialise results dictionary
        results = self.init_result_dict(desc=self.ACR_obj.acquisition_type())
        results.files = self.img_desc(dcm)

        try:
            psg = self.get_signal_ghosting(dcm)
            results.add_measurement(
                Measurement(
                    name="Ghosting",
                    type="measured",
                    subtype="signal ghosting",
                    value=round(psg, 3),
                    unit="%",
                ),
            )
            results.add_measurement(
                Measurement(
                    name="Ghosting",
                    type="measured",
                    subtype="ROI mean",
                    description="large ROI",
                    value=round(self.large_roi_mean, 2),
                    visibility="intermediate",
                ),
            )
            for roi in self.ghost_rois.values():
                results.add_measurement(
                    Measurement(
                        name="Ghosting",
                        type="measured",
                        subtype="ROI mean",
                        description=f"{roi.name} ghost ROI",
                        value=round(roi.mean, 2),
                        visibility="intermediate",
                    ),
                )

        except Exception as e:
            logger.exception(
                "Could not calculate the percent-signal ghosting for %s"
                " because of : %s",
                self.img_desc(dcm),
                e,
            )
            traceback.print_exc(file=sys.stdout)

        # only return reports if requested
        if self.report:
            results.add_report_image(self.report_files)

        return results

    def get_signal_ghosting(self, dcm):
        """Calculate the percentage signal ghosting (PSG). \n
        Sample signal intensity from ellipses outside the phantom in
        four directions and calculate the mean signal value within each.
        Percentage signal ghosting (PSG) is then expressed as the mean signal
        in these four ROIs as a percentage of the mean signal in a ROI in the
        centre of the phantom.

        Args:
            dcm (pydicom.Dataset): DICOM image object.

        Returns:
            float: percentage ghosting value.
        """
        _, rescaled, _ = self.ACR_obj.get_presentation_pixels(dcm)
        img = np.ascontiguousarray(
            reorient(np.asarray(rescaled, dtype=float), *self.orientation)
        )
        psg = self.measure_ghosting(img)
        if self.report:
            self.write_report(dcm, img, psg)
        return psg

    def measure_ghosting(self, img):
        """Place the ROIs and calculate PSG on an image in the standard view.

        Sets ``large_roi_mean``, ``ghost_rois`` and ``ghost_notes``.

        Args:
            img (np.ndarray): slice 7 pixel values, in the standard view.

        Returns:
            float: percentage ghosting value.
        """
        row_sp, col_sp = self.spacing

        # Phantom: centre and edges along the lines through the centre (mm)
        _, mask = segment_phantom(img)
        cy, cx = (float(c) for c in np.argwhere(mask).mean(axis=0))
        rows = np.nonzero(mask[:, int(round(cx))])[0]
        cols = np.nonzero(mask[int(round(cy)), :])[0]
        phantom = {
            "top": (rows[0] - 0.5 - cy) * row_sp,
            "bottom": (rows[-1] + 0.5 - cy) * row_sp,
            "left": (cols[0] - 0.5 - cx) * col_sp,
            "right": (cols[-1] + 0.5 - cx) * col_sp,
        }

        # Reconstructed field of view: rows/columns that are not all zero
        nz_rows = np.nonzero((img != 0).any(axis=1))[0]
        nz_cols = np.nonzero((img != 0).any(axis=0))[0]
        fov = {
            "top": (nz_rows[0] - 0.5 - cy) * row_sp,
            "bottom": (nz_rows[-1] + 0.5 - cy) * row_sp,
            "left": (nz_cols[0] - 0.5 - cx) * col_sp,
            "right": (nz_cols[-1] + 0.5 - cx) * col_sp,
        }

        # Pixel coordinates in mm relative to the phantom centre
        yy, xx = np.indices(img.shape, dtype=float)
        x_mm = (xx - cx) * col_sp
        y_mm = (yy - cy) * row_sp

        # Large ROI
        self.centre = (cx, cy)
        r_large = np.sqrt(self.LARGE_ROI_AREA / np.pi)
        large = np.hypot(x_mm, y_mm - self.LARGE_ROI_OFFSET) <= r_large
        if mask[large].mean() < 0.99:
            logger.warning("The large ROI extends beyond the phantom")
        self.large_roi_mean = float(img[large].mean())

        # Ghost ROIs
        self.ghost_rois = {}
        self.ghost_notes = []
        for side in ("top", "bottom", "left", "right"):
            roi = self.place_ghost_roi(side, phantom, fov)
            if roi is None:
                continue
            inside = (
                ((x_mm - roi.x) / roi.a) ** 2 + ((y_mm - roi.y) / roi.b) ** 2
            ) <= 1
            if not inside.any():
                continue
            roi.mean = float(img[inside].mean())
            self.ghost_rois[side] = roi

        # Missing ROIs: use the opposite ROI of the same pair
        means = {}
        for pair in (("top", "bottom"), ("left", "right")):
            available = [s for s in pair if s in self.ghost_rois]
            if not available:
                msg = (
                    "No room for a ghost ROI on either side"
                    f" ({' or '.join(pair)})"
                    " of the phantom"
                )
                raise ValueError(msg)
            for side in pair:
                if side in self.ghost_rois:
                    means[side] = self.ghost_rois[side].mean
                else:
                    other = available[0]
                    means[side] = self.ghost_rois[other].mean
                    note = (
                        f"No room for the {side} ghost ROI; the {other} ROI"
                        " was used for both"
                    )
                    logger.warning(note)
                    self.ghost_notes.append(note)

        return 100 * abs(
            (
                (means["top"] + means["bottom"])
                - (means["left"] + means["right"])
            )
            / (2 * self.large_roi_mean)
        )

    def place_ghost_roi(self, side, phantom, fov):
        """Place a 10 cm2 ghost ROI in the background beside the phantom.

        The ellipse has a 4:1 axis ratio with its long axis parallel to the
        phantom edge, and is centred midway between the phantom edge and the
        edge of the field of view, at least GHOST_ROI_MARGIN mm from each. If
        the gap is too narrow, the short axis is
        reduced and the long axis lengthened to keep the area.

        Args:
            side (str): "top", "bottom", "left" or "right".
            phantom (dict): phantom edges (mm from its centre).
            fov (dict): field of view edges (mm from the phantom centre).

        Returns:
            EllipseROI or None if there is no room on that side.
        """
        vertical = side in ("top", "bottom")
        gap = abs(fov[side] - phantom[side]) - 2 * self.GHOST_ROI_MARGIN
        b_nominal = np.sqrt(
            self.GHOST_ROI_AREA / (np.pi * self.GHOST_ROI_RATIO)
        )
        b = min(b_nominal, gap / 2)
        if b < self.MIN_SHORT_AXIS_FRACTION * b_nominal:
            return None
        a = self.GHOST_ROI_AREA / (np.pi * b)
        # the long axis must stay within the field of view
        if vertical:
            a_max = min(-fov["left"], fov["right"]) - self.GHOST_ROI_MARGIN
        else:
            a_max = min(-fov["top"], fov["bottom"]) - self.GHOST_ROI_MARGIN
        a = min(a, a_max)
        centre = 0.5 * (fov[side] + phantom[side])
        if (
            abs(b - b_nominal) > 1e-6
            or a * b < 0.99 * b_nominal**2 * self.GHOST_ROI_RATIO
        ):
            logger.info(
                "The %s ghost ROI was resized to fit (%.0f mm2)",
                side,
                np.pi * a * b,
            )
        if vertical:
            return EllipseROI(name=side, x=0.0, y=centre, a=a, b=b)
        return EllipseROI(name=side, x=centre, y=0.0, a=b, b=a)

    def write_report(self, dcm, img, psg):
        import matplotlib.pyplot as plt

        row_sp, col_sp = self.spacing
        cx, cy = self.centre
        r_large = np.sqrt(self.LARGE_ROI_AREA / np.pi)
        theta = np.linspace(0, 2 * np.pi, 360)

        def to_px(x_mm, y_mm):
            return cx + np.asarray(x_mm) / col_sp, cy + np.asarray(
                y_mm
            ) / row_sp

        fig, axes = plt.subplots(1, 2, figsize=(16, 8.5), layout="constrained")
        background = [roi.mean for roi in self.ghost_rois.values()]
        windows = (
            (None, None, "Full window"),
            (
                0,
                max(4 * max(background), 0.02 * self.large_roi_mean),
                "Windowed to the background (ghosts)",
            ),
        )
        for ax, (vmin, vmax, label) in zip(axes, windows):
            ax.imshow(
                img, cmap="gray", vmin=vmin, vmax=vmax, aspect=row_sp / col_sp
            )
            ax.plot(
                *to_px(
                    r_large * np.cos(theta),
                    r_large * np.sin(theta) + self.LARGE_ROI_OFFSET,
                ),
                c="tab:blue",
            )
            ax.text(
                *to_px(0, self.LARGE_ROI_OFFSET),
                f"Mean = {self.large_roi_mean:.1f}",
                c="tab:blue",
                ha="center",
                va="center",
                bbox=dict(facecolor="white", alpha=0.7, lw=0),
            )
            for roi in self.ghost_rois.values():
                ax.plot(
                    *to_px(
                        roi.x + roi.a * np.cos(theta),
                        roi.y + roi.b * np.sin(theta),
                    ),
                    c="tab:red",
                )
                ax.text(
                    *to_px(roi.x, roi.y),
                    f"{roi.mean:.1f}",
                    c="tab:red",
                    ha="center",
                    va="center",
                    fontsize=9,
                    bbox=dict(facecolor="white", alpha=0.7, lw=0),
                )
            ax.set_title(label)
            ax.axis("off")

        title = f"Percent Signal Ghosting = {psg:.3f}%"
        if self.orientation != (0, False):
            title += (
                f"\n(image {describe_orientation(*self.orientation)}"
                " to standard view)"
            )
        for note in self.ghost_notes:
            title += f"\n{note}"
        fig.suptitle(title)

        img_path = os.path.realpath(
            os.path.join(self.report_path, f"{self.img_desc(dcm)}.png")
        )
        fig.savefig(img_path, dpi=80)
        plt.close(fig)
        self.report_files.append(img_path)
        return img_path
