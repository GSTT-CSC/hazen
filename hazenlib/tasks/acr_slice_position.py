"""
ACR Slice Position

https://www.acraccreditation.org/-/media/acraccreditation/documents/mri/largephantomguidance.pdf

Calculates the bar length difference for slices 1 and 11 of the ACR phantom.

The two crossed 45 degree wedges at the top of slices 1 and 11 appear as a
pair of adjacent dark bars. If the slice is displaced along the slice
direction, one bar becomes longer than the other; the bar length difference
is twice the slice position error. Following the ACR guidance, the
difference is reported as (right bar length - left bar length), so a
negative value means the left bar is longer.

Method
------
0. The phantom orientation is determined from slice 1 (as in the geometric
   accuracy task) and slices 1 and 11 are rotated by a multiple of 90 degrees
   (and mirrored if needed) into the standard view, with the wedges at the
   top. This keeps left/right, and therefore the sign of the result, correct
   even if a series was exported rotated or flipped, and the report images
   are shown in the standard view.
1. The phantom centre and radius are found with the Hough circle detector
   and refined by fitting a circle to the phantom edge.
2. The wedge box is segmented above the centre and the small residual tilt
   of the phantom is measured by fitting straight lines to the long side
   edges of the box, so that the bars run exactly along the sampling axis.
3. All further sampling happens in an "upright" frame (u to the right,
   v downwards, in mm, origin at the phantom centre) interpolated straight
   from the pixels, which also handles non-square pixels.
4. A line profile is averaged across the middle of each bar and the end of
   each bar is located with sub-sample precision as the point where the
   profile crosses halfway between the dark bar level and the bright
   background level beyond it.
5. The bar length difference is the difference between the two end
   positions.

Earlier versions cross-correlated the two profiles with a signed mean error,
which could lock onto the wrong feature (e.g. the bright edge at the top of
the wedge box) and report an incorrect shift.

Created by Yassine Azma
yassine.azma@rmh.nhs.uk

28/12/2022

Refactored by Luis M. Santos
luis.santos2@nih.gov

07/02/2025

Rewritten with edge-based bar end detection and 90 degree orientation
correction
08/10/2026
"""

import os
import sys
import traceback
from dataclasses import dataclass

import numpy as np
import scipy.ndimage
import skimage.measure
from hazenlib.ACRObject import ACRObject
from hazenlib.HazenTask import HazenTask
from hazenlib.logger import logger
from hazenlib.tasks.acr_geometric_accuracy import (
    describe_orientation,
    determine_phantom_orientation,
    reorient,
    reorient_spacing,
)
from hazenlib.types import Measurement, Result


@dataclass
class WedgeGeometry:
    """Location and orientation of the wedge pair.

    All lengths are in mm in the upright frame: origin at the phantom
    centre, ``u`` to the right and ``v`` downwards, with the wedges
    pointing towards negative ``v``.
    """

    centre: tuple  # phantom centre in pixels (col, row)
    radius: float  # phantom radius in mm
    theta: float  # direction of the wedges from the centre in radians,
    # measured in the reoriented image (x right, y down); close to -pi/2
    u_mid: float  # u of the line dividing the two bars
    width: float  # total width of the wedge box
    v_top: float  # top of the wedge box
    v_end_estimate: tuple  # coarse (left, right) bar ends from the mask


class ACRSlicePosition(HazenTask):
    """Slice position measurement class for DICOM images of the ACR phantom."""

    # Geometry of the ACR large phantom, in mm.
    WEDGE_DEPTH = 28  # the wedge box reaches at least this far below the edge
    SEARCH_HALF_WIDTH = 25  # half-width of the region searched for the box
    MIN_BOX_WIDTH = 5
    MAX_BOX_WIDTH = 30
    MAX_TILT_DEG = 15  # residual tilt allowed after the 90 degree correction

    SAMPLE_STEP = 0.1  # mm, profile sampling step
    GRID_STEP = 0.25  # mm, sampling step of the upright search image
    PROFILE_SMOOTH = 0.3  # mm, gaussian sigma applied to the profiles
    EDGE_MARGIN = 1.0  # mm kept away from the box sides and centre line
    END_SEARCH = 6.0  # mm either side of the coarse end estimate
    MAX_REFINE_ITER = 5

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

    @property
    def pixel_spacing(self):
        """(row, column) spacing in mm as stored in the DICOM, before any
        reorientation. ACRObject.dx/dy hold PixelSpacing[0]/[1]."""
        return (self.ACR_obj.dx, self.ACR_obj.dy)

    @property
    def orientation(self):
        """(k, flip) correction putting the phantom in its standard view.

        Determined once from slice 1, which has strongly asymmetric
        features, and applied to slices 1 and 11.
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
    def row_spacing(self):
        """Row spacing (mm) of the reoriented image."""
        return reorient_spacing(self.pixel_spacing, self.orientation[0])[0]

    @property
    def col_spacing(self):
        """Column spacing (mm) of the reoriented image."""
        return reorient_spacing(self.pixel_spacing, self.orientation[0])[1]

    def run(self) -> Result:
        """Main function for performing slice position measurement
        using the first and last slices from the ACR phantom image set.

        Returns:
            dict: results are returned in a standardised dictionary structure specifying the task name, input DICOM Series Description + SeriesNumber + InstanceNumber, task measurement key-value pairs, optionally path to the generated images for visualisation
        """
        # Identify relevant slices
        dcms = [self.ACR_obj.slice_stack[0], self.ACR_obj.slice_stack[-1]]

        # Initialise results dictionary
        results = self.init_result_dict(desc=self.ACR_obj.acquisition_type())
        results.files = [self.img_desc(dcm) for dcm in dcms]

        for slice_num, dcm in zip((1, 11), dcms, strict=True):
            try:
                result = self.get_slice_position(dcm)
                results.add_measurement(
                    Measurement(
                        name="SlicePosition",
                        type="measured",
                        subtype="length difference",
                        description=f"Slice {slice_num} {self.img_desc(dcm)}",
                        value=round(result, 2),
                    ),
                )

            except Exception as e:
                logger.exception(
                    "Could not calculate the bar length difference for %s"
                    " because of : %s",
                    self.img_desc(dcm),
                    e,
                )
                traceback.print_exc(file=sys.stdout)
                continue

        # only return reports if requested
        if self.report:
            results.add_report_image(self.report_files)

        return results

    # ------------------------------------------------------------------
    # Coordinate handling
    # ------------------------------------------------------------------
    def to_pixels(self, u, v, centre, theta):
        """Map upright-frame coordinates (mm) to pixel (row, col).

        ``theta`` is the direction of the wedges from the phantom centre in
        physical image coordinates (x right, y down). The upright frame's
        "up" (negative v) points along theta and its "right" (positive u)
        is up rotated by +90 degrees, so the mapping is a proper rotation
        and left/right are never mirrored.
        """
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        s, c = np.sin(theta), np.cos(theta)
        x_mm = -u * s - v * c
        y_mm = u * c - v * s
        col = centre[0] + x_mm / self.col_spacing
        row = centre[1] + y_mm / self.row_spacing
        return row, col

    def sample(self, img, u, v, centre, theta):
        """Bilinearly sample the original image at upright coordinates."""
        row, col = self.to_pixels(u, v, centre, theta)
        return scipy.ndimage.map_coordinates(
            img, [row, col], order=1, mode="constant", cval=0.0
        )

    # ------------------------------------------------------------------
    # Wedge localisation
    # ------------------------------------------------------------------
    @staticmethod
    def phantom_level(img, centre, radius_px):
        """Typical signal of the phantom fluid (median inside the disk)."""
        rows, cols = np.ogrid[: img.shape[0], : img.shape[1]]
        inside = (cols - centre[0]) ** 2 + (rows - centre[1]) ** 2 < (
            0.8 * radius_px
        ) ** 2
        return float(np.median(img[inside]))

    def refine_circle(self, img, centre, radius, level):
        """Refine the phantom centre and radius by fitting its edge.

        The Hough detector can be off by a few mm in noisy images. Here the
        edge is located on rays every 2 degrees (outermost half-level
        crossing) and a circle is fitted, rejecting outliers such as the
        notch above the wedges.

        Args:
            centre (tuple): approximate centre in pixels (col, row)
            radius (float): approximate radius in mm

        Returns:
            tuple: (centre in pixels (col, row), radius in mm)
        """
        angles = np.deg2rad(np.arange(0, 360, 2.0))
        r = np.arange(0.7 * radius, 1.2 * radius, self.SAMPLE_STEP)
        x = np.cos(angles)[:, None] * r[None, :]
        y = np.sin(angles)[:, None] * r[None, :]
        col = centre[0] + x / self.col_spacing
        row = centre[1] + y / self.row_spacing
        prof = scipy.ndimage.map_coordinates(
            img, [row.ravel(), col.ravel()], order=1, mode="constant"
        ).reshape(row.shape)
        prof = scipy.ndimage.gaussian_filter1d(
            prof, 0.5 / self.SAMPLE_STEP, axis=1
        )
        pts = []
        for k in range(len(angles)):
            above = np.nonzero(prof[k] > 0.5 * level)[0]
            if len(above) and above[-1] < len(r) - 1:
                i = above[-1]
                p0, p1 = prof[k, i], prof[k, i + 1]
                re = r[i] + self.SAMPLE_STEP * (p0 - 0.5 * level) / (p0 - p1)
                pts.append((re * np.cos(angles[k]), re * np.sin(angles[k])))
        pts = np.array(pts)
        if len(pts) < 20:
            return centre, radius

        keep = np.ones(len(pts), dtype=bool)
        for _ in range(3):
            # algebraic circle fit: x^2 + y^2 + D x + E y + F = 0
            px, py = pts[keep, 0], pts[keep, 1]
            A = np.column_stack([px, py, np.ones_like(px)])
            D, E, F = np.linalg.lstsq(A, -(px**2 + py**2), rcond=None)[0]
            xc, yc = -D / 2, -E / 2
            rad = np.sqrt(xc**2 + yc**2 - F)
            resid = np.hypot(pts[:, 0] - xc, pts[:, 1] - yc) - rad
            keep = np.abs(resid) < 1.5
            if keep.sum() < 20:
                return centre, radius

        new_centre = (
            centre[0] + xc / self.col_spacing,
            centre[1] + yc / self.row_spacing,
        )
        logger.info(
            "Phantom edge fit: centre (%.1f, %.1f) px, radius %.1f mm",
            *new_centre,
            rad,
        )
        return new_centre, float(rad)

    def coarse_wedge_direction(self, img, centre, radius):
        """Coarse direction (radians) of the wedges, near straight up.

        In the standard view the wedges point up from the centre. Rays are
        cast within +/- MAX_TILT_DEG of straight up through the band just
        inside the phantom edge that the wedge box crosses; the darkest ray
        gives the residual tilt to within about a degree.
        """
        angles = -np.pi / 2 + np.deg2rad(
            np.arange(-self.MAX_TILT_DEG, self.MAX_TILT_DEG + 0.25, 0.25)
        )
        r = np.linspace(radius - self.WEDGE_DEPTH, radius - 12, 33)
        x = np.cos(angles)[:, None] * r[None, :]
        y = np.sin(angles)[:, None] * r[None, :]
        col = centre[0] + x / self.col_spacing
        row = centre[1] + y / self.row_spacing
        vals = (
            scipy.ndimage.map_coordinates(
                img, [row.ravel(), col.ravel()], order=1, mode="nearest"
            )
            .reshape(row.shape)
            .mean(axis=1)
        )
        vals = scipy.ndimage.uniform_filter1d(vals, 9, mode="nearest")
        # middle of the run of rays darker than halfway to the minimum
        dark = np.nonzero(vals < 0.5 * (vals.min() + np.median(vals)))[0]
        i_min = int(np.argmin(vals))
        runs = np.split(dark, np.nonzero(np.diff(dark) > 1)[0] + 1)
        run = next(r_ for r_ in runs if i_min in r_)
        return float(angles[run].mean())

    def segment_wedge_box(self, img, centre, theta, radius, level):
        """Segment the dark wedge box in the upright frame.

        Returns:
            tuple: (u grid, v grid, boolean mask of the box)
        """
        step = self.GRID_STEP
        u = np.arange(
            -self.SEARCH_HALF_WIDTH, self.SEARCH_HALF_WIDTH + step, step
        )
        v = np.arange(-radius, -radius / 4, step)
        uu, vv = np.meshgrid(u, v)
        grid = self.sample(img, uu, vv, centre, theta)
        # Light smoothing so noise does not fragment the coarse mask; the
        # bar ends themselves are measured on separate line profiles.
        grid = scipy.ndimage.gaussian_filter(grid, 0.5 / step)

        inside = uu**2 + vv**2 < (radius - 1.5) ** 2
        dark = (grid < 0.5 * level) & inside
        dark = scipy.ndimage.binary_opening(dark, iterations=2)
        dark = scipy.ndimage.binary_fill_holes(dark)
        labels = skimage.measure.label(dark)
        if labels.max() == 0:
            raise ValueError("Wedge box not found in the expected location")

        # Choose the largest dark component that reaches the outer ring
        best, best_area = 0, 0
        outer_v = -(radius - self.WEDGE_DEPTH)
        for region in skimage.measure.regionprops(labels):
            rows, cols = region.coords[:, 0], region.coords[:, 1]
            if v[rows].min() > outer_v:
                continue
            if abs(np.median(u[cols])) > self.SEARCH_HALF_WIDTH / 2:
                continue
            if region.area > best_area:
                best, best_area = region.label, region.area
        if not best:
            raise ValueError("Wedge box not found in the expected location")
        return u, v, labels == best

    @staticmethod
    def describe_box(u, v, box):
        """Coarse description of the segmented wedge box.

        The line dividing the two bars is fitted through the centres of the
        rows near the top of the box, where both bars are always present.
        Each bar's end is then the last row where the mask still covers the
        middle of that bar.

        Returns:
            tuple: (v_top, divider line coefficients u = f(v),
            box width, (left end, right end))
        """
        step = v[1] - v[0]
        widths = box.sum(axis=1) * step

        # A dark feature touching the box (e.g. the notch in the phantom
        # edge when the phantom is tilted) adds pixels beside the box. Trim
        # every row to the median extent of the rows of typical box width.
        typical = np.percentile(widths[widths > 0], 75)
        core = np.nonzero(np.abs(widths - typical) < 0.25 * typical)[0]
        if len(core):
            left = np.median([np.argmax(box[i]) for i in core])
            right = np.median(
                [box.shape[1] - 1 - np.argmax(box[i][::-1]) for i in core]
            )
            margin = ACRSlicePosition.EDGE_MARGIN / step
            cols = np.arange(box.shape[1])
            box = box & (cols >= left - margin) & (cols <= right + margin)
            widths = box.sum(axis=1) * step

        rows_present = np.nonzero(box.any(axis=1))[0]
        top_row = rows_present[0]
        v_top = v[top_row]
        top_sel = (v >= v_top + 2) & (v <= v_top + 12)
        box_width = float(np.median(widths[top_sel]))
        full = (v >= v_top + 2) & (widths > 0.85 * box_width)
        if full.sum() < 5:
            raise ValueError("Could not find the top of the wedge box")
        centres = np.array(
            [
                0.5
                * (
                    u[np.nonzero(box[i])[0]].min()
                    + u[np.nonzero(box[i])[0]].max()
                )
                for i in np.nonzero(full)[0]
            ]
        )
        # The coarse direction is within a couple of degrees, so a constant
        # divider is accurate enough to separate the bars; the fine angle
        # comes from the box edge fit.
        divider = np.array([0.0, float(np.median(centres))])

        ends = []
        for side in (-1, 1):
            u_bar = np.polyval(divider, v) + side * box_width / 4
            cols = np.clip(
                np.round((u_bar - u[0]) / step).astype(int), 0, len(u) - 1
            )
            covered = box[np.arange(len(v)), cols]
            # last row of the run that starts near the top of the box
            start = int(np.nonzero(full)[0][0])
            run = np.nonzero(~covered[start:])[0]
            last = start + (run[0] - 1 if len(run) else len(v) - 1 - start)
            ends.append(float(v[last]))
        return float(v_top), divider, box_width, tuple(ends)

    def fit_box_edges(self, img, centre, theta, level, u_mids, v_rows):
        """Find the left and right edges of the wedge box on each row.

        Args:
            u_mids (np.ndarray): approximate centre of the box on each row.

        Returns:
            tuple: arrays (v, u_left, u_right) of rows where both edges
            were found.
        """
        du = self.SAMPLE_STEP
        offs = np.arange(-15, 15 + du, du)
        vs, lefts, rights = [], [], []
        for v_row, u_mid in zip(v_rows, u_mids):
            p = self.sample(
                img, u_mid + offs, np.full_like(offs, v_row), centre, theta
            )
            p = scipy.ndimage.gaussian_filter1d(p, self.PROFILE_SMOOTH / du)
            centre_idx = len(offs) // 2
            dark = np.median(p[np.abs(offs) < 2])
            half = 0.5 * (dark + level)
            # walk out from the centre until the signal rises above half
            left = right = None
            for i in range(centre_idx, 0, -1):
                if p[i] < half <= p[i - 1]:
                    left = offs[i] - du * (half - p[i]) / (p[i - 1] - p[i])
                    break
            for i in range(centre_idx, len(p) - 1):
                if p[i] < half <= p[i + 1]:
                    right = offs[i] + du * (half - p[i]) / (p[i + 1] - p[i])
                    break
            if left is not None and right is not None:
                vs.append(v_row)
                lefts.append(u_mid + left)
                rights.append(u_mid + right)
        return np.array(vs), np.array(lefts), np.array(rights)

    @staticmethod
    def robust_line(x, y):
        """Least squares line with one pass of outlier rejection."""
        coef = np.polyfit(x, y, 1)
        resid = y - np.polyval(coef, x)
        keep = np.abs(resid) < max(3 * np.median(np.abs(resid)) * 1.4826, 0.25)
        if keep.sum() >= 3:
            coef = np.polyfit(x[keep], y[keep], 1)
        return coef

    def find_wedges(self, img, centre, radius_px):
        """Locate the wedge box and the residual tilt of the phantom.

        Args:
            img (np.ndarray): pixel array in the standard view (wedges at
                the top, see ``reorient``)
            centre (tuple[int]): phantom centre in pixels (col, row)
            radius_px (float): phantom radius in pixels

        Returns:
            WedgeGeometry: geometry of the wedge pair in the upright frame.
        """
        radius = float(radius_px) * float(
            np.sqrt(self.row_spacing * self.col_spacing)
        )
        level = self.phantom_level(img, centre, radius_px)
        centre, radius = self.refine_circle(img, centre, radius, level)
        # In the standard view the wedges point (nearly) straight up.
        theta = self.coarse_wedge_direction(img, centre, radius)

        for _ in range(self.MAX_REFINE_ITER):
            u, v, box = self.segment_wedge_box(
                img, centre, theta, radius, level
            )
            v_top, divider, box_width, v_end_est = self.describe_box(u, v, box)

            # Rows where both bars are present, clear of either end.
            v_lo, v_hi = v_top + 3.0, min(v_end_est) - 3.0
            if v_hi - v_lo < 4.0:
                raise ValueError("Wedge box too short to measure")
            v_rows = np.arange(v_lo, v_hi, 0.5)
            vs, lefts, rights = self.fit_box_edges(
                img,
                centre,
                theta,
                level,
                np.polyval(divider, v_rows),
                v_rows,
            )
            if len(vs) < 5:
                raise ValueError("Could not find the sides of the wedge box")
            coef_l = self.robust_line(vs, lefts)
            coef_r = self.robust_line(vs, rights)
            slope = 0.5 * (coef_l[0] + coef_r[0])
            v_c = 0.5 * (v_lo + v_hi)
            u_mid = 0.5 * (np.polyval(coef_l, v_c) + np.polyval(coef_r, v_c))
            width = float(np.median(rights - lefts))

            # The box axis moves by ``slope`` mm in u for every mm in v.
            # Once it is aligned, the values above belong to this theta.
            if abs(slope) < 1e-3:
                break
            # Rotate the frame onto the box axis (limited per step).
            theta -= np.clip(np.arctan(slope), -np.deg2rad(5), np.deg2rad(5))
        else:
            theta += np.clip(np.arctan(slope), -np.deg2rad(5), np.deg2rad(5))
            logger.warning(
                "Wedge orientation did not fully converge (residual slope "
                "%.4f)",
                slope,
            )

        if not self.MIN_BOX_WIDTH < width < self.MAX_BOX_WIDTH:
            raise ValueError(f"Implausible wedge box width {width:.1f} mm")
        tilt = np.rad2deg(theta + np.pi / 2)
        if abs(tilt) > self.MAX_TILT_DEG:
            raise ValueError(
                f"Phantom tilted by {tilt:.1f} deg after the 90 degree "
                "orientation correction; only small tilts are supported"
            )

        logger.info(
            "Wedge box: tilt %.2f deg, centre line u=%.2f mm, "
            "width %.2f mm, top v=%.1f mm",
            tilt,
            u_mid,
            width,
            v_top,
        )
        return WedgeGeometry(
            centre=tuple(centre),
            radius=radius,
            theta=float(theta),
            u_mid=float(u_mid),
            width=width,
            v_top=float(v_top),
            v_end_estimate=tuple(float(x) for x in v_end_est),
        )

    # ------------------------------------------------------------------
    # Bar end measurement
    # ------------------------------------------------------------------
    def bar_profile(self, img, geo, side, v):
        """Line profile along a bar, averaged across the middle of the bar.

        Args:
            side (int): -1 for the left bar, +1 for the right bar.
        """
        quarter = geo.width / 4
        u_c = geo.u_mid + side * quarter
        half_band = max(quarter - self.EDGE_MARGIN, 0.5)
        offsets = np.arange(-half_band, half_band + 1e-9, 0.25)
        prof = np.zeros_like(v)
        for du in offsets:
            prof += self.sample(
                img, np.full_like(v, u_c + du), v, geo.centre, geo.theta
            )
        prof /= len(offsets)
        return (
            scipy.ndimage.gaussian_filter1d(
                prof, self.PROFILE_SMOOTH / self.SAMPLE_STEP
            ),
            u_c,
            half_band,
        )

    def find_bar_end(self, v, prof, v_top, v_est):
        """Position (mm) where the bar ends, at half the dark/bright step.

        Returns:
            tuple: (end position, half level, dark level, bright level)
        """
        bright = np.median(prof[(v > v_est + 5) & (v < v_est + 12)])
        dark_sel = (v > v_top + 3) & (v < v_est - 3)
        if dark_sel.sum() < 10:
            dark_sel = (v > v_top + 1) & (v < v_est - 1)
        dark = np.median(prof[dark_sel])
        if bright - dark <= 0:
            raise ValueError("No contrast at the end of the bar")
        half = 0.5 * (bright + dark)

        window = np.nonzero(
            (v >= v_est - self.END_SEARCH) & (v <= v_est + self.END_SEARCH)
        )[0]
        # Last sample below half level, scanning back from beyond the end
        for i in window[::-1][1:]:
            if prof[i] < half <= prof[i + 1]:
                end = v[i] + (v[i + 1] - v[i]) * (half - prof[i]) / (
                    prof[i + 1] - prof[i]
                )
                return float(end), half, dark, bright
        raise ValueError("Could not locate the end of the bar")

    def get_slice_position(self, dcm):
        """Measure slice position. \n
        Locates the two opposing wedges and calculates the length difference.

        Args:
            dcm (pydicom.Dataset): DICOM image object.

        Returns:
            float: bar length difference in mm (right minus left).
        """
        logger.info(
            f"Computing slice position for slice #{dcm.InstanceNumber}"
        )
        _, rescaled, _ = self.ACR_obj.get_presentation_pixels(dcm)
        # Put the phantom in the standard view (wedges at the top, left and
        # right as seen in the ACR guidance).
        img = np.ascontiguousarray(
            reorient(np.asarray(rescaled, dtype=float), *self.orientation)
        )
        centre, radius_px = self.ACR_obj.find_phantom_center(
            img, self.row_spacing, self.col_spacing
        )
        geo = self.find_wedges(img, centre, radius_px)

        v = np.arange(
            geo.v_top + 1.0,
            min(max(geo.v_end_estimate) + 15.0, -geo.radius / 4),
            self.SAMPLE_STEP,
        )
        ends, details = {}, {}
        for name, side, v_est in (
            ("L", -1, geo.v_end_estimate[0]),
            ("R", 1, geo.v_end_estimate[1]),
        ):
            prof, u_c, hb = self.bar_profile(img, geo, side, v)
            end, half, dark, bright = self.find_bar_end(
                v, prof, geo.v_top, v_est
            )
            ends[name] = end
            details[name] = dict(
                prof=prof, u=u_c, half_band=hb, half=half, end=end
            )

        # Bars extend towards -v, so a longer bar ends at a larger v.
        dL = ends["R"] - ends["L"]
        logger.info(
            "Bar ends (mm from centre): left %.2f, right %.2f -> dL %.2f",
            ends["L"],
            ends["R"],
            dL,
        )

        if self.report:
            self.write_report(dcm, img, geo, v, details, dL)

        return dL

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    def write_report(self, dcm, img, geo, v, details, dL):
        import matplotlib.pyplot as plt

        colours = {"L": "tab:blue", "R": "tab:red"}
        fig, axes = plt.subplots(1, 3, figsize=(18, 7), layout="constrained")

        # 1. Image in the standard view with the profile lines.
        # Pixels are shown with their physical aspect ratio.
        ax = axes[0]
        ax.imshow(img, cmap="gray", aspect=self.row_spacing / self.col_spacing)
        ax.scatter(*geo.centre, c="yellow", s=15)
        for name, d in details.items():
            rows, cols = self.to_pixels(
                np.full(2, d["u"]), v[[0, -1]], geo.centre, geo.theta
            )
            ax.plot(cols, rows, c=colours[name], lw=1.5)
        title = "Line profiles"
        if self.orientation != (0, False):
            title += (
                f"\n(image {describe_orientation(*self.orientation)} "
                "to standard view)"
            )
        ax.set_title(title)
        ax.axis("off")

        # 2. Wedge region, sampled along the bars (corrects small tilts)
        ax = axes[1]
        step = 0.2
        u = np.arange(geo.u_mid - 15, geo.u_mid + 15, step)
        vv_ = np.arange(-geo.radius - 3, v[-1] + 3, step)
        uu, vv = np.meshgrid(u, vv_)
        crop = self.sample(img, uu, vv, geo.centre, geo.theta)
        ax.imshow(crop, cmap="gray", extent=[u[0], u[-1], vv_[-1], vv_[0]])
        for name, d in details.items():
            for edge in (-d["half_band"], d["half_band"]):
                ax.axvline(d["u"] + edge, c=colours[name], lw=0.8, ls=":")
            ax.plot(
                [d["u"] - geo.width / 4, d["u"] + geo.width / 4],
                [d["end"]] * 2,
                c=colours[name],
                lw=2,
            )
        ax.set_xlabel("u (mm)")
        ax.set_ylabel("v (mm from phantom centre)")
        ax.set_title(
            f"Wedges (tilt {np.rad2deg(geo.theta + np.pi / 2):+.1f}°)"
            " with detected bar ends"
        )

        # 3. Profiles
        ax = axes[2]
        for name, d in details.items():
            label = "Left" if name == "L" else "Right"
            ax.plot(v, d["prof"], c=colours[name], label=f"{label} bar")
            ax.axhline(d["half"], c=colours[name], lw=0.8, ls="--")
            ax.axvline(d["end"], c=colours[name], lw=1)
        ax.set_xlabel("v (mm from phantom centre)")
        ax.set_ylabel("Signal")
        ax.grid(alpha=0.4)
        ax.legend()
        ax.set_title(
            f"Bar ends: L {details['L']['end']:.2f}, "
            f"R {details['R']['end']:.2f} mm  →  "
            f"length difference (R−L) = {dL:.2f} mm"
        )

        img_path = os.path.realpath(
            os.path.join(self.report_path, f"{self.img_desc(dcm)}.png")
        )
        fig.savefig(img_path, dpi=80)
        plt.close(fig)
        self.report_files.append(img_path)

        return img_path
