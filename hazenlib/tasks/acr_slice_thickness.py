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

    #. The phantom orientation is determined from slice 1 (as in the
        geometric accuracy task) and the slice is rotated by a multiple
        of 90 degrees (and mirrored if needed) into the standard view.
        This replaces the previous special case for sagittal images and
        makes "top" and "bottom" ramp refer to the phantom itself. The
        report shows the image in the standard view.
    #. Pre-process the image with CLAHE contrast enhancement and Otsu
        binarization, detect contours and take the most elongated
        rectangular contour of plausible size as the slice thickness
        insert. Its oriented bounding box is computed in mm, so the
        geometry is correct for non-square pixels.
    #. The tilt and position of the insert are refined by fitting
        straight lines to its two long walls, so that the profiles run
        exactly along the ramps.
    #. The insert is resampled in its own frame (u along the insert,
        v across it, in mm). Its dark interior is found at the ends of
        the insert, where the ramps do not reach. The top ramp fills the
        upper half of the interior and the bottom ramp the lower half;
        each ramp's profile is averaged across the middle of its half,
        clear of the walls and of the line between the ramps.
    #. The full width at half maximum of each profile is measured
        directly, without assuming a shape (ramps are flat-topped on some
        scanners and bell-shaped on others):
        - peak: maximum of a parabola fitted around the middle of the
          ramp, so noise on a flat top does not bias it upwards;
        - background on each side: a low percentile of the dark insert
          well beyond that end of the ramp (clear of the ramp's tails,
          and below ringing or ghosting artefacts);
        - half maximum on each side halfway between peak and background;
        - crossings located walking out from the peak on a smoothed
          profile, then refined with a straight-line fit to the edge, so
          a noise dip cannot end the ramp early.
        The bright ends of the insert are excluded throughout.
    #. Apply the ACR formula with both ramp lengths in mm:
        thickness = 0.2 * (L1 * L2) / (L1 + L2), which corrects for
        right-left misalignment of the phantom.

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

    The insert detection relies on the insert contour being clearly
    distinguishable from background. Plausibility checks on the insert
    size and the ramp contrast make the task fail with an explicit
    error, rather than report a value, when this is not the case (for
    example on a localizer).

Documented by Luis M. Santos, M.D.
luis.santos2@nih.gov

and

Alex Drysdale
alexander.drysdale@wales.nhs.uk

Measurement in mm, ramp localisation, direct FWHM measurement and
90 degree orientation correction added 08/10/2026.

"""

# Python Imports
import os
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path

# Module Imports
import cv2
import numpy as np
import pydicom
import scipy.ndimage as ndi

# Local Imports
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
class Insert:
    """The slice thickness insert, as an oriented rectangle in mm.

    Positions are in mm from the top-left corner of the (reoriented)
    image, x to the right and y down. ``u`` runs along the insert
    (left to right in the standard view) and ``v`` across it (downwards).
    """

    centre: np.ndarray  # (x, y) in mm
    e_u: np.ndarray  # unit vector along the insert, pointing right
    e_v: np.ndarray  # unit vector across the insert, pointing down
    length: float  # mm
    height: float  # mm

    @property
    def tilt(self) -> float:
        """Tilt of the insert in degrees (positive = clockwise)."""
        return float(np.rad2deg(np.arctan2(self.e_u[1], self.e_u[0])))


@dataclass
class Ramp:
    """Measurement of one ramp."""

    name: str  # "top" or "bottom"
    v: float  # position across the insert (mm)
    half_band: float  # half-width of the band averaged across the ramp (mm)
    u: np.ndarray = field(repr=False)  # positions along the ramp (mm)
    raw: np.ndarray = field(repr=False)  # profile before smoothing
    profile: np.ndarray = field(repr=False)  # smoothed profile
    peak: float = 0.0
    background: tuple = (0.0, 0.0)  # (left, right)
    half_max: tuple = (0.0, 0.0)  # (left, right)
    ends: tuple = (0.0, 0.0)  # (left, right) half-maximum crossings (mm)

    @property
    def width(self) -> float:
        """FWHM of the ramp profile (mm)."""
        return self.ends[1] - self.ends[0]


class ACRSliceThickness(HazenTask):
    """Slice width measurement class for DICOM images of the ACR phantom.

    Automatically detects the slice thickness insert using computer
    vision, locates the two crossed ramps and measures the full width at
    half maximum of each ramp's signal profile to calculate slice
    thickness. Series exported rotated by a multiple of 90 degrees or
    mirrored are corrected to the standard view; small residual tilts of
    the phantom are handled by measuring in the insert's own frame.
    """

    # Expected size of the slice thickness insert (mm)
    INSERT_LENGTH_RANGE = (140.0, 200.0)
    INSERT_HEIGHT_RANGE = (5.0, 25.0)
    MAX_INSERT_TILT_DEG = 15.0
    MIN_CONTOUR_AREA_MM2 = 200.0

    END_MARGIN = 4.0  # mm kept clear of the bright ends of the insert
    EDGE_MARGIN = 0.5  # mm kept clear of the long sides of the insert
    SAMPLE_STEP = 0.1  # mm, sampling along and across the insert
    PROFILE_SMOOTH = 1.0  # mm, Gaussian sigma along the ramp (display)
    COARSE_SMOOTH = 3.0  # mm, Gaussian sigma for locating ramp ends
    # Fraction of the half-length of the insert beyond which the ramps of a
    # normal slice do not reach; used to find the insert's dark interior.
    RAMP_FREE_FRACTION = 0.75
    INTERIOR_HEIGHT_RANGE = (4.0, 20.0)  # mm, plausible interior height
    RAMP_BAND_MARGIN = 0.75  # mm kept clear of the walls and ramp divider
    # The background on each side is taken from this fraction of the ramp
    # width beyond the ramp end, out to the end of the insert, and must
    # cover at least MIN_BACKGROUND_LENGTH mm.
    BACKGROUND_GAP_FRACTION = 0.6
    MIN_BACKGROUND_GAP = 3.0
    MIN_BACKGROUND_LENGTH = 5.0
    BACKGROUND_PERCENTILE = 10
    # The peak level is fitted over +/- this fraction of the ramp width
    PEAK_WINDOW_FRACTION = 0.15
    # Each half-maximum crossing is refined by a line fit over +/- this
    # fraction of the ramp width (at least MIN_EDGE_WINDOW mm)
    EDGE_WINDOW_FRACTION = 0.05
    MIN_EDGE_WINDOW = 1.5
    MIN_RAMP_CONTRAST = 3.0  # peak above background, in noise SDs

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
        self._orientation = None

    # ------------------------------------------------------------------
    # Orientation and spacing
    # ------------------------------------------------------------------
    @property
    def pixel_spacing(self) -> tuple:
        """(row, column) spacing in mm as stored in the DICOM, before any
        reorientation. ACRObject.dx/dy hold PixelSpacing[0]/[1]."""
        return (self.ACR_obj.dx, self.ACR_obj.dy)

    @property
    def orientation(self) -> tuple:
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
    def spacing(self) -> tuple:
        """(row, column) spacing in mm of the reoriented image."""
        return reorient_spacing(self.pixel_spacing, self.orientation[0])

    # ------------------------------------------------------------------
    # Task
    # ------------------------------------------------------------------
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
                        description=f"{ramp} ramp length",
                        unit="mm",
                        value=round(
                            thickness_results["ramps"][ramp]["width"], 2
                        ),
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

        Identify the ramps, measure the line profiles, measure the
        FWHM, and use this to calculate the slice thickness.

        Returns:
            dict: Dictionary containing:
                - ``"thickness"`` (float): Calculated slice thickness in mm.
                - ``"ramps"`` (dict): ``{"top": {"width": mm},
                  "bottom": {"width": mm}}``, the FWHM of each ramp.

        """
        img = np.ascontiguousarray(
            reorient(dcm.pixel_array.astype(float), *self.orientation)
        )
        insert = self.find_insert(img)
        ramps = self.measure_ramps(img, insert)
        top, bottom = ramps
        slice_thickness = (
            0.2 * (top.width * bottom.width) / (top.width + bottom.width)
        )
        logger.info(
            "Ramp lengths: top %.2f mm, bottom %.2f mm -> slice thickness"
            " %.2f mm",
            top.width,
            bottom.width,
            slice_thickness,
        )

        if self.report:
            self.write_report(dcm, img, insert, ramps, slice_thickness)

        return {
            "thickness": slice_thickness,
            "ramps": {r.name: {"width": r.width} for r in ramps},
        }

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    def to_pixels(self, insert: Insert, u, v) -> tuple:
        """(row, col) pixel coordinates of insert-frame positions (mm)."""
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        x = insert.centre[0] + u * insert.e_u[0] + v * insert.e_v[0]
        y = insert.centre[1] + u * insert.e_u[1] + v * insert.e_v[1]
        row_sp, col_sp = self.spacing
        return y / row_sp, x / col_sp

    def sample(self, img: np.ndarray, insert: Insert, u, v) -> np.ndarray:
        """Bilinearly sample the image at insert-frame positions (mm)."""
        row, col = self.to_pixels(insert, u, v)
        return ndi.map_coordinates(
            img, [row, col], order=1, mode="constant", cval=0.0
        )

    # ------------------------------------------------------------------
    # Insert detection
    # ------------------------------------------------------------------
    def find_insert(self, img: np.ndarray) -> Insert:
        """Locate the slice thickness insert.

        The image is contrast enhanced (CLAHE) and binarised (Otsu). Of
        the contours of plausible size, the most elongated is taken to be
        the insert. The oriented bounding box is fitted to the contour
        points in mm, so it is correct for non-square pixels.

        Args:
            img (np.ndarray): pixel array in the standard view.

        Returns:
            Insert: geometry of the insert.

        """
        row_sp, col_sp = self.spacing
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
        contours, _ = cv2.findContours(
            img_binary.astype(np.uint8),
            mode=cv2.RETR_TREE,
            method=cv2.CHAIN_APPROX_NONE,
        )

        candidates = []
        for contour in contours:
            if (
                cv2.contourArea(contour) * row_sp * col_sp
                < self.MIN_CONTOUR_AREA_MM2
            ):
                continue
            pts_mm = (contour[:, 0, :] * (col_sp, row_sp)).astype(np.float32)
            (cx, cy), (w, h), _ = cv2.minAreaRect(pts_mm)
            length, height = max(w, h), min(w, h)
            if not (
                self.INSERT_LENGTH_RANGE[0]
                <= length
                <= self.INSERT_LENGTH_RANGE[1]
                and self.INSERT_HEIGHT_RANGE[0]
                <= height
                <= self.INSERT_HEIGHT_RANGE[1]
            ):
                continue
            candidates.append((height / length, pts_mm))
        if not candidates:
            msg = (
                "Slice thickness insert not found: no elongated structure"
                f" of {self.INSERT_LENGTH_RANGE[0]:.0f}-"
                f"{self.INSERT_LENGTH_RANGE[1]:.0f} mm by"
                f" {self.INSERT_HEIGHT_RANGE[0]:.0f}-"
                f"{self.INSERT_HEIGHT_RANGE[1]:.0f} mm"
            )
            raise ValueError(msg)
        _, pts_mm = min(candidates, key=lambda c: c[0])

        box = cv2.boxPoints(cv2.minAreaRect(pts_mm))
        centre = box.mean(axis=0).astype(float)
        sides = [box[1] - box[0], box[2] - box[1]]
        long_side, short_side = sorted(sides, key=np.linalg.norm)[::-1]
        e_u = long_side / np.linalg.norm(long_side)
        if e_u[0] < 0:
            e_u = -e_u  # point to the right
        e_v = np.array([-e_u[1], e_u[0]])  # +90 deg with y down: points down
        insert = Insert(
            centre=centre,
            e_u=e_u.astype(float),
            e_v=e_v.astype(float),
            length=float(np.linalg.norm(long_side)),
            height=float(np.linalg.norm(short_side)),
        )
        insert = self.refine_insert(img, insert)
        if abs(insert.tilt) > self.MAX_INSERT_TILT_DEG:
            msg = (
                f"Slice thickness insert tilted by {insert.tilt:.1f} deg after"
                " the 90 degree orientation correction; only small tilts are"
                " supported"
            )
            raise ValueError(msg)
        logger.info(
            "Insert: %.1f x %.1f mm, tilt %.2f deg",
            insert.length,
            insert.height,
            insert.tilt,
        )
        return insert

    def refine_insert(self, img: np.ndarray, insert: Insert) -> Insert:
        """Refine the tilt and position of the insert from its long walls.

        The bounding box of a thresholded contour can be off by a degree or
        so, which would make the ramp profiles cut obliquely across the
        ramps. Instead, the water/insert edge is located on profiles across
        the insert at many positions along it, and straight lines are
        fitted to the upper and lower walls.
        """
        step = self.SAMPLE_STEP
        v = np.arange(
            -insert.height / 2 - 4, insert.height / 2 + 4 + step / 2, step
        )
        for _ in range(3):
            u_pos = np.arange(
                -insert.length / 2 + 10, insert.length / 2 - 10 + 1e-9, 2.0
            )
            uu, vv = np.meshgrid(u_pos, v)
            prof = ndi.gaussian_filter1d(
                self.sample(img, insert, uu, vv), 0.3 / step, axis=0
            )
            water = np.median(np.r_[prof[:10], prof[-10:]])
            dark = np.median(prof[np.abs(v) < 1.0])
            half = 0.5 * (water + dark)
            tops, bottoms, keep = [], [], []
            for j in range(len(u_pos)):
                p = prof[:, j]
                below = np.nonzero(p < half)[0]
                if len(below) == 0 or below[0] == 0 or below[-1] == len(v) - 1:
                    continue
                # walk in from the water on each side to the first dark
                # sample, so ramps inside the insert are not reached
                i, k = below[0], below[-1]
                tops.append(
                    v[i - 1] + step * (p[i - 1] - half) / (p[i - 1] - p[i])
                )
                bottoms.append(v[k] + step * (p[k] - half) / (p[k] - p[k + 1]))
                keep.append(u_pos[j])
            if len(keep) < 10:
                return insert  # keep the bounding box estimate
            keep = np.array(keep)
            c_top = np.polyfit(keep, tops, 1)
            c_bot = np.polyfit(keep, bottoms, 1)
            slope = 0.5 * (c_top[0] + c_bot[0])
            v_shift = 0.5 * (c_top[1] + c_bot[1])
            height = float(c_bot[1] - c_top[1])
            angle = np.arctan(slope)
            c, s = np.cos(angle), np.sin(angle)
            e_u = c * insert.e_u + s * insert.e_v
            e_u /= np.linalg.norm(e_u)
            e_v = np.array([-e_u[1], e_u[0]])
            insert = Insert(
                centre=insert.centre + v_shift * insert.e_v,
                e_u=e_u,
                e_v=e_v,
                length=insert.length,
                height=height,
            )
            if abs(slope) < 1e-4 and abs(v_shift) < 0.01:
                break
        return insert

    # ------------------------------------------------------------------
    # Ramp measurement
    # ------------------------------------------------------------------
    def measure_ramps(self, img: np.ndarray, insert: Insert) -> list:
        """Locate the two ramps across the insert and measure each.

        Returns:
            list[Ramp]: the top and bottom ramps, in that order.

        """
        step = self.SAMPLE_STEP
        u_max = insert.length / 2 - self.END_MARGIN
        u = np.arange(-u_max, u_max + step / 2, step)

        # The dark interior of the insert, found across its outer ends where
        # the ramps do not reach and the walls give clean edges. (The
        # bounding box of the contour can be larger than the interior.)
        v_scan = np.arange(
            -insert.height / 2 - 3, insert.height / 2 + 3 + step / 2, step
        )
        ends = np.abs(u) > self.RAMP_FREE_FRACTION * u_max
        uu, vv = np.meshgrid(u[ends], v_scan)
        end_level = ndi.gaussian_filter1d(
            np.median(self.sample(img, insert, uu, vv), axis=1), 0.2 / step
        )
        dark = float(end_level.min())
        water = float(np.median(np.r_[end_level[:10], end_level[-10:]]))
        inside = end_level < 0.5 * (dark + water)
        i_dark = int(np.argmin(end_level))
        lo = hi = i_dark
        while lo > 0 and inside[lo - 1]:
            lo -= 1
        while hi < len(v_scan) - 1 and inside[hi + 1]:
            hi += 1
        v_a, v_b = float(v_scan[lo]), float(v_scan[hi])
        if (
            not self.INTERIOR_HEIGHT_RANGE[0]
            <= v_b - v_a
            <= (self.INTERIOR_HEIGHT_RANGE[1])
        ):
            msg = (
                f"Implausible height of the insert interior: {v_b - v_a:.1f}"
                " mm"
            )
            raise ValueError(msg)

        # The top ramp fills the upper half of the interior and the bottom
        # ramp the lower half. Each profile is averaged across the middle
        # of its half, clear of the walls and of the line between the ramps.
        quarter = (v_b - v_a) / 4
        half_band = max(quarter - self.RAMP_BAND_MARGIN, 0.5)
        v_ramps = (v_a + quarter, v_b - quarter)
        logger.info(
            "Insert interior %.1f to %.1f mm; ramps at %.1f and %.1f mm",
            v_a,
            v_b,
            *v_ramps,
        )

        ramps = []
        for name, v_ramp in zip(("top", "bottom"), v_ramps, strict=True):
            offsets = np.arange(-half_band, half_band + step / 2, step)
            uu, vv = np.meshgrid(u, v_ramp + offsets)
            raw = self.sample(img, insert, uu, vv).mean(axis=0)
            ramp = Ramp(
                name=name,
                v=float(v_ramp),
                half_band=half_band,
                u=u,
                raw=raw,
                profile=ndi.gaussian_filter1d(raw, self.PROFILE_SMOOTH / step),
            )
            self.measure_fwhm(ramp)
            ramps.append(ramp)
        return ramps

    def measure_fwhm(self, ramp: Ramp) -> None:
        """Measure the full width at half maximum of a ramp profile.

        The half-maximum level on each side is halfway between the peak
        and the background on that side, taken from the dark insert well
        beyond that end of the ramp. Crossings are first found walking
        outwards from the peak on a well smoothed profile (so internal dips
        or the far ends of the insert cannot be picked up), then refined by
        a straight-line fit to the unsmoothed profile around each edge.
        """
        u = ramp.u
        step = u[1] - u[0]
        # Heavier smoothing for locating the ramp; a symmetric kernel moves
        # the half-maximum point of an edge only negligibly.
        p = ndi.gaussian_filter1d(ramp.raw, self.COARSE_SMOOTH / step)
        # The ramp peaks in the central part of the insert; near the ends,
        # bleed from the bright end walls can be higher than the ramp.
        central = np.abs(u) <= self.RAMP_FREE_FRACTION * np.abs(u).max()
        i_peak = int(np.argmax(np.where(central, p, -np.inf)))
        peak = float(p[i_peak])

        def crossing(level: float, direction: int) -> tuple:
            """Walk from the peak until the profile drops below level."""
            i = i_peak
            while 0 <= i + direction < len(p) and p[i + direction] >= level:
                i += direction
            j = i + direction
            if not 0 <= j < len(p):
                msg = (
                    f"The {ramp.name} ramp extends beyond the measurable part"
                    " of the insert"
                )
                raise ValueError(msg)
            # linear interpolation between samples i (>= level) and j
            frac = (p[i] - level) / (p[i] - p[j])
            return float(u[i] + direction * frac * step), j

        # First pass with a global background, to locate the ramp ends
        background0 = float(np.percentile(p, 5))
        half0 = 0.5 * (peak + background0)
        left0, _ = crossing(half0, -1)
        right0, _ = crossing(half0, +1)

        # Peak level: the maximum of a parabola fitted around the middle of
        # the ramp. On a flat-topped ramp this averages the plateau, so noise
        # does not bias the peak upwards (as the maximum sample would); on a
        # bell-shaped ramp it recovers the true maximum.
        centre = 0.5 * (left0 + right0)
        half_window = max(self.PEAK_WINDOW_FRACTION * (right0 - left0), 2.0)
        win = np.abs(u - centre) <= half_window
        coef = np.polyfit(u[win] - centre, ramp.raw[win], 2)
        fitted = np.polyval(coef, u[win] - centre)
        # never above the maximum of the lightly smoothed profile
        peak = min(float(fitted.max()), float(ramp.profile.max()))

        # Background on each side: the dark insert beyond the ramp. The ramp
        # profile is the slice profile stretched 10 times, so it can have
        # long tails; the background region starts well clear of them. The
        # plastic itself gives no signal, so anything above its floor
        # (ghosting, ringing) is artefact: a low percentile of the smoothed
        # profile is used rather than the mean.
        gap = self.BACKGROUND_GAP_FRACTION * (right0 - left0)
        backgrounds = []
        for end, direction in ((left0, -1), (right0, +1)):
            # For long ramps (thick slices) shrink the gap so that at least
            # MIN_BACKGROUND_LENGTH mm of insert remains, down to a minimum.
            room = (u.max() if direction > 0 else -u.min()) - direction * end
            this_gap = max(
                min(gap, room - self.MIN_BACKGROUND_LENGTH),
                self.MIN_BACKGROUND_GAP,
            )
            start = end + direction * this_gap
            sel = (u - start) * direction >= 0
            backgrounds.append(
                float(np.percentile(p[sel], self.BACKGROUND_PERCENTILE))
                if sel.sum() * step >= self.MIN_BACKGROUND_LENGTH
                else None
            )
        if backgrounds[0] is None and backgrounds[1] is None:
            msg = f"No background available around the {ramp.name} ramp"
            raise ValueError(msg)
        # If one end of the insert is too close, use the other side's level
        available = next(b for b in backgrounds if b is not None)
        backgrounds = [b if b is not None else available for b in backgrounds]

        # Contrast check: the ramp must stand clear of the profile noise
        noise = float(np.std(ramp.raw - ramp.profile))
        if peak - max(backgrounds) < self.MIN_RAMP_CONTRAST * max(noise, 1e-9):
            msg = f"The {ramp.name} ramp signal is too weak to measure"
            raise ValueError(msg)

        half = tuple(0.5 * (peak + b) for b in backgrounds)
        left, _ = crossing(half[0], -1)
        right, _ = crossing(half[1], +1)

        # Refine each crossing with a straight-line fit to the edge. Walking
        # out from the peak stops at the first noise dip below half maximum,
        # which shortens the ramp on noisy images; a fit averages the noise.
        half_window = max(
            self.EDGE_WINDOW_FRACTION * (right - left), self.MIN_EDGE_WINDOW
        )

        def refine(position: float, level: float, rising: bool) -> float:
            for _ in range(3):
                sel = np.abs(u - position) <= half_window
                slope, intercept = np.polyfit(u[sel], ramp.raw[sel], 1)
                if (slope > 0) != rising or slope == 0:
                    return position  # no clear edge; keep the walked value
                new = (level - intercept) / slope
                if abs(new - position) > half_window:
                    return position
                position = float(new)
            return position

        left = refine(left, half[0], rising=True)
        right = refine(right, half[1], rising=False)

        ramp.peak = peak
        ramp.background = tuple(backgrounds)
        ramp.half_max = half
        ramp.ends = (left, right)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    def write_report(
        self,
        dcm: pydicom.Dataset,
        img: np.ndarray,
        insert: Insert,
        ramps: list,
        slice_thickness: float,
    ) -> str:
        """Save the report image and return its path."""
        import matplotlib.pyplot as plt  # noqa: PLC0415

        colours = {"top": "tab:blue", "bottom": "tab:orange"}
        fig = plt.figure(figsize=(16, 10), layout="constrained")
        grid = fig.add_gridspec(2, 2, height_ratios=[1.2, 1])
        ax_img = fig.add_subplot(grid[0, 0])
        ax_ins = fig.add_subplot(grid[0, 1])
        ax_prof = [fig.add_subplot(grid[1, i]) for i in range(2)]

        # 1. Image in the standard view with the insert and ramp lines
        row_sp, col_sp = self.spacing
        ax_img.imshow(img, cmap="gray", aspect=row_sp / col_sp)
        half_l, half_h = insert.length / 2, insert.height / 2
        corners_u = np.array([-half_l, half_l, half_l, -half_l, -half_l])
        corners_v = np.array([-half_h, -half_h, half_h, half_h, -half_h])
        rows, cols = self.to_pixels(insert, corners_u, corners_v)
        ax_img.plot(cols, rows, c="lime", lw=0.8)
        for r in ramps:
            rows, cols = self.to_pixels(insert, np.array(r.ends), [r.v, r.v])
            ax_img.plot(cols, rows, c=colours[r.name], lw=1.5)
        title = "Slice 1 with insert and ramp lengths"
        if self.orientation != (0, False):
            title += (
                f"\n(image {describe_orientation(*self.orientation)} "
                "to standard view)"
            )
        ax_img.set_title(title)
        ax_img.axis("off")

        # 2. Insert, resampled along its own axis
        step = 0.2
        u = np.arange(-half_l - 3, half_l + 3, step)
        v = np.arange(-half_h - 3, half_h + 3, step)
        uu, vv = np.meshgrid(u, v)
        crop = self.sample(img, insert, uu, vv)
        inside = (np.abs(uu) < half_l - self.END_MARGIN) & (
            np.abs(vv) < half_h - self.EDGE_MARGIN
        )
        lo, hi = np.percentile(crop[inside], (1, 99.5))
        ax_ins.imshow(
            crop,
            cmap="gray",
            vmin=lo,
            vmax=hi,
            extent=[u[0], u[-1], v[-1], v[0]],
            aspect="auto",
        )
        for r in ramps:
            for edge in (-r.half_band, r.half_band):
                ax_ins.axhline(r.v + edge, c=colours[r.name], lw=0.6, ls=":")
            ax_ins.plot(r.ends, [r.v, r.v], c=colours[r.name], lw=2)
            ax_ins.plot(r.ends, [r.v, r.v], "|", c=colours[r.name], ms=14)
        ax_ins.set_xlabel("Distance along insert (mm)")
        ax_ins.set_ylabel("Across insert (mm)")
        ax_ins.set_title(
            f"Insert (tilt {insert.tilt:+.1f}°), windowed to the ramps"
        )

        # 3. Profiles
        for ax, r in zip(ax_prof, ramps, strict=True):
            c = colours[r.name]
            ax.plot(r.u, r.raw, c=c, alpha=0.3, label="Profile")
            ax.plot(r.u, r.profile, c=c, label="Smoothed (1 mm)")
            for end, half in zip(r.ends, r.half_max, strict=True):
                ax.plot(end, half, "o", c="k", ms=5)
            ax.hlines(
                r.half_max,
                [r.u[0], r.ends[1]],
                [r.ends[0], r.u[-1]],
                colors=c,
                linestyles="--",
                lw=0.8,
            )
            ax.axvspan(*r.ends, color=c, alpha=0.08)
            ax.set_title(f"{r.name.capitalize()} ramp: FWHM {r.width:.2f} mm")
            ax.set_xlabel("Distance along insert (mm)")
            ax.set_ylabel("Pixel value")
            ax.grid(alpha=0.3)
            ax.legend(loc="upper right")
        fig.suptitle(
            f"Slice thickness = 0.2 × (top × bottom) / (top + bottom)"
            f" = {slice_thickness:.2f} mm"
        )

        img_path = os.path.realpath(
            Path(self.report_path)
            / f"{self.img_desc(dcm)}_slice_thickness.png",
        )
        fig.savefig(img_path, dpi=100)
        plt.close(fig)
        self.report_files.append(img_path)
        return img_path
