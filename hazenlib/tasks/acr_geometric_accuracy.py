"""
ACR Geometric Accuracy

https://www.acraccreditation.org/-/media/acraccreditation/documents/mri/largephantomguidance.pdf

Calculates geometric accuracy for slices 1 and 5 of the ACR phantom.

This script calculates the horizontal and vertical lengths of the ACR phantom in Slice 1 in accordance with the ACR
Guidance.
This script calculates the horizontal, vertical and diagonal lengths of the ACR phantom in Slice 5 in accordance with
the ACR Guidance.
The average distance measurement error, maximum distance measurement error and coefficient of variation of all distance
measurements is reported as recommended by IPEM Report 112, "Quality Control and Artefacts in Magnetic Resonance
Imaging".

Method
------
0. The phantom orientation is determined from slice 1 (dark bar horizontal, resolution insert below it, ring insert
   upper-left) and every slice is rotated by a multiple of 90 degrees (and mirrored if needed) into the standard view
   before analysis. This makes the direction labels and report images correct even if a series was exported rotated.
1. The slice is lightly smoothed and segmented with an Otsu threshold; holes are filled, the largest connected
   component is kept and its convex hull taken. This mask is only used as a *rough* guide to where the phantom is.
2. The phantom centre is initialised at the mask centroid and then refined so that it sits at the midpoint of the
   measured horizontal and vertical diameters.
3. For each measurement direction, intensity profiles are sampled (bilinear interpolation, 0.1 mm steps, in physical
   mm space so that diagonals are true 45 degree lines even with non-square pixels) along the line through the centre
   and along parallel lines offset by up to +/-45 mm.
4. On each profile the phantom edges are located at the outermost strong peak of the intensity gradient (the 50% point
   of the blurred wall step), refined to sub-sample precision.
5. At each end of the diameter a circle is fitted robustly (RANSAC, then least squares on the inliers) to the edge
   points from all the parallel profiles, and evaluated on the measurement line. Where the wall is clear this equals
   the edge on the line itself. Where internal structures meet the wall - notably the wedge housing and the notch
   cut-outs at the top of the phantom, which a single profile through the centre runs into - those points are
   rejected and the wall is interpolated across them instead of the structure being mistaken for the phantom edge.
   The wall points used are shown in the report.

Created by Yassine Azma
yassine.azma@rmh.nhs.uk

18/11/2022
"""

import os
import sys
import traceback

import numpy as np
import scipy.ndimage as ndi
from hazenlib.ACRObject import ACRObject
from hazenlib.HazenTask import HazenTask
from hazenlib.logger import logger
from hazenlib.orientation import (
    describe_orientation,
    segment_phantom,
)
from hazenlib.types import Measurement, Result
from hazenlib.utils import fit_circle

NOMINAL_DIAMETER_MM = 190.0

# Fitting the phantom wall around each end of a diameter (see measure_length).
WALL_ARC_HALF_WIDTH_MM = (
    45.0  # wall sampled up to this far either side of the line
)
WALL_FIT_TOLERANCE_MM = (
    0.25  # max distance of a wall point from the fitted circle
)
MIN_WALL_INLIER_FRACTION = (
    0.3  # fraction of sampled points that must be on the wall
)
MIN_WALL_RADIUS_MM = 70.0  # plausible local radius of curvature of the wall
MAX_WALL_RADIUS_MM = 140.0

# Unit vectors (x, y) in physical space. Image y increases downwards.
DIRECTIONS = {
    "Horizontal": (1.0, 0.0),
    "Vertical": (0.0, 1.0),
    "Diagonal SW": (
        -1.0 / np.sqrt(2),
        1.0 / np.sqrt(2),
    ),  # top-right -> bottom-left
    "Diagonal SE": (
        1.0 / np.sqrt(2),
        1.0 / np.sqrt(2),
    ),  # top-left -> bottom-right
}

SLICE_1_DIRECTIONS = ("Horizontal", "Vertical")
SLICE_5_DIRECTIONS = ("Horizontal", "Vertical", "Diagonal SW", "Diagonal SE")


# --------------------------------------------------------------------------- #
# Measurement helpers (module level so they can be tested without DICOM data)
# --------------------------------------------------------------------------- #
def _edge_position(s, grad, s_rough, inward, n_iter=2, search=10.0):
    """Locate one phantom edge on a profile at sub-sample precision.

    The edge is taken as the outermost strong peak of the intensity gradient
    pointing into the phantom (background -> water). For a blurred step this
    is the 50% point, but unlike a half-maximum threshold it does not need an
    estimate of the water signal level, which is unreliable where internal
    structures (e.g. the slice 5 grid on the diagonals) sit close to the wall.

    Args:
        s (np.ndarray): position along the profile (mm), uniformly spaced.
        grad (np.ndarray): derivative of the (smoothed) profile with respect to s.
        s_rough (float): approximate edge position (mm) from the mask.
        inward (int): +1 if the phantom lies at larger s than the edge, -1 otherwise.
        search (float): mm either side of the current estimate to search.

    Returns:
        float: edge position (mm).
    """
    step = s[1] - s[0]
    g_in = (
        grad * inward
    )  # positive where intensity rises going into the phantom

    edge = s_rough
    for _ in range(n_iter):
        d = (s - edge) * inward  # > 0 inside phantom
        idx = np.nonzero(np.abs(d) <= search)[0]
        if inward < 0:
            idx = idx[::-1]  # always walk from outside towards inside
        g = g_in[idx]
        if g.size < 3 or g.max() <= 0:
            logger.warning("Could not refine phantom edge; using mask edge")
            break

        # Outermost local maximum that is a substantial rising step. Taking the
        # first from outside (not the global max) avoids locking onto the
        # rising edges of water-filled holes in internal structures.
        strong = g >= 0.5 * g.max()
        is_peak = np.zeros_like(strong)
        is_peak[1:-1] = (g[1:-1] >= g[:-2]) & (g[1:-1] >= g[2:])
        candidates = np.nonzero(strong & is_peak)[0]
        k = int(candidates[0]) if candidates.size else int(np.argmax(g))

        # Parabolic sub-sample refinement of the peak
        offset = 0.0
        if 0 < k < g.size - 1:
            denom = g[k - 1] - 2 * g[k] + g[k + 1]
            if denom != 0:
                offset = np.clip(
                    0.5 * (g[k - 1] - g[k + 1]) / denom, -0.5, 0.5
                )
        edge = s[idx[k]] + offset * step * inward
    return edge


def _robust_wall_fit(t, s, tol=WALL_FIT_TOLERANCE_MM, n_trials=400, seed=0):
    """Fit a circle to wall edge points, ignoring points on internal features.

    Edges found where an internal structure meets the wall (e.g. the wedge
    housing at the top of the phantom) do not lie on the wall circle. A
    RANSAC search finds the circle supported by the most points, which is
    then refined by least squares on its inliers.

    Args:
        t (np.ndarray): offset of each profile from the measurement line (mm).
        s (np.ndarray): edge position along each profile (mm).

    Returns:
        tuple: (tc, sc, r, inlier mask) or None if no reliable fit was found.
    """
    rng = np.random.default_rng(seed)
    n = len(t)
    # Circles through all the random triples at once: x^2 + y^2 + D x +
    # E y + F = 0 through each triple's three points
    idx = np.array([rng.choice(n, 3, replace=False) for _ in range(n_trials)])
    A = np.stack([t[idx], s[idx], np.ones(idx.shape)], axis=-1)
    ok = np.abs(np.linalg.det(A)) > 1e-12  # not collinear
    b = -(t[idx[ok]] ** 2 + s[idx[ok]] ** 2)
    D, E, F = np.linalg.solve(A[ok], b[..., None])[..., 0].T
    tc, sc = -D / 2, -E / 2
    r = np.sqrt(np.maximum(tc**2 + sc**2 - F, 0.0))
    plausible = (0.5 * MIN_WALL_RADIUS_MM < r) & (r < 2 * MAX_WALL_RADIUS_MM)
    if not plausible.any():
        return None
    tc, sc, r = tc[plausible], sc[plausible], r[plausible]
    dist = np.hypot(t[None, :] - tc[:, None], s[None, :] - sc[:, None])
    inliers = np.abs(dist - r[:, None]) < tol
    # the first trial with the most inliers
    inl = inliers[int(np.argmax(inliers.sum(axis=1)))]
    if inl.sum() < 3:
        return None
    for _ in range(3):
        tc, sc, r = fit_circle(t[inl], s[inl])
        new = np.abs(np.hypot(t - tc, s - sc) - r) < tol
        if new.sum() < 3 or np.array_equal(new, inl):
            break
        inl = new

    # Only interpolate: the wall must be seen on both sides of the line.
    if (
        inl.sum() < MIN_WALL_INLIER_FRACTION * n
        or not (t[inl] < -0.25 * t.max()).any()
        or not (t[inl] > 0.25 * t.max()).any()
        or not MIN_WALL_RADIUS_MM < r < MAX_WALL_RADIUS_MM
    ):
        return None
    return tc, sc, r, inl


def measure_length(
    smooth,
    mask,
    centre,
    direction,
    spacing,
    half_length=140.0,
    step=0.1,
    arc_half_width=WALL_ARC_HALF_WIDTH_MM,
    arc_step=0.5,
):
    """Measure the phantom diameter along a line through `centre`.

    Each end of the diameter is found from the phantom wall around it rather
    than from the single profile along the line: the edge is located on
    parallel profiles offset by up to +/- `arc_half_width` mm, a circle is
    fitted robustly to those wall points, and the fitted wall is evaluated on
    the measurement line. Where the wall is clear this equals the edge on the
    line itself; where internal structures meet the wall (e.g. the wedge
    housing and fill shoulder at the top of the phantom) they are rejected
    instead of being mistaken for the phantom edge.

    Args:
        smooth (np.ndarray): smoothed image.
        mask (np.ndarray): rough phantom mask.
        centre (tuple): (x, y) centre in pixels.
        direction (tuple): (ux, uy) unit vector in physical space.
        spacing (tuple): (row_spacing, column_spacing) in mm, as DICOM PixelSpacing.

    Returns:
        dict: "Start"/"End" (x, y) in pixels, "Distance" in mm, and
        "WallPoints" (x, y arrays in pixels of the wall points used).
    """
    cx, cy = centre
    ux, uy = direction
    px_, py_ = -uy, ux  # perpendicular unit vector
    row_sp, col_sp = spacing
    pixel_mm = 0.5 * (row_sp + col_sp)

    s = np.arange(-half_length, half_length + step, step)
    offsets = np.arange(
        -arc_half_width, arc_half_width + arc_step / 2, arc_step
    )

    def to_px(t, s_):
        return (
            cx + (s_ * ux + t * px_) / col_sp,
            cy + (s_ * uy + t * py_) / row_sp,
        )

    # All the parallel profiles at once: one row per offset
    xs, ys = to_px(offsets[:, None], s[None, :])
    coords = [ys.ravel(), xs.ravel()]
    profs = ndi.map_coordinates(
        smooth, coords, order=1, mode="constant", cval=0.0
    ).reshape(xs.shape)
    mprofs = ndi.map_coordinates(
        mask.astype(float), coords, order=0, mode="constant", cval=0.0
    ).reshape(xs.shape)
    # Smooth over ~1 pixel before differentiating: bilinear interpolation makes
    # the profile piecewise linear, so its raw derivative is stepwise and noisy.
    # A symmetric kernel does not move the position of a step edge.
    grads = (
        ndi.gaussian_filter1d(profs, sigma=pixel_mm / step, order=1, axis=1)
        / step
    )

    edges = {+1: [], -1: []}
    for grad, mprof in zip(grads, mprofs):
        inside = np.nonzero(mprof > 0.5)[0]
        if inside.size == 0:
            edges[+1].append(np.nan)
            edges[-1].append(np.nan)
            continue
        edges[+1].append(_edge_position(s, grad, s[inside[0]], +1))
        edges[-1].append(_edge_position(s, grad, s[inside[-1]], -1))

    centre_idx = int(np.argmin(np.abs(offsets)))
    if np.isnan(edges[+1][centre_idx]):
        raise ValueError("Profile does not intersect the phantom")

    ends, wall_pts = {}, {}
    for inward in (+1, -1):
        e = np.asarray(edges[inward])
        ok = ~np.isnan(e)
        fit = _robust_wall_fit(offsets[ok], e[ok])
        if fit is None:
            logger.warning(
                "Could not fit the phantom wall around the %s end of direction %s; "
                "using the edge on the measurement line",
                "start" if inward > 0 else "end",
                direction,
            )
            ends[inward] = e[centre_idx]
            wall_pts[inward] = np.zeros(0, dtype=bool), offsets[ok], e[ok]
            continue
        tc, sc, r, inl = fit
        # The start of the line (inward = +1) is on the near side of the circle.
        ends[inward] = sc - inward * np.sqrt(max(r**2 - tc**2, 0.0))
        wall_pts[inward] = inl, offsets[ok], e[ok]

    s0, s1 = ends[+1], ends[-1]
    used = [to_px(t[inl], e[inl]) for inl, t, e in wall_pts.values()]
    return {
        "Start": to_px(0.0, s0),
        "End": to_px(0.0, s1),
        "Distance": s1 - s0,
        "WallPoints": (
            np.concatenate([u[0] for u in used]),
            np.concatenate([u[1] for u in used]),
        ),
    }


def measure_phantom(img, spacing, directions):
    """Segment the phantom, refine its centre and measure the requested diameters.

    Returns:
        tuple: (centre (x, y) in pixels, mask, dict of direction name -> measurement dict)
    """
    smooth, mask = segment_phantom(img)
    cy, cx = ndi.center_of_mass(mask)
    centre = (cx, cy)

    # Refine the centre: the true centre is the midpoint of the diameters
    for _ in range(3):
        h = measure_length(
            smooth, mask, centre, DIRECTIONS["Horizontal"], spacing
        )
        v = measure_length(
            smooth, mask, centre, DIRECTIONS["Vertical"], spacing
        )
        centre = (
            0.5 * (h["Start"][0] + h["End"][0]),
            0.5 * (v["Start"][1] + v["End"][1]),
        )

    lengths = {
        name: measure_length(smooth, mask, centre, DIRECTIONS[name], spacing)
        for name in directions
    }
    return centre, mask, lengths


# --------------------------------------------------------------------------- #
# Hazen task
# --------------------------------------------------------------------------- #
class ACRGeometricAccuracy(HazenTask):
    """Geometric accuracy measurement class for DICOM images of the ACR phantom."""

    def __init__(self, orientation=None, **kwargs):
        if kwargs.pop("verbose", None) is not None:
            logger.warning(
                "verbose is not a supported argument for %s",
                type(self).__name__,
            )
        super().__init__(**kwargs)
        self.ACR_obj = ACRObject(self.dcm_list, orientation)

    def run(self) -> Result:
        """Main function for performing geometric accuracy measurement using the first and fifth slices from the ACR phantom image set.

        Returns:
            dict: results are returned in a standardised dictionary structure specifying the task name, input DICOM Series Description + SeriesNumber + InstanceNumber, task measurement key-value pairs, optionally path to the generated images for visualisation.
        """
        results = self.init_result_dict(desc=self.ACR_obj.acquisition_type())
        results.files = [
            self.img_desc(self.ACR_obj.slice_stack[0]),
            self.img_desc(self.ACR_obj.slice_stack[4]),
        ]

        all_lengths = []
        for slice_index, names in (
            (0, SLICE_1_DIRECTIONS),
            (4, SLICE_5_DIRECTIONS),
        ):
            dcm = self.ACR_obj.slice_stack[slice_index]
            try:
                lengths = self.get_geometric_accuracy(slice_index)
            except Exception as e:
                logger.exception(
                    "Could not calculate the geometric accuracy for %s because of : %s",
                    self.img_desc(dcm),
                    e,
                )
                traceback.print_exc(file=sys.stdout)
                continue

            for length, name in zip(lengths, names):
                results.add_measurement(
                    Measurement(
                        name="GeometricAccuracy",
                        type="measured",
                        subtype=f"Slice {slice_index + 1} {name} distance",
                        description=self.img_desc(dcm),
                        value=round(length, 2),
                        unit="mm",
                    ),
                )
            all_lengths.extend(lengths)

        if all_lengths:
            mean_err, max_err, cov_l = self.get_distortion_metrics(all_lengths)
            results.add_measurement(
                Measurement(
                    name="GeometricAccuracy",
                    type="measured",
                    subtype="distortion",
                    description="Mean relative error",
                    value=round(mean_err, 2),
                    visibility="intermediate",
                ),
            )
            results.add_measurement(
                Measurement(
                    name="GeometricAccuracy",
                    type="measured",
                    subtype="distortion",
                    description="Max absolute error",
                    value=round(max_err, 2),
                    visibility="intermediate",
                ),
            )
            results.add_measurement(
                Measurement(
                    name="GeometricAccuracy",
                    type="fitted",
                    subtype="distortion",
                    description="Coefficient of variation",
                    value=round(cov_l, 2),
                    unit="%",
                    visibility="intermediate",
                ),
            )

        if self.report:
            results.add_report_image(self.report_files)
            results.overlays = self.overlays

        return results

    def get_geometric_accuracy(self, slice_index):
        """Measure geometric accuracy for input slice.

        Args:
            slice_index (int): the index of the slice position, for example slice 5 is at index 4.

        Returns:
            tuple of float: horizontal and vertical distances (mm), plus the SW and SE
            diagonal distances for slice 5.
        """
        img_dcm = self.ACR_obj.slice_stack[slice_index]
        img = self.ACR_obj.standard_view(img_dcm)
        spacing = self.ACR_obj.spacing
        names = SLICE_5_DIRECTIONS if slice_index == 4 else SLICE_1_DIRECTIONS

        centre, mask, lengths = measure_phantom(img, spacing, names)

        if self.report:
            self._plot(img_dcm, img, mask, centre, lengths, slice_index)

        return tuple(lengths[name]["Distance"] for name in names)

    def _plot(self, img_dcm, img, mask, centre, lengths, slice_index):
        import matplotlib.pyplot as plt

        colours = {
            "Horizontal": "blue",
            "Vertical": "orange",
            "Diagonal SW": "yellow",
            "Diagonal SE": "purple",
        }

        fig, axes = plt.subplots(3, 1)
        fig.set_size_inches(8, 24)
        fig.tight_layout(pad=4)

        axes[0].imshow(img, cmap="gray")
        axes[0].scatter(centre[0], centre[1], c="red")
        axes[0].axis("off")
        axes[0].set_title("Centroid Location")

        axes[1].imshow(mask)
        axes[1].axis("off")
        axes[1].set_title("Rough Phantom Mask")

        axes[2].imshow(img, cmap="gray")
        for name, d in lengths.items():
            # wall points used to locate the ends of this diameter
            axes[2].scatter(
                *d["WallPoints"], s=1, color=colours[name], alpha=0.6
            )
            (x0, y0), (x1, y1) = d["Start"], d["End"]
            axes[2].plot(
                [x0, x1],
                [y0, y1],
                "-o",
                color=colours[name],
                markersize=3,
                label=f"{name}: {d['Distance']:.2f} mm",
            )
        axes[2].legend()
        axes[2].axis("off")
        title = f"Geometric Accuracy for Slice {slice_index + 1}"
        if self.ACR_obj.orientation != (0, False):
            title += f"\n(image {describe_orientation(*self.ACR_obj.orientation)} to standard view)"
        axes[2].set_title(title)

        img_path = os.path.realpath(
            os.path.join(self.report_path, f"{self.img_desc(img_dcm)}.png")
        )
        fig.savefig(img_path)
        self.save_overlay(fig, axes[2], img_path, slice_index + 1)
        plt.close(fig)
        self.report_files.append(img_path)

    @staticmethod
    def get_distortion_metrics(L):
        """Calculates the mean error, the maximum error and the coefficient of
        variation between the distances measured on slices 1 and 5.

        Args:
            L (sequence): distances from slices 1 and 5 (mm).

        Returns:
            tuple of floats: mean_err, max_err, cov_l
        """
        err = [x - NOMINAL_DIAMETER_MM for x in L]
        mean_err = np.mean(err)
        max_err = np.max(np.absolute(err))
        cov_l = 100 * np.std(L) / np.mean(L)
        return mean_err, max_err, cov_l
