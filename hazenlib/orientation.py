"""Orientation of ACR phantom images.

Series are sometimes exported rotated by a multiple of 90 degrees or
mirrored. The ACR tasks find the correction from slice 1, whose dark bar,
resolution insert and ring insert are asymmetric, and analyse every slice in
the standard view: dark bar horizontal, resolution insert below it, ring
insert upper-left.
"""

import numpy as np
import scipy.ndimage as ndi
import skimage.filters
import skimage.measure
import skimage.morphology

from hazenlib.logger import logger

# Phantom tilts smaller than this are ignored when determining the orientation
MIN_ORIENTATION_TILT_DEG = 2.0
# Below these scores (in the best orientation) the slice-1 cues are too weak to
# trust - e.g. slice 1 missing, a noise image - and no correction is applied.
MIN_BAR_SCORE = 0.3
MIN_INSERT_SCORE = 0.15


def segment_phantom(img, sigma=1.0, log_scale=False):
    """Return a lightly smoothed copy of the image and a rough phantom mask.

    With ``log_scale`` the threshold is robust to strong shading: with head
    and surface coils the dimmest water can be far below the Otsu threshold
    of the raw image, so part of the phantom is cut off and its centroid
    moves. On a log scale even dim water is far above the background noise,
    so the threshold is taken there instead.
    """
    smooth = ndi.gaussian_filter(img.astype(float), sigma)
    if log_scale:
        log = np.log1p(np.clip(smooth, 0, None))
        mask = ndi.binary_fill_holes(log > skimage.filters.threshold_otsu(log))
        mask = ndi.binary_opening(mask, iterations=2)  # detach noise specks
    else:
        mask = smooth > skimage.filters.threshold_otsu(smooth)
        mask = ndi.binary_fill_holes(mask)

    labels = skimage.measure.label(mask)
    if labels.max() == 0:
        raise ValueError("No phantom found in image")
    largest = np.argmax(np.bincount(labels.ravel())[1:]) + 1
    return smooth, skimage.morphology.convex_hull_image(labels == largest)


def reorient(img, k, flip):
    """Rotate an image by k * 90 degrees anticlockwise (np.rot90), then optionally mirror left-right."""
    out = np.rot90(img, k)
    return np.fliplr(out) if flip else out


def reorient_spacing(spacing, k):
    """(row, column) pixel spacing after np.rot90 by k: odd k swaps rows and columns."""
    return tuple(spacing[::-1]) if k % 2 else tuple(spacing)


def describe_orientation(k, flip):
    """Human-readable description of the correction applied by `reorient`."""
    parts = []
    if k % 4:
        parts.append(
            {
                1: "rotated 90° anticlockwise",
                2: "rotated 180°",
                3: "rotated 90° clockwise",
            }[k % 4]
        )
    if flip:
        parts.append("mirrored left-right")
    return " and ".join(parts) if parts else "no correction"


def _dark_pixels(smooth, mask, spacing, inner_radius=85.0, sigma_mm=15.0):
    """Pixels of the phantom interior that are dark relative to the *local* water signal.

    Head and surface coils can give strong shading, with the water signal
    varying by a factor of several across the phantom. A single threshold
    (half the brightest water) then marks dim water as "dark" and the
    orientation cues are lost. Instead the water level is estimated locally,
    by normalised Gaussian smoothing over the water pixels only (so that the
    dark structures themselves don't pull it down), and a pixel is dark when
    it is below half of that local level.

    Returns:
        tuple: (dark, inner, y, x) - boolean dark mask, boolean mask of the
        phantom interior within `inner_radius` mm of its centre, and the
        physical y / x coordinates (mm, from the centre) of every pixel.
    """
    row_sp, col_sp = spacing
    cy, cx = ndi.center_of_mass(mask)
    yy, xx = np.indices(mask.shape, dtype=float)
    y = (yy - cy) * row_sp
    x = (xx - cx) * col_sp
    # Stay away from the wall so its partial volume doesn't count as "dark"
    inner = (np.hypot(x, y) < inner_radius) & mask

    sigma = (sigma_mm / row_sp, sigma_mm / col_sp)
    level = np.full(smooth.shape, np.percentile(smooth[inner], 90))
    for _ in range(3):
        water = (inner & (smooth > 0.6 * level)).astype(float)
        num = ndi.gaussian_filter(smooth * water, sigma)
        den = ndi.gaussian_filter(water, sigma)
        level = np.where(den > 1e-3, num / np.maximum(den, 1e-12), level)
    dark = inner & (smooth < 0.5 * level)
    return dark, inner, y, x


def _cue_scores(dark, inner, mask, spacing):
    """Features of ACR slice 1 that are high when the phantom is in the standard view.

    In the standard view slice 1 has: a long dark bar running horizontally
    through the centre; the large resolution insert below the bar; and the
    small ring insert in the upper-left quadrant.

    Returns:
        tuple: (bar, below, ring) scores, each in [-1, 1] and > 0 when standard.
    """
    row_sp, col_sp = spacing
    cy, cx = ndi.center_of_mass(mask)
    yy, xx = np.indices(inner.shape, dtype=float)
    y = (yy - cy) * row_sp
    x = (xx - cx) * col_sp

    def frac(region):
        region = region & inner
        return dark[region].mean() if region.any() else 0.0

    bar = frac((np.abs(y) < 3) & (np.abs(x) < 70)) - frac(
        (np.abs(x) < 3) & (np.abs(y) < 70)
    )
    below = frac((y > 10) & (np.abs(x) < 80)) - frac(
        (y < -10) & (np.abs(x) < 80)
    )
    ring = frac((y < -15) & (x < -15)) - frac((y < -15) & (x > 15))
    return bar, below, ring


def _residual_tilt(smooth, mask, spacing):
    """In-plane tilt of slice 1 away from the nearest multiple of 90 degrees.

    The long dark bar through the centre of slice 1 is found as the line
    through the phantom centre (at any angle) that crosses the most dark
    pixels. Its angle is wrapped to [-45, 45) degrees, so the result is the
    small tilt left over once the bar is horizontal or vertical.

    Returns:
        float: tilt in degrees in physical space (x to the right, y downwards),
        positive when the bar runs down to the right.
    """
    dark, inner, y, x = _dark_pixels(smooth, mask, spacing)
    x, y, dark = x[inner], y[inner], dark[inner]

    def bar_score(angle):
        c, s = np.cos(angle), np.sin(angle)
        band = (np.abs(-x * s + y * c) < 3) & (np.abs(x * c + y * s) < 70)
        return dark[band].mean() if band.any() else 0.0

    # The bar is wider than the sampling band, so the score has a plateau
    # a few degrees wide; take its centre (argmax would pick one end of it
    # and report a tilt of 2-4 degrees on a straight phantom).
    angles = np.deg2rad(np.arange(-90.0, 90.0, 0.25))
    scores = np.array([bar_score(a) for a in angles])
    near_best = angles[scores >= scores.max() - 0.02]
    # mean of axial angles (theta and theta + 180 are the same line)
    angle = 0.5 * np.rad2deg(
        np.arctan2(np.sin(2 * near_best).mean(), np.cos(2 * near_best).mean())
    )
    return float((angle + 45.0) % 90.0 - 45.0)


def _scores_by_orientation(smooth, mask, spacing):
    """Yield ((k, flip), (bar, below, ring)) for each 90-degree candidate."""
    # Dark pixels don't depend on the candidate orientation: find them once
    dark, inner, _, _ = _dark_pixels(smooth, mask, spacing)
    for k in range(4):
        for flip in (False, True):
            yield (
                (k, flip),
                _cue_scores(
                    reorient(dark, k, flip),
                    reorient(inner, k, flip),
                    reorient(mask, k, flip),
                    reorient_spacing(spacing, k),
                ),
            )


def determine_phantom_orientation(img, spacing):
    """Find the 90-degree rotation (and mirroring) that puts slice 1 of the ACR
    phantom into its standard view.

    A small in-plane tilt of the phantom (up to 45 degrees) is measured first
    and removed before the 90-degree candidates are scored; otherwise the
    features, which are sampled in narrow bands, are missed when the phantom
    is not square to the image axes.

    Args:
        img (np.ndarray): pixel array of slice 1.
        spacing (tuple): (row_spacing, column_spacing) in mm.

    Returns:
        tuple: (k, flip) to pass to `reorient`.
    """
    smooth, mask = segment_phantom(img, log_scale=True)
    tilt = _residual_tilt(smooth, mask, spacing)
    if abs(tilt) > MIN_ORIENTATION_TILT_DEG:
        # Untilt on an isotropic grid, so that the angle is the physical angle
        iso = min(spacing)
        zoom = (spacing[0] / iso, spacing[1] / iso)
        smooth = ndi.rotate(ndi.zoom(smooth, zoom, order=1), tilt, order=1)
        mask = (
            ndi.rotate(
                ndi.zoom(mask.astype(float), zoom, order=1), tilt, order=1
            )
            > 0.5
        )
        spacing = (iso, iso)
    best, best_key, best_feats = (0, False), None, None
    for (k, flip), feats in _scores_by_orientation(smooth, mask, spacing):
        # Decide hierarchically: bar direction (strongest cue) first, then
        # which side the resolution insert is on, then mirroring. Prefer
        # no mirroring when the ring cue is ambiguous.
        key = tuple(f > 0 for f in feats) + (not flip, sum(feats))
        if best_key is None or key > best_key:
            best, best_key, best_feats = (k, flip), key, feats
    bar, below, _ = best_feats
    if bar < MIN_BAR_SCORE or below < MIN_INSERT_SCORE:
        logger.warning(
            "Phantom orientation could not be determined reliably from slice 1"
            " (bar score %.2f, insert score %.2f); no orientation correction"
            " applied. Check that the series contains ACR slice 1.",
            bar,
            below,
        )
        return (0, False)
    return best


def is_slice_1(img, spacing):
    """Whether an image shows ACR slice 1: in some 90-degree orientation, a
    long dark bar through the centre with the resolution insert below it.

    Args:
        img (np.ndarray): pixel array.
        spacing (tuple): (row_spacing, column_spacing) in mm.
    """
    smooth, mask = segment_phantom(img, log_scale=True)
    return any(
        bar >= MIN_BAR_SCORE and below >= MIN_INSERT_SCORE
        for _, (bar, below, _) in _scores_by_orientation(smooth, mask, spacing)
    )
