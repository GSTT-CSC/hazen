"""Tests for the ACR phantom orientation detection."""

import unittest

import numpy as np

from hazenlib.ACRObject import ACRObject
from hazenlib.logger import logger
from hazenlib.orientation import (
    determine_phantom_orientation,
    is_slice_1,
    reorient,
    reorient_spacing,
)
from hazenlib.utils import dcmread, get_dicom_files
from tests import TEST_DATA_DIR


def slice_1(name):
    """Pixels and (row, column) spacing of slice 1 of a test dataset."""
    files = get_dicom_files(TEST_DATA_DIR / "acr" / name)
    acr_obj = ACRObject([dcmread(f) for f in files], orientation=(0, False))
    return acr_obj.slice_stack[0].pixel_array, acr_obj.pixel_spacing


def expected(k, flip):
    """Correction undoing an export rotated by k * 90 degrees (then mirrored)."""
    return ((-k) % 4 if not flip else k, flip)


class TestDetectsEveryOrientation(unittest.TestCase):
    """All eight 90 degree rotations / mirrorings of slice 1 are found, and
    undoing them gives back the standard view."""

    ACR_DATA = "Siemens"

    def test_every_orientation(self):
        img, spacing = slice_1(self.ACR_DATA)
        for k in range(4):
            for flip in (False, True):
                with self.subTest(k=k, flip=flip):
                    exported = reorient(img, k, flip)
                    found = determine_phantom_orientation(
                        exported, reorient_spacing(spacing, k)
                    )
                    assert found == expected(k, flip)
                    np.testing.assert_array_equal(
                        reorient(exported, *found), img
                    )


class TestDetectsEveryOrientationPhilipsAchieva(TestDetectsEveryOrientation):
    ACR_DATA = "PhilipsAchieva"


class TestDetectsEveryOrientationGEMR450W(TestDetectsEveryOrientation):
    ACR_DATA = "GE_MR450W_1.5T_T2"


class TestDetectsEveryOrientationGESigna3T(TestDetectsEveryOrientation):
    ACR_DATA = "GE_Signa_3T_T1"


class TestStrongShading(unittest.TestCase):
    """With surface coils the dimmest water can be far below the Otsu
    threshold of the raw image; the phantom must still be segmented whole."""

    def test_shaded_top_to_bottom(self):
        img, spacing = slice_1("Siemens")
        shaded = img * np.linspace(0.1, 1.0, img.shape[0])[:, None]
        exported = reorient(shaded, 1, True)
        found = determine_phantom_orientation(
            exported, reorient_spacing(spacing, 1)
        )
        assert found == expected(1, True)


class TestSlice1Cues(unittest.TestCase):
    """A water disc with the long dark bar through its centre and/or the
    resolution insert below it."""

    SPACING = (1.0, 1.0)

    @staticmethod
    def disc(bar, insert):
        y, x = np.indices((256, 256)) - 127.5
        img = np.where(np.hypot(x, y) < 95, 1000.0, 0.0)
        if bar:
            img[(np.abs(y) < 4) & (np.abs(x) < 70)] = 50.0
        if insert:
            img[(y > 20) & (y < 60) & (np.abs(x) < 40)] = 50.0
        return img

    def test_is_slice_1(self):
        cases = [
            (True, True, True),
            (False, True, False),  # insert but no bar
            (True, False, False),  # bar but no insert
        ]
        for bar, insert, result in cases:
            with self.subTest(bar=bar, insert=insert):
                assert is_slice_1(self.disc(bar, insert), self.SPACING) is (
                    result
                )

    def test_no_cues_no_correction(self):
        """Without the slice 1 features, no correction is applied."""
        img = np.rot90(self.disc(bar=True, insert=False))
        with self.assertLogs(logger, level="WARNING") as logs:
            found = determine_phantom_orientation(img, self.SPACING)
        assert found == (0, False)
        assert any("could not be determined" in m for m in logs.output)
