"""Test for the ACR low contrast object detectability (LCOD) task."""

# ruff: noqa: PT009 SLF001

from __future__ import annotations

# Python imporst
import unittest
from dataclasses import dataclass
from pathlib import Path

# Module imports
import numpy as np

# Local imports
from hazenlib.tasks.acr_low_contrast_object_detectability import (
    ACRLowContrastObjectDetectability,
    LCODTemplate,
    LowContrastObject,
    Spoke,
)
from hazenlib.utils import get_dicom_files
from tests import TEST_DATA_DIR, TEST_REPORT_DIR

class DummyDICOM:
    """A minimal stand-in for a pydicom FileDataset."""

    def __init__(
        self,
        shape: tuple[int] = (64, 64),
        pixel_spacing: tuple[float] = (1.0, 1.0),
    ) -> None:
        """Initialise the DummyDICOM."""
        # content not used in LCODTemplate.mask()
        self.pixel_array = np.ones(shape, dtype=np.int16)
        self.PixelSpacing = pixel_spacing

@dataclass
class SliceScore:
    """Dataclass for the slice scores."""

    index: int
    score: int


class TestLowContrastObjects(unittest.TestCase):
    """Test the for the low contrast objects."""

    def test_spoke_initialises_objects(self) -> None:
        """Spoke. should create three LowContrastObject instances."""
        cx, cy, theta = 10.0, 20.0, np.pi / 4   # 45
        diameter = 4.0
        spoke = Spoke(cx, cy, theta, diameter)

        self.assertEqual(len(spoke), 3) # three distances defined
        self.assertTrue(all(isinstance(o, LowContrastObject) for o in spoke))

        # Verify coordinates follow the expected polar conversion
        for d, obj in zip(spoke.dist, spoke):
            expected_x = cx + d * np.cos(theta)
            expected_y = cy + d * np.sin(theta)
            self.assertAlmostEqual(obj.x, expected_x)
            self.assertAlmostEqual(obj.y, expected_y)
            self.assertEqual(obj.diameter, diameter)

    def test_spoke_iter_and_len(self) -> None:
        """Spoke should be iterable and report its length correctly."""
        spoke = Spoke(0, 0, 0, 2.0)
        xs = [obj.x for obj in spoke]
        self.assertEqual(len(spoke), len(xs))
        self.assertEqual(list(spoke), list(spoke.objects))


class TestLCODTemplateSpokes(unittest.TestCase):
    """Test the LCODTemplate."""

    def setUp(self) -> None:
        """Create the LCODTemplate object."""
        self.template = LCODTemplate(cx=30.0, cy=40.0, theta=0.0)

    def test_spokes_property_is_cached(self) -> None:
        """The cached_property should return the same tuple on repeated access."""
        first = self.template.spokes
        second = self.template.spokes
        self.assertIs(first, second)   # cached_property returns same object

    def test_spokes_geometry(self) -> None:
        """Each Spoke should have the correct angle and diameter."""
        spokes = self.template.spokes
        self.assertEqual(len(spokes), len(self.template.radi))

        for i, (spoke, radius) in enumerate(zip(spokes, self.template.radi)):
            # Expected angle: theta + i * (360 / N)
            expected_theta = self.template.theta + i * (360 / len(self.template.radi))
            self.assertAlmostEqual(spoke.theta, expected_theta)

            # Diameter stored in Spoke is twice the radius
            self.assertAlmostEqual(spoke.diameter, 2 * radius)

    def test_calc_spokes_is_lru_cached(self) -> None:
        """Calling _calc_spokes with the same arguments should hit the LRU cache."""
        # Populate the cache on first call
        self.template._calc_spokes(
            self.template.cx,
            self.template.cy,
            self.template.theta,
            self.template.radi,
        )
        # Access the internal cache dict indirectly via the function's cache_info()
        before = self.template._calc_spokes.cache_info().hits

        self.template._calc_spokes(
            self.template.cx,
            self.template.cy,
            self.template.theta,
            self.template.radi,
        )
        after = self.template._calc_spokes.cache_info().hits

        # The first call populates the cache, second call should be a hit
        self.assertEqual(after, before + 1)


class TestLCODTemplateMask(unittest.TestCase):
    """Test the LCODTemplate mask method."""

    def setUp(self) -> None:
        """Create the DummyDICOM and LCODTemplate."""
        # Small synthetic DICOM: 20x20 pixels, spacing 1 mm per pixel
        self.dcm = DummyDICOM(shape=(128, 128), pixel_spacing=(1.0, 1.0))
        self.template = LCODTemplate(cx=64.0, cy=64.0, theta=0.0)

    def test_mask_shape_and_dtype(self) -> None:
        """Mask should have the same shape as the DICOM pixel array and be bool."""
        mask = self.template.mask(self.dcm)
        self.assertEqual(mask.shape, self.dcm.pixel_array.shape)
        self.assertEqual(mask.dtype, np.bool)

    def test_mask_object_pixels(self) -> None:
        """Pixels that fall inside any LowContrastObject should be set to 0."""
        mask = self.template.mask(self.dcm)

        # Re-create the coordinate grids used by the mask routine
        y_grid, x_grid = np.meshgrid(*[
            np.linspace(0.0, s * dv, num=s, endpoint=False)
            for s, dv in zip(self.dcm.pixel_array.shape, self.dcm.PixelSpacing)
        ])

        # For each object, confirm that the mask is zero at the centre point
        # (the centre is guaranteed to lie on a pixel because we chose spacing=1)
        spoke_in_image = False
        for spoke in self.template.spokes:
            for obj in spoke:
                # Find the nearest pixel index
                y_idx = int(round(obj.y * self.dcm.PixelSpacing[0]))
                x_idx = int(round(obj.x * self.dcm.PixelSpacing[1]))
                if 0 <= y_idx < mask.shape[0] and 0 <= x_idx < mask.shape[1]:
                    self.assertEqual(
                        mask[y_idx, x_idx], True,
                        msg=f"Pixel at ({y_idx},{x_idx}) not masked",
                    )
                    spoke_in_image = True

        # Assert that a spoke was in the image
        self.assertTrue(spoke_in_image, "No test object was in the image...?!")

        # Additionally, ensure that at least one pixel remains unmasked (value 1)
        self.assertTrue(
            np.any(mask == 0),
            "All pixels were masked unexpected for this geometry",
        )

    def test_mask_respects_offset(self) -> None:
        """The optional offset should shift the coordinate grid before masking."""
        mask = self.template.mask(self.dcm)

        # Diagonal shift
        for v in range(3):

            # Shift output diagonally
            p = 2 ** v
            offset = tuple(ps * p for ps in self.dcm.PixelSpacing)
            mask_shifted = self.template.mask(self.dcm, offset=offset)
            self.assertTrue(np.any(mask != mask_shifted))
            self.assertTrue(np.all(mask[p:, p:] == mask_shifted[:-p, :-p]))


class TestFindSpokes(unittest.TestCase):
    """Validate that ``find_spokes`` returns a correct set of spokes."""

    def setUp(self) -> None:
        """Set up the fake data and find spokes."""
        dim = 256
        self.dcm = DummyDICOM(shape=(dim, dim))

        # Sets up a DummyDICOM image with spokes.
        self.cx = dim // 2 - 12
        self.cy = dim // 2 + 12
        self.theta = 12

        self.template = LCODTemplate(self.cx, self.cy, self.theta)
        self.dcm.pixel_array = self.template.mask(self.dcm) * self.dcm.pixel_array


        # By pass constructor - only need the attributes used by find_spokes
        self.task = ACRLowContrastObjectDetectability.__new__(
            ACRLowContrastObjectDetectability,
        )
        self.task.rotation = 0.0
        self.task.find_center = lambda _: (dim / 2, dim / 2)

    def test_find_spokes(self) -> None:
        """Test spoke finding algorithm."""
        spokes = self.task.find_spokes(self.dcm)

        for spoke, spoke_true in zip(spokes, self.template.spokes):
            for obj, obj_true in zip(spoke, spoke_true):
                self.assertAlmostEqual(obj.x, obj_true.x, places=0)
                self.assertAlmostEqual(obj.y, obj_true.y, places=0)
            self.assertAlmostEqual(spoke.cx, spoke_true.cx, places=0)
            self.assertAlmostEqual(spoke.cy, spoke_true.cy, places=0)
            self.assertAlmostEqual(spoke.theta, spoke_true.theta, places=1)


class TestACRLowContrastObjectDetectability(unittest.TestCase):
    """Test Class for the LCOD task.

    Defaults to testing the slice scores.
    """

    ACR_DATA = Path(TEST_DATA_DIR / "acr" / "Siemens")
    SCORES = (
        SliceScore(8, 9),
        SliceScore(9, 10),
        SliceScore(10, 10),
        SliceScore(11, 10),
    )

    def setUp(self) -> None:
        """Set up for the tests."""
        input_files = get_dicom_files(self.ACR_DATA)
        self.acr_object_detectability = ACRLowContrastObjectDetectability(
            input_data=input_files,
            report_dir=Path(TEST_REPORT_DIR),
            report=True,
        )
        self.results = self.acr_object_detectability.run()

    def test_slice_score(self) -> None:
        """Test the score for a slice."""
        for score in self.SCORES:
            slice_score = self.results.get_measurement(
                name="LowContrastObjectDetectability",
                measurement_type="measured",
                subtype=f"slice {score.index}",
            )[0].value
            self.assertEqual(slice_score, score.score)


    def test_total_score(self) -> None:
        """Test the total score."""
        total_score = self.results.get_measurement(
            name="LowContrastObjectDetectability",
            measurement_type="measured",
            subtype="total",
        )[0].value
        correct_total_score = sum(s.score for s in self.SCORES)
        self.assertEqual(total_score, correct_total_score)


class TestACRLowContrastObjectDetectabilitySiemens2(
        TestACRLowContrastObjectDetectability,
):
    """Test case for more Siemens data."""

    ACR_DATA = Path(TEST_DATA_DIR / "acr" / "Siemens2")
    SCORES = (
        SliceScore(8, 1),
        SliceScore(9, 7),
        SliceScore(10, 9),
        SliceScore(11, 10),
    )


class TestACRLowContrastObjectDetectabilitySiemensMTF(
        TestACRLowContrastObjectDetectability,
):
    """Test case for more Siemens data."""

    ACR_DATA = Path(TEST_DATA_DIR / "acr" / "SiemensMTF")
    SCORES = (
        SliceScore(8, 7),
        SliceScore(9, 10),
        SliceScore(10, 10),
        SliceScore(11, 10),
    )


class TestACRLowContrastObjectDetectabilityGE(
        TestACRLowContrastObjectDetectability,
):
    """Test class for GE data."""

    ACR_DATA = Path(TEST_DATA_DIR / "acr" / "GE")
    SCORES = (
        SliceScore(8, 0),
        SliceScore(9, 2),
        SliceScore(10, 8),
        SliceScore(11, 7),
    )


if __name__ == "__main__":
    unittest.main(failfast=True)
