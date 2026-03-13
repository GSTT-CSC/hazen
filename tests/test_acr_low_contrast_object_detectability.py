"""Test for the ACR low contrast object detectability (LCOD) task."""

# ruff: noqa: PT009 SLF001

from __future__ import annotations

# Python imports
import os
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Module imports
import numpy as np
from hazenlib.tasks.acr_low_contrast_object_detectability import (
    ACRLowContrastObjectDetectability,
)
from hazenlib.types import LCODTemplate, LowContrastObject, Spoke
from hazenlib.utils import get_dicom_files

# Local imports
from tests import TEST_DATA_DIR, TEST_REPORT_DIR


@dataclass
class DummyDICOMEntry:
    """A minimal stand-in for a pydicom.FileDataset entry."""

    value: Any


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

        # Needed for getting pixel spacing
        self.Manufacturer = "GE"
        self.SOPClassUID = "1.2.840.10008.5.1.4.1.1.4"  # Not an enhanced DICOM

    def __getitem__(self, key: str | tuple) -> DummyDICOMEntry:
        """Return None."""
        return DummyDICOMEntry(value=None)

    def set_pixel_data(
        self,
        pixel_data: np.ndarray,
        *_: Any,  # noqa: ANN401
    ) -> None:
        """Set the pixel data."""
        self.pixel_array = pixel_data


@dataclass
class SliceScore:
    """Dataclass for the slice scores."""

    index: int
    score: int


class TestLowContrastObjects(unittest.TestCase):
    """Test the for the low contrast objects."""

    def test_spoke_initialises_objects(self) -> None:
        """Spoke. should create three LowContrastObject instances."""
        cx, cy, theta = 10.0, 20.0, np.pi / 4  # 45
        diameter = 4.0
        spoke = Spoke(cx, cy, theta, diameter)

        self.assertEqual(len(spoke), 3)  # three distances defined
        self.assertTrue(all(isinstance(o, LowContrastObject) for o in spoke))

        # Verify coordinates
        for d, obj in zip(spoke.dist, spoke):
            expected_x = cx + d * np.sin(np.deg2rad(theta))
            expected_y = cy - d * np.cos(np.deg2rad(theta))
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
        self.assertIs(first, second)  # cached_property returns same object

    def test_spokes_geometry(self) -> None:
        """Each Spoke should have the correct angle and diameter."""
        spokes = self.template.spokes
        self.assertEqual(len(spokes), len(self.template.diameters))

        for i, (spoke, diameter) in enumerate(
            zip(spokes, self.template.diameters)
        ):
            # Expected angle: theta + i * (360 / N)
            expected_theta = self.template.theta + i * (
                360 / len(self.template.diameters)
            )
            self.assertAlmostEqual(spoke.theta, expected_theta)

            # Diameter stored in Spoke is twice the radius
            self.assertAlmostEqual(spoke.diameter, diameter)

    def test_calc_spokes_is_lru_cached(self) -> None:
        """Calling _calc_spokes with the same arguments should hit the LRU cache."""
        # Populate the cache on first call
        self.template._calc_spokes(
            self.template.cx,
            self.template.cy,
            self.template.theta,
            self.template.diameters,
        )
        # Access the internal cache dict indirectly via the function's cache_info()
        before = self.template._calc_spokes.cache_info().hits

        self.template._calc_spokes(
            self.template.cx,
            self.template.cy,
            self.template.theta,
            self.template.diameters,
        )
        after = self.template._calc_spokes.cache_info().hits

        # The first call populates the cache, second call should be a hit
        self.assertEqual(after, before + 1)

    def test_spoke_profile_extraction(self) -> None:
        """Spoke.profile should return the correct radial intensity profile.

        The test builds a synthetic DICOM where the pixel value equals the
        row index (i.e. intensity = y-coordinate).  With a spoke pointing
        straight up the profile samples a vertical line through the
        centre, therefore the returned values must match the y-coordinates
        used for the sampling.
        """
        shape = (128, 128)  # rows, cols
        pixel_spacing = (1.0, 1.0)  # square pixels
        dcm = DummyDICOM(shape=shape, pixel_spacing=pixel_spacing)

        for cy, cx in ([64, 64], [68, 64], [60, 68]):
            y_grid, x_grid = np.meshgrid(
                *[np.linspace(0, s, s) for s in shape]
            )

            vals = np.sqrt((y_grid - cy) ** 2 + (x_grid - cx) ** 2)
            dcm.pixel_array = vals

            theta = 0.0  # centre in the middle of the array
            diameter = cy + 1

            spoke = Spoke(cx, cy, theta, diameter)
            spoke.length = diameter

            size = int(diameter)
            profile = spoke.profile(dcm, size=size)

            expected_v = dcm.pixel_array[cy::-1, cx]

            self.assertEqual(profile.shape, (size,))
            np.testing.assert_allclose(
                profile, expected_v, rtol=1e-6, atol=1e-6
            )


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
        y_grid, x_grid = np.meshgrid(
            *[
                np.linspace(0.0, s * dv, num=s, endpoint=False)
                for s, dv in zip(
                    self.dcm.pixel_array.shape, self.dcm.PixelSpacing
                )
            ]
        )

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
                        mask[y_idx, x_idx],
                        True,
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
            p = 2**v
            offset = tuple(ps * p for ps in self.dcm.PixelSpacing)
            mask_shifted = self.template.mask(self.dcm, offset=offset)
            self.assertTrue(np.any(mask != mask_shifted))
            self.assertTrue(np.all(mask[p:, p:] == mask_shifted[:-p, :-p]))


class TestACRLCODTemplateFinding(unittest.TestCase):
    """Test for the centre finding for the LCOD test."""

    ACR_DATA = Path(TEST_DATA_DIR / "acr" / "GE_Artist_1.5T_T1")

    def setUp(self) -> None:
        """Set up for the tests."""
        input_files = get_dicom_files(self.ACR_DATA)

        report_env = os.getenv("HAZEN_REPORT", "false").lower()
        report = report_env in ("true", "1", "yes")
        self.acr_object_detectability = ACRLowContrastObjectDetectability(
            input_data=input_files,
            report_dir=Path(TEST_REPORT_DIR),
            report=report,
        )

    def test_get_current_slice_template(self) -> None:
        """Test that get_current_slice_template returns a template for a valid slice."""
        result = self.acr_object_detectability.get_current_slice_template(11)
        self.assertTrue(result)


class TestACRLowContrastObjectDetectability(unittest.TestCase):
    """Test Class for the LCOD task.

    Defaults to testing the slice scores.
    """

    ACR_DATA = Path(TEST_DATA_DIR / "acr" / "GE_Artist_1.5T_T1")
    SCORES = (
        SliceScore(8, 10),
        SliceScore(9, 10),
        SliceScore(10, 10),
        SliceScore(11, 10),
    )
    SLICE_TOLERANCE: int = 1  # Accepted +- slice tolerance
    TOTAL_TOLERANCE: int = 2  # Accepted +- total tolerance

    def setUp(self) -> None:
        """Set up for the tests."""
        input_files = get_dicom_files(self.ACR_DATA)

        # Get report flag from environment (default: False)
        # To enable reports, you would run something like:

        ###########
        ## Linux ##
        ###########

        # HAZEN_REPORT=true pytest
        # or:
        # HAZEN_REPORT=1 python -m unittest

        ###################
        ## Windows (cmd) ##
        ###################

        # set HAZEN_REPORT=true && pytest
        # or:
        # set HAZEN_REPORT=1 && python -m unittestac

        ##########################
        ## Windows (powershell) ##
        ##########################

        # $env:HAZEN_REPORT="true"; pytest
        # or:
        # $env:HAZEN_REPORT="1"; python -m unittest
        report_env = os.getenv("HAZEN_REPORT", "false").lower()
        report = report_env in ("true", "1", "yes")

        self.acr_object_detectability = ACRLowContrastObjectDetectability(
            input_data=input_files,
            report_dir=Path(TEST_REPORT_DIR),
            report=report,
        )
        self.results = self.acr_object_detectability.run()

    def test_result_reproducability(self) -> None:
        """Test the results are the same for each identical run."""
        results = self.acr_object_detectability.run()
        self.assertEqual(results, self.results)

    def _score_testing(self, score: SliceScore) -> None:
        result = self.results.get_measurement(
            name="LowContrastObjectDetectability",
            measurement_type="measured",
            subtype=f"slice {score.index}",
        )
        self.assertEqual(len(result), 1)

        slice_score = result[0].value
        self.assertTrue(
            abs(slice_score - score.score) <= self.SLICE_TOLERANCE,
            msg=(
                f"{self.ACR_DATA} slice {score.index}\n"
                f"Expected {score.score} +- {self.SLICE_TOLERANCE}"
                f" but got {slice_score}"
            ),
        )

    def test_slice_score_8(self) -> None:
        """Test the score for slice 8."""
        self._score_testing(self.SCORES[0])

    def test_slice_score_9(self) -> None:
        """Test the score for slice 9."""
        self._score_testing(self.SCORES[1])

    def test_slice_score_10(self) -> None:
        """Test the score for slice 10."""
        self._score_testing(self.SCORES[2])

    def test_slice_score_11(self) -> None:
        """Test the score for slice 11."""
        self._score_testing(self.SCORES[3])

    def test_total_score(self) -> None:
        """Test the total score."""
        total_score = self.results.get_measurement(
            name="LowContrastObjectDetectability",
            measurement_type="measured",
            subtype="total",
        )[0].value
        correct_total_score = sum(s.score for s in self.SCORES)
        self.assertTrue(
            abs(total_score - correct_total_score) <= self.TOTAL_TOLERANCE,
            msg=(
                f"Expected total score to be {correct_total_score}"
                f" +- {self.TOTAL_TOLERANCE} but got {total_score}"
            ),
        )


class TestACRLowContrastObjectDetectabilitySiemensAera(
    TestACRLowContrastObjectDetectability,
):
    """Test class for Siemens Aera data."""

    ACR_DATA = Path(TEST_DATA_DIR / "acr" / "Siemens_Aera_1.5T_T1")
    SCORES = (
        SliceScore(8, 10),
        SliceScore(9, 10),
        SliceScore(10, 9),
        SliceScore(11, 10),
    )


class TestACRLowContrastObjectDetectabilitySiemensSkyra(
    TestACRLowContrastObjectDetectability,
):
    """Test class for Siemens Skyra data."""

    ACR_DATA = Path(TEST_DATA_DIR / "acr" / "Siemens_MagnetomSkyra_3T_T1")
    SCORES = (
        SliceScore(8, 10),
        SliceScore(9, 10),
        SliceScore(10, 10),
        SliceScore(11, 10),
    )


class TestACRLowContrastObjectDetectabilitySiemensSolaFit(
    TestACRLowContrastObjectDetectability,
):
    """Test class for Siemens Sola Fit."""

    ACR_DATA = Path(TEST_DATA_DIR / "acr" / "SiemensSolaFit")
    SCORES = (
        SliceScore(8, 10),
        SliceScore(9, 8),
        SliceScore(10, 9),
        SliceScore(11, 9),
    )


class TestACRLowContrastObjectDetectabilityPhilipsAchieva(
    TestACRLowContrastObjectDetectability,
):
    """Test class for Philips Achieva data."""

    ACR_DATA = Path(TEST_DATA_DIR / "acr" / "PhilipsAchieva")
    SCORES = (
        SliceScore(8, 10),
        SliceScore(9, 10),
        SliceScore(10, 10),
        SliceScore(11, 10),
    )


if __name__ == "__main__":
    unittest.main(failfast=True)
