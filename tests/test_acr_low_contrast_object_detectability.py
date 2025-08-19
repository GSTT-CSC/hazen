"""Test for the ACR low contrast object detectability (LCOD) task."""

# ruff: noqa:PT009

# Python imporst
import unittest
from dataclasses import dataclass
from pathlib import Path

# Module imports
from hazenlib.tasks.acr_low_contrast_object_detectability import (
    ACRLowContrastObjectDetectability,
)
from hazenlib.utils import get_dicom_files

# Local imports
from tests import TEST_DATA_DIR, TEST_REPORT_DIR


@dataclass
class SliceScore:
    """Dataclass for the slice scores."""

    index: int
    score: int


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
