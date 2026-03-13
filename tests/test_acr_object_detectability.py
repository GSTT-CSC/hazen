"""Tests for the ACR Object Detectablity module."""

import pathlib
import unittest

from hazenlib.tasks.acr_object_detectability import ACRObjectDetectability
from hazenlib.utils import get_dicom_files

from tests import TEST_DATA_DIR, TEST_REPORT_DIR


class TestACRObjectDetectability(unittest.TestCase):
    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "Siemens")
    SCORE_8 = 10
    TOTAL_SCORE = 40

    def setUp(self):
        input_files = get_dicom_files(self.ACR_DATA)
        self.acr_object_detectability = ACRObjectDetectability(
            input_data=input_files,
            report_dir=pathlib.PurePath.joinpath(TEST_REPORT_DIR),
        )
        self.results = self.acr_object_detectability.run()

    def test_slice_8_score(self):
        slice_8 = self.results.get_measurement(subtype="slice 8")[0].value
        assert slice_8 == self.SCORE_8

    def test_total_score(self):
        total_score = self.results.get_measurement(subtype="total")[0].value
        assert total_score == self.TOTAL_SCORE


class TestACRObjectDetectabilitySiemensSolaFit(TestACRObjectDetectability):
    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "SiemensSolaFit")
    SCORE_8 = 9
    TOTAL_SCORE = 38


class TestACRObjectDetectabilityGE(TestACRObjectDetectability):
    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "GE")
    SCORE_8 = 0  # in reality, this slice should be scored 0 but this is here to force future reassessments of the algorithm to check this dataset.
    TOTAL_SCORE = 0


class TestACRObjectDetectabilityPhilipsAchieva(TestACRObjectDetectability):
    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "PhilipsAchieva")
    SCORE_8 = 10
    TOTAL_SCORE = 40
