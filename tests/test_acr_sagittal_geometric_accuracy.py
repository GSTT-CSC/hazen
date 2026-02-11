"""Module to test the ACR sagittal localiser."""

# ruff: noqa:PT009

import logging
import pathlib
import unittest

import numpy as np
from hazenlib.tasks.acr_sagittal_geometric_accuracy import (
    ACRSagittalGeometricAccuracy,
)
from hazenlib.utils import get_dicom_files

from tests import TEST_DATA_DIR, TEST_REPORT_DIR

logger = logging.getLogger(__name__)


class TestACRSagittalGeometricAccuracySiemens(unittest.TestCase):
    """Base test class for the Siemens Sagittal Localizer Geometric Accuracy."""

    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "SiemensSolaFitLocalizer")
    L1 = 149.41

    def setUp(self) -> None:
        """Set up the test case."""
        input_files = get_dicom_files(self.ACR_DATA)

        self.acr_geometric_accuracy_task = ACRSagittalGeometricAccuracy(
            input_data=input_files,
            report_dir=pathlib.PurePath.joinpath(TEST_REPORT_DIR),
        )

        self.dcm_1 = self.acr_geometric_accuracy_task.ACR_obj.slice_stack[0]
        self.slice1_val = np.round(
            self.acr_geometric_accuracy_task.get_geometric_accuracy(
                self.dcm_1,
            ),
            2,
        )

    def test_geometric_accuracy_slice_1(self) -> None:
        """Test the geometric accuracy of slice 1."""
        logger.info(
            "new_release: %f\nfixed value: %f",
            self.slice1_val,
            self.L1,
        )
        self.assertAlmostEqual(self.slice1_val, self.L1, 0)


class TestACRSagittalGeometricAccuracyPhilipsAchieva(
    TestACRSagittalGeometricAccuracySiemens,
):
    """Base test class for the Philips Sagittal Localizer Geometric Accuracy."""

    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "PhilipsAchievaLocalizer")
    L1 = 146.48


class TestACRSagittalGeometricAccuracyGESigna(
    TestACRSagittalGeometricAccuracySiemens,
):
    """Base test class for the GE Sagittal Localizer Geometric Accuracy."""

    ACR_DATA = (
        pathlib.Path(TEST_DATA_DIR)
        / "acr"
        / "GE_Signa_1.5T_Sagittal_Localizer"
    )
    L1 = 150.0
