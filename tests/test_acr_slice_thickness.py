"""Testing module for the ACR Slice thickness."""

# ruff: noqa: PT009

# Python imports
import logging
import pathlib
import unittest

# Local imports
from hazenlib.tasks.acr_slice_thickness import ACRSliceThickness
from hazenlib.utils import get_dicom_files
from tests import TEST_DATA_DIR

logger = logging.getLogger(__name__)


class TestACRSliceThicknessSiemens(unittest.TestCase):
    """Base class for ACR Slice Thickness tests."""

    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "Siemens")
    dz = 5.89

    def setUp(self) -> None:
        """Set up each test."""
        input_files = get_dicom_files(self.ACR_DATA)

        self.acr_slice_thickness_task = ACRSliceThickness(
            input_data=input_files,
        )

        self.dcm = self.acr_slice_thickness_task.ACR_obj.slice_stack[0]
        self.results = self.acr_slice_thickness_task.get_slice_thickness(
            self.dcm,
        )

    def test_slice_thickness(self) -> None:
        """Test the slice thickness."""
        slice_thickness_val = round(self.results["thickness"], 2)

        logger.info(
            "test_slice_thickness.py::TestSliceThickness::test_slice_thickness\n"
            "new_release_value: %s\n"
            "fixed_value: %s",
            slice_thickness_val,
            self.dz,
        )
        self.assertAlmostEqual(slice_thickness_val, self.dz, places=2)


class TestACRSliceThicknessPhilipsAchieva(TestACRSliceThicknessSiemens):
    """Tests for the Philips Achieva data."""

    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "PhilipsAchieva")
    dz = 5.36


class TestACRSliceThicknessPhilipsAchieva2(TestACRSliceThicknessSiemens):
    """Tests for the Philips Achieva 2 data."""

    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "PhilipsAchieva2")
    dz = 5.05


class TestACRSliceThicknessSiemensSolaFit(TestACRSliceThicknessSiemens):
    """Tests for the Siemens Sola Fit."""

    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "SiemensSolaFit")
    dz = 5.39


class TestACRSliceThicknessSiemensLargeSliceLocationDelta(
    TestACRSliceThicknessSiemens,
):
    """Tests for the Siemens Large Slice Location Delta."""

    ACR_DATA = pathlib.Path(
        TEST_DATA_DIR / "acr" / "SiemensLargeSliceLocationDelta",
    )
    dz = 4.75


class TestACRSliceThicknessPhilips3TDStream(TestACRSliceThicknessSiemens):
    """Tests for the Philips 3TD Stream data."""

    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "Philips3TdStream")
    dz = 5.12


class TestACRSliceThicknessPhilips3TDStream2(TestACRSliceThicknessSiemens):
    """Tests for the Philips 3TD 2 Stream data."""

    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "Philips3TdStream2")
    dz = 5.49


class TestACRPhilipsSliceThicknessLineProfileLocalMinimaIssue(
    TestACRSliceThicknessSiemens,
):
    """Tests for the Philips Line Profile Local Minima Issue data."""

    ACR_DATA = pathlib.Path(
        TEST_DATA_DIR
        / "acr"
        / "PhilipsSliceThicknessLineProfileLocalMinimaIssue",
    )
    dz = 5.48


class TestACRGESignaSliceThickness(TestACRSliceThicknessSiemens):
    """Tests for the GE Signa data."""

    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "GE_Signa_1.5T_T1")
    dz = 5.65
