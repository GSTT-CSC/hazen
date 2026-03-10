import logging
import pathlib
import unittest

from hazenlib.tasks.acr_spatial_resolution import ACRSpatialResolution
from hazenlib.utils import get_dicom_files

from tests import TEST_DATA_DIR, TEST_REPORT_DIR

# noqa: S101

logger = logging.getLogger(__name__)


class TestACRSpatialResolutionSiemens(unittest.TestCase):
    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "Siemens")
    EXPECTED_RESOLVED_ROWS = [3, 2, -1, 2, 3, 2]
    EXPECTED_UNRESOLVED_ARRAY = [1.0]
    EXPECTED_UL_RESOLUTION = 0.9
    EXPECTED_LR_RESOLUTION = 0.9
    EXPECTED_RESOLUTION = 0.9

    def setUp(self):
        input_files = get_dicom_files(self.ACR_DATA)

        self.acr_spatial_resolution_task = ACRSpatialResolution(
            input_data=input_files,
            report_dir=pathlib.PurePath.joinpath(TEST_REPORT_DIR),
        )

        self.dcm = self.acr_spatial_resolution_task.ACR_obj.slice_stack[0]
        self.detected_rows = (
            self.acr_spatial_resolution_task.get_spatially_resolved_rows(
                self.dcm
            )
        )
        self.resolved_arrays = (
            self.acr_spatial_resolution_task.get_resolved_arrays(
                self.detected_rows
            )
        )
        self.ul_resolution, self.lr_resolution, self.best_resolution = (
            self.acr_spatial_resolution_task.get_best_resolution(
                self.resolved_arrays
            )
        )

    def test_get_best_resolution(self):
        assert self.EXPECTED_UL_RESOLUTION == self.ul_resolution
        assert self.EXPECTED_LR_RESOLUTION == self.lr_resolution
        assert self.EXPECTED_RESOLUTION == self.best_resolution

    def test_unresolved_array(self):
        unresolved_resolutions = [
            k for k, v in self.resolved_arrays.items() if not v["resolved"]
        ]
        assert unresolved_resolutions == self.EXPECTED_UNRESOLVED_ARRAY

    def test_get_spatially_resolved_rows(self):
        assert self.EXPECTED_RESOLVED_ROWS == self.detected_rows


class TestACRSpatialResolutionSiemensSolaFit(TestACRSpatialResolutionSiemens):
    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "SiemensSolaFit")
    EXPECTED_RESOLVED_ROWS = [1, 3, 1, 2, 1, 3]
    EXPECTED_UNRESOLVED_ARRAY = []
    EXPECTED_UL_RESOLUTION = 0.9
    EXPECTED_LR_RESOLUTION = 0.9
    EXPECTED_RESOLUTION = 0.9


class TestACRSpatialResolutionPhilipsAchieva(TestACRSpatialResolutionSiemens):
    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "PhilipsAchieva")
    EXPECTED_RESOLVED_ROWS = [1, 1, 1, 1, -1, -1]
    EXPECTED_UNRESOLVED_ARRAY = [0.9]
    EXPECTED_UL_RESOLUTION = 1.0
    EXPECTED_LR_RESOLUTION = 1.0
    EXPECTED_RESOLUTION = 1.0


class TestACRSpatialResolutionGE(TestACRSpatialResolutionSiemens):
    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "GE")
    EXPECTED_RESOLVED_ROWS = [1, 1, 1, 1, 1, 1]
    EXPECTED_UNRESOLVED_ARRAY = []
    EXPECTED_UL_RESOLUTION = 0.9
    EXPECTED_LR_RESOLUTION = 0.9
    EXPECTED_RESOLUTION = 0.9
