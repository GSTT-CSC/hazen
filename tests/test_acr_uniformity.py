# Python imports
import unittest
import pathlib

# Module imports
import numpy as np

# Local imports
from hazenlib.utils import (
    get_dicom_files,
    create_circular_mask,
    create_circular_roi_at,
)
from hazenlib.tasks.acr_uniformity import ACRUniformity
from tests import TEST_DATA_DIR, TEST_REPORT_DIR


class TestACRUniformitySiemens(unittest.TestCase):
    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "Siemens")
    piu = 67.49

    def setUp(self):
        input_files = get_dicom_files(self.ACR_DATA)

        self.acr_uniformity_task = ACRUniformity(
            input_data=input_files,
            report_dir=pathlib.PurePath.joinpath(TEST_REPORT_DIR),
        )

    def test_roi_values_within_roi(self) -> None:
        """Test the ROI values actually lie within the ROI."""
        img = self.acr_uniformity_task.ACR_obj.slice_stack[6].pixel_array
        (
            (center_x, center_y),
            _,
        )= self.acr_uniformity_task.ACR_obj.find_phantom_center(
            img,
            self.acr_uniformity_task.ACR_obj.dx,
            self.acr_uniformity_task.ACR_obj.dy,
        )

        large_roi = create_circular_roi_at(
            img,
            self.acr_uniformity_task.r_large,
            center_x,
            center_y,
        )

        large_roi_valid_space = create_circular_mask(
            img,
            self.acr_uniformity_task.r_large_filter,
            center_x,
            center_y,
        )

        (
            x_min,
            y_min,
            _,
            x_max,
            y_max,
            _,
        ) = self.acr_uniformity_task.get_mean_roi_values(
            large_roi,
            ~large_roi_valid_space,
        )

        for (x, y) in ([x_min, y_min], [x_max, y_max]):
            dist_from_center = (
                np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                + self.acr_uniformity_task.r_small
            )
            self.assertLessEqual(
                dist_from_center,
                self.acr_uniformity_task.r_large,
            )

    def test_uniformity(self):
        results = self.acr_uniformity_task.get_integral_uniformity(
            self.acr_uniformity_task.ACR_obj.slice_stack[6]
        )
        rounded_results = round(results, 2)

        print("\ntest_uniformity.py::TestUniformity::test_uniformity")
        print("new_release_values:", rounded_results)
        print("fixed_values:", self.piu)

        assert rounded_results == self.piu

class TestACRUniformityGESignaArtistT1(TestACRUniformitySiemens):
    ACR_DATA = pathlib.Path(TEST_DATA_DIR) / "acr" / "GE_Signa_Artist_1.5T_T1"
    piu = 92.03


class TestACRUniformitySiemensSolaFit(TestACRUniformitySiemens):
    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "SiemensSolaFit")
    piu = 95.77


class TestACRUniformityPhilipsAchieva(TestACRUniformitySiemens):
    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "PhilipsAchieva")
    piu = 80.66


class TestACRUniformityGE(TestACRUniformitySiemens):
    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "GE")
    piu = 83.78
