import os
import unittest
import pathlib

from hazenlib.utils import get_dicom_files
from hazenlib.tasks.acr_slice_position import ACRSlicePosition
from tests import TEST_DATA_DIR


class TestACRSlicePositionSiemens(unittest.TestCase):
    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "Siemens")
    slice_1_x_pts = [123, 129]
    slice_11_x_pts = [123, 129]
    slice_1_y_pts = [40, 82]
    slice_11_y_pts = [44, 82]
    dL = -0.59, -1.56
    tolerance = 0.05  # 5% tolerance

    def setUp(self):
        input_files = get_dicom_files(self.ACR_DATA)
        self.acr_slice_position_task = ACRSlicePosition(input_data=input_files)

        self.dcm_1 = self.acr_slice_position_task.ACR_obj.slice_stack[0]
        img_1 = self.dcm_1.pixel_array
        mask_1 = self.acr_slice_position_task.ACR_obj.get_mask_image(img_1)
        self.slice1_x_pts, self.slice1_y_pts = self.acr_slice_position_task.find_wedges(
            img_1, mask_1
        )

        self.dcm_11 = self.acr_slice_position_task.ACR_obj.slice_stack[-1]
        img_11 = self.dcm_11.pixel_array
        mask_11 = self.acr_slice_position_task.ACR_obj.get_mask_image(img_11)
        (
            self.slice11_x_pts,
            self.slice11_y_pts,
        ) = self.acr_slice_position_task.find_wedges(img_11, mask_11)

    # IMAGE 1
    def test_find_wedge_slice1_x(self):
        assert all(abs(a - b) / b <= self.tolerance for a, b in zip(self.slice1_x_pts, self.slice_1_x_pts))

    def test_find_wedge_slice1_y(self):
        assert all(abs(a - b) / b <= self.tolerance for a, b in zip(self.slice1_y_pts, self.slice_1_y_pts))

    # IMAGE 11
    def test_find_wedge_slice11_x(self):
        assert all(abs(a - b) / b <= self.tolerance for a, b in zip(self.slice11_x_pts, self.slice_11_x_pts))

    def test_find_wedge_slice11_y(self):
        assert all(abs(a - b) / b <= self.tolerance for a, b in zip(self.slice11_y_pts, self.slice_11_y_pts))

    def test_slice_position(self):
        slice_position_val_1 = round(
            self.acr_slice_position_task.get_slice_position(self.dcm_1), 2
        )
        slice_position_val_11 = round(
            self.acr_slice_position_task.get_slice_position(self.dcm_11), 2
        )

        print("\ntest_slice_position.py::TestSlicePosition::test_slice_position")
        print("new_release_value:", slice_position_val_1)
        print("fixed_value:", self.dL[0])

        assert abs(slice_position_val_1 - self.dL[0]) / self.dL[0] <= self.tolerance
        assert abs(slice_position_val_11 - self.dL[1]) / self.dL[1] <= self.tolerance


class TestACRSlicePositionGE(TestACRSlicePositionSiemens):
    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "GE")
    slice_1_x_pts = [246, 257]
    slice_11_x_pts = [246, 257]
    slice_1_y_pts = [84, 164]
    slice_11_y_pts = [89, 162]
    dL = 0.41, 0.3
