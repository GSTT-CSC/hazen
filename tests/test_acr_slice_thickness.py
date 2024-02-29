import os
import unittest
import pathlib
import pydicom

from hazenlib.utils import get_dicom_files
from hazenlib.tasks.acr_slice_thickness import ACRSliceThickness
from hazenlib.ACRObject import ACRObject
from tests import TEST_DATA_DIR


class TestACRSliceThicknessSiemens(unittest.TestCase):
    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "Siemens")
    x_pts = [71, 181]
    y_pts = [132, 126]
    dz = 4.91

    def setUp(self):
        input_files = get_dicom_files(self.ACR_DATA)

        self.acr_slice_thickness_task = ACRSliceThickness(input_data=input_files)

        self.dcm = self.acr_slice_thickness_task.ACR_obj.slice_stack[0]
        self.centre, _ = self.acr_slice_thickness_task.ACR_obj.find_phantom_center(
            self.dcm.pixel_array, self.dcm.PixelSpacing[0], self.dcm.PixelSpacing[1]
        )

    def test_ramp_find(self):
        x_pts, y_pts = self.acr_slice_thickness_task.find_ramps(
            self.dcm.pixel_array, self.centre
        )
        assert (x_pts == self.x_pts).all() == True

        assert (y_pts == self.y_pts).all() == True

    def test_slice_thickness(self):
        slice_thickness_val = round(
            self.acr_slice_thickness_task.get_slice_thickness(self.dcm), 2
        )

        print("\ntest_slice_thickness.py::TestSliceThickness::test_slice_thickness")
        print("new_release_value:", slice_thickness_val)
        print("fixed_value:", self.dz)

        assert slice_thickness_val == self.dz


class TestACRSliceThicknessGE(TestACRSliceThicknessSiemens):
    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "GE")
    x_pts = [146, 357]
    y_pts = [262, 250]
    dz = 5.02
