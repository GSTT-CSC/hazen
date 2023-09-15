import os
import unittest
import pathlib
import pydicom

from hazenlib.utils import get_dicom_files
from hazenlib.tasks.acr_slice_thickness import ACRSliceThickness
from hazenlib.ACRObject import ACRObject
from tests import TEST_DATA_DIR


class TestACRSliceThicknessSiemens(unittest.TestCase):
    x_pts = [71, 181]
    y_pts = [132, 126]
    dz = 4.91

    def setUp(self):
        ACR_DATA_SIEMENS = pathlib.Path(TEST_DATA_DIR / 'acr' / 'Siemens')
        siemens_files = get_dicom_files(ACR_DATA_SIEMENS)

        self.acr_slice_thickness_task = ACRSliceThickness(input_data=siemens_files)

        self.dcm = self.acr_slice_thickness_task.ACR_obj.dcms[0]

    def test_ramp_find(self):
        res = self.dcm.PixelSpacing
        centre = self.acr_slice_thickness_task.ACR_obj.centre
        assert (self.acr_slice_thickness_task.find_ramps(self.dcm.pixel_array, centre, res)[0] ==
                self.x_pts).all() == True

        assert (self.acr_slice_thickness_task.find_ramps(self.dcm.pixel_array, centre, res)[1] ==
                self.y_pts).all() == True

    def test_slice_thickness(self):
        slice_thickness_val = round(self.acr_slice_thickness_task.get_slice_thickness(self.dcm), 2)

        print("\ntest_slice_thickness.py::TestSliceThickness::test_slice_thickness")
        print("new_release_value:", slice_thickness_val)
        print("fixed_value:", self.dz)

        assert slice_thickness_val == self.dz


class TestACRSliceThicknessGE(TestACRSliceThicknessSiemens):
    x_pts = [146, 356]
    y_pts = [262, 250]
    dz = 5.02

    def setUp(self):
        ACR_DATA_GE = pathlib.Path(TEST_DATA_DIR / 'acr' / 'GE')
        ge_files = get_dicom_files(ACR_DATA_GE)

        self.acr_slice_thickness_task = ACRSliceThickness(input_data=ge_files)

        self.dcm = self.acr_slice_thickness_task.ACR_obj.dcms[0]
