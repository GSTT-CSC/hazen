import os
import unittest
import pathlib
import pydicom

from hazenlib.tasks.acr_slice_thickness import ACRSliceThickness
from tests import TEST_DATA_DIR


class TestACRSliceThicknessSiemens(unittest.TestCase):
    ACR_SLICE_POSITION_DATA = pathlib.Path(TEST_DATA_DIR / 'acr')
    centre = [128, 129]
    x_pts = [71, 181]
    y_pts = [132, 126]
    dz = 4.91

    def setUp(self):
        self.acr_slice_thickness_task = ACRSliceThickness(data_paths=[os.path.join(TEST_DATA_DIR, 'acr')])
        self.dcm = pydicom.read_file(os.path.join(TEST_DATA_DIR, 'acr', 'Siemens', '0.dcm'))

    def test_object_centre(self):
        assert self.acr_slice_thickness_task.centroid_com(self.dcm.pixel_array)[1] == self.centre

    def test_ramp_find(self):
        res = self.dcm.PixelSpacing
        _, centre = self.acr_slice_thickness_task.centroid_com(self.dcm.pixel_array)
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


class TestACRSliceThicknessGE(unittest.TestCase):
    ACR_SLICE_POSITION_DATA = pathlib.Path(TEST_DATA_DIR / 'acr')
    centre = [253, 255]
    x_pts = [146, 356]
    y_pts = [262, 250]
    dz = 5.02

    def setUp(self):
        self.acr_slice_thickness_task = ACRSliceThickness(data_paths=[os.path.join(TEST_DATA_DIR, 'acr')])
        self.dcm = pydicom.read_file(os.path.join(TEST_DATA_DIR, 'acr', 'GE', '10.dcm'))

    def test_object_centre(self):
        assert self.acr_slice_thickness_task.centroid_com(self.dcm.pixel_array)[1] == self.centre

    def test_ramp_find(self):
        res = self.dcm.PixelSpacing
        _, centre = self.acr_slice_thickness_task.centroid_com(self.dcm.pixel_array)
        assert (self.acr_slice_thickness_task.find_ramps(self.dcm.pixel_array, centre, res)[0] ==
                self.x_pts).all() == True

        assert (self.acr_slice_thickness_task.find_ramps(self.dcm.pixel_array, centre, res)[1] ==
                self.y_pts).all() == True

    def test_slice_thickness(self):
        assert round(self.acr_slice_thickness_task.get_slice_thickness(self.dcm), 2) == self.dz