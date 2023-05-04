import os
import unittest
import pathlib
import pydicom

from hazenlib.tasks.acr_slice_thickness import ACRSliceThickness
from hazenlib.acr_tools import ACRTools
from tests import TEST_DATA_DIR


class TestACRSliceThicknessSiemens(unittest.TestCase):
    ACR_SLICE_POSITION_DATA = pathlib.Path(TEST_DATA_DIR / 'acr')
    x_pts = [71, 181]
    y_pts = [132, 126]
    dz = 4.91

    def setUp(self):
        self.acr_slice_thickness_task = ACRSliceThickness(data_paths=[os.path.join(TEST_DATA_DIR, 'acr')])
        self.acr_slice_thickness_task.ACR_obj = ACRTools(
            [pydicom.read_file(os.path.join(TEST_DATA_DIR, 'acr', 'Siemens', f'{i}')) for i in
             os.listdir(os.path.join(TEST_DATA_DIR, 'acr', 'Siemens'))])

        self.dcm = self.acr_slice_thickness_task.ACR_obj.dcm[0]

    def test_ramp_find(self):
        res = self.dcm.PixelSpacing
        centre = self.acr_slice_thickness_task.ACR_obj.centre
        assert (self.acr_slice_thickness_task.find_ramps(self.dcm.pixel_array, centre, res)[0] ==
                self.x_pts).all() == True

        assert (self.acr_slice_thickness_task.find_ramps(self.dcm.pixel_array, centre, res)[1] ==
                self.y_pts).all() == True

    def test_slice_thickness(self):
        assert round(self.acr_slice_thickness_task.get_slice_thickness(self.dcm), 2) == self.dz


class TestACRSliceThicknessGE(unittest.TestCase):
    ACR_SLICE_POSITION_DATA = pathlib.Path(TEST_DATA_DIR / 'acr')
    x_pts = [146, 356]
    y_pts = [262, 250]
    dz = 5.02

    def setUp(self):
        self.acr_slice_thickness_task = ACRSliceThickness(data_paths=[os.path.join(TEST_DATA_DIR, 'acr')])
        self.acr_slice_thickness_task.ACR_obj = ACRTools(
            [pydicom.read_file(os.path.join(TEST_DATA_DIR, 'acr', 'GE', f'{i}')) for i in
             os.listdir(os.path.join(TEST_DATA_DIR, 'acr', 'GE'))])

        self.dcm = self.acr_slice_thickness_task.ACR_obj.dcm[0]

    def test_ramp_find(self):
        res = self.dcm.PixelSpacing
        centre = self.acr_slice_thickness_task.ACR_obj.centre
        assert (self.acr_slice_thickness_task.find_ramps(self.dcm.pixel_array, centre, res)[0] ==
                self.x_pts).all() == True
        assert (self.acr_slice_thickness_task.find_ramps(self.dcm.pixel_array, centre, res)[1] ==
                self.y_pts).all() == True

    def test_slice_thickness(self):
        assert round(self.acr_slice_thickness_task.get_slice_thickness(self.dcm), 2) == self.dz