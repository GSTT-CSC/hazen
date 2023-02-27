import os
import unittest
import pathlib
import pydicom

from hazenlib.tasks.acr_slice_position import ACRSlicePosition
from tests import TEST_DATA_DIR


class TestACRSlicePositionSiemens(unittest.TestCase):
    ACR_SLICE_POSITION_DATA = pathlib.Path(TEST_DATA_DIR / 'acr')
    centre = [128, 129]
    x_pts = [(123, 129), (123, 129)]
    y_pts = [(40, 83), (44, 82)]
    dL = -0.59, -1.56

    def setUp(self):
        self.acr_slice_position_task = ACRSlicePosition(data_paths=[os.path.join(TEST_DATA_DIR, 'acr')])
        self.dcm_1 = pydicom.read_file(os.path.join(TEST_DATA_DIR, 'acr', 'Siemens', '0.dcm'))
        self.dcm_11 = pydicom.read_file(os.path.join(TEST_DATA_DIR, 'acr', 'Siemens', '10.dcm'))

    def test_object_centre(self):
        assert self.acr_slice_position_task.centroid_com(self.dcm_1.pixel_array)[1] == self.centre

    def test_wedge_find(self):
        # IMAGE 1
        res = self.dcm_1.PixelSpacing
        mask, _ = self.acr_slice_position_task.centroid_com(self.dcm_1.pixel_array)
        assert (self.acr_slice_position_task.find_wedges(self.dcm_1.pixel_array, mask, res)[0] ==
                self.x_pts[0]).all() == True

        assert (self.acr_slice_position_task.find_wedges(self.dcm_1.pixel_array, mask, res)[1] ==
                self.y_pts[0]).all() == True

        # IMAGE 11
        res = self.dcm_11.PixelSpacing
        mask, _ = self.acr_slice_position_task.centroid_com(self.dcm_11.pixel_array)
        assert (self.acr_slice_position_task.find_wedges(self.dcm_11.pixel_array, mask, res)[0] ==
                self.x_pts[1]).all() == True

        assert (self.acr_slice_position_task.find_wedges(self.dcm_11.pixel_array, mask, res)[1] ==
                self.y_pts[1]).all() == True

    def test_slice_position(self):
        assert round(self.acr_slice_position_task.get_slice_position(self.dcm_1), 2) == self.dL[0]
        assert round(self.acr_slice_position_task.get_slice_position(self.dcm_11), 2) == self.dL[1]


class TestACRSlicePositionGE(unittest.TestCase):
    ACR_SLICE_POSITION_DATA = pathlib.Path(TEST_DATA_DIR / 'acr')
    centre = [253, 257]
    x_pts = [(246, 257), (246, 257)]
    y_pts = [(82, 163), (85, 165)]
    dL = 0.3, 0.41

    def setUp(self):
        self.acr_slice_position_task = ACRSlicePosition(data_paths=[os.path.join(TEST_DATA_DIR, 'acr')])
        self.dcm_1 = pydicom.read_file(os.path.join(TEST_DATA_DIR, 'acr', 'GE', '0.dcm'))
        self.dcm_11 = pydicom.read_file(os.path.join(TEST_DATA_DIR, 'acr', 'GE', '10.dcm'))

    def test_object_centre(self):
        assert self.acr_slice_position_task.centroid_com(self.dcm_1.pixel_array)[1] == self.centre

    def test_wedge_find(self):
        # IMAGE 1
        res = self.dcm_1.PixelSpacing
        mask, _ = self.acr_slice_position_task.centroid_com(self.dcm_1.pixel_array)
        assert (self.acr_slice_position_task.find_wedges(self.dcm_1.pixel_array, mask, res)[0] ==
                self.x_pts[0]).all() == True

        assert (self.acr_slice_position_task.find_wedges(self.dcm_1.pixel_array, mask, res)[1] ==
                self.y_pts[0]).all() == True

        # IMAGE 11
        res = self.dcm_11.PixelSpacing
        mask, _ = self.acr_slice_position_task.centroid_com(self.dcm_11.pixel_array)
        assert (self.acr_slice_position_task.find_wedges(self.dcm_11.pixel_array, mask, res)[0] ==
                self.x_pts[1]).all() == True

        assert (self.acr_slice_position_task.find_wedges(self.dcm_11.pixel_array, mask, res)[1] ==
                self.y_pts[1]).all() == True

    def test_slice_position(self):
        assert round(self.acr_slice_position_task.get_slice_position(self.dcm_1), 2) == self.dL[0]
        assert round(self.acr_slice_position_task.get_slice_position(self.dcm_11), 2) == self.dL[1]