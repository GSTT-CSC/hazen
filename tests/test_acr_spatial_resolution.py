import os
import unittest
import pathlib
import pydicom
import numpy as np

from hazenlib.tasks.acr_spatial_resolution import ACRSpatialResolution
from tests import TEST_DATA_DIR, TEST_REPORT_DIR


class TestACRSpatialResolutionSiemens(unittest.TestCase):
    ACR_SPATIAL_RESOLUTION_DATA = pathlib.Path(TEST_DATA_DIR / 'acr')
    centre = (128, 124)
    rotation_angle = 9
    y_ramp_pos = 118
    width = 13
    edge_type = 'vertical', 'downward'
    edge_loc = [5, 7]
    slope = -0.165
    MTF50 = (1.16, 1.32)

    def setUp(self):
        self.acr_spatial_resolution_task = ACRSpatialResolution(data_paths=[os.path.join(TEST_DATA_DIR, 'acr')],
                                                                report_dir=pathlib.PurePath.joinpath(TEST_REPORT_DIR))
        self.dcm = pydicom.read_file(os.path.join(TEST_DATA_DIR, 'acr', 'SiemensMTF', '0.dcm'))
        self.crop_image = self.acr_spatial_resolution_task.crop_image(self.dcm.pixel_array, self.centre[0],
                                                                      self.y_ramp_pos, self.width)

    def test_object_centre(self):
        data = self.dcm.pixel_array
        assert self.acr_spatial_resolution_task.centroid_com(data)[1] == self.centre

    def test_rotation(self):
        data = self.dcm.pixel_array
        assert self.acr_spatial_resolution_task.find_rotation(data) == self.rotation_angle

    def test_find_y_ramp(self):
        data = self.dcm.pixel_array
        res = self.dcm.PixelSpacing
        assert self.acr_spatial_resolution_task.y_position_for_ramp(res, data, self.centre) == self.y_ramp_pos

    def test_get_edge_type(self):
        assert self.acr_spatial_resolution_task.get_edge_type(self.crop_image) == self.edge_type

    def test_get_edge_loc(self):
        assert (self.acr_spatial_resolution_task.edge_location_for_plot(self.crop_image, self.edge_type[0] ==
                                                                        self.edge_loc)).all()

    def test_retrieve_slope(self):
        assert np.round(self.acr_spatial_resolution_task.fit_normcdf_surface(self.crop_image, self.edge_type[0],
                                                                             self.edge_type[1])[0], 3) == self.slope

    def test_get_MTF50(self):
        mtf50_val = self.acr_spatial_resolution_task.get_mtf50(self.dcm)

        print("\ntest_get_MTF50.py::TestGetMTF50::test_get_MTF50")
        print("new_release_value:", mtf50_val)
        print("fixed_value:", self.MTF50)

        assert mtf50_val == self.MTF50

class TestACRSpatialResolutionGE(unittest.TestCase):
    ACR_SPATIAL_RESOLUTION_DATA = pathlib.Path(TEST_DATA_DIR / 'acr')
    centre = (254, 255)
    rotation_angle = 0
    y_ramp_pos = 244
    width = 26
    edge_type = 'vertical', 'upward'
    edge_loc = [5, 7]
    slope = 0.037
    MTF50 = (0.71, 0.69)

    def setUp(self):
        self.acr_spatial_resolution_task = ACRSpatialResolution(data_paths=[os.path.join(TEST_DATA_DIR, 'acr')],
                                                                report_dir=pathlib.PurePath.joinpath(TEST_REPORT_DIR))
        self.dcm = pydicom.read_file(os.path.join(TEST_DATA_DIR, 'acr', 'GE', '10.dcm'))
        self.crop_image = self.acr_spatial_resolution_task.crop_image(self.dcm.pixel_array, self.centre[0],
                                                                      self.y_ramp_pos, self.width)

    def test_object_centre(self):
        data = self.dcm.pixel_array
        assert self.acr_spatial_resolution_task.centroid_com(data)[1] == self.centre

    def test_rotation(self):
        data = self.dcm.pixel_array
        assert self.acr_spatial_resolution_task.find_rotation(data) == self.rotation_angle

    def test_find_y_ramp(self):
        data = self.dcm.pixel_array
        res = self.dcm.PixelSpacing
        assert self.acr_spatial_resolution_task.y_position_for_ramp(res, data, self.centre) == self.y_ramp_pos

    def test_get_edge_type(self):
        assert self.acr_spatial_resolution_task.get_edge_type(self.crop_image) == self.edge_type

    def test_get_edge_loc(self):
        assert (self.acr_spatial_resolution_task.edge_location_for_plot(self.crop_image, self.edge_type[0] ==
                                                                        self.edge_loc)).all()

    def test_retrieve_slope(self):
        assert np.round(self.acr_spatial_resolution_task.fit_normcdf_surface(self.crop_image, self.edge_type[0],
                                                                             self.edge_type[1])[0], 3) == self.slope

    def test_get_MTF50(self):
        assert self.acr_spatial_resolution_task.get_mtf50(self.dcm) == self.MTF50
