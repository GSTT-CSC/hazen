import os
import unittest
import pathlib
import pydicom
import numpy as np

from hazenlib.utils import get_dicom_files
from hazenlib.tasks.acr_spatial_resolution import ACRSpatialResolution
from hazenlib.ACRObject import ACRObject
from tests import TEST_DATA_DIR, TEST_REPORT_DIR


class TestACRSpatialResolutionSiemens(unittest.TestCase):
    centre = (128, 124)
    rotation_angle = 9
    y_ramp_pos = 118
    width = 13
    edge_type = 'vertical', 'downward'
    edge_loc = [5, 7]
    slope = -0.165
    MTF50 = (1.18, 1.35)

    def setUp(self):
        ACR_DATA_SIEMENS = pathlib.Path(TEST_DATA_DIR / 'acr' / 'Siemens')
        siemens_files = get_dicom_files(ACR_DATA_SIEMENS)

        self.acr_spatial_resolution_task = ACRSpatialResolution(
            input_data=siemens_files,
            report_dir=pathlib.PurePath.joinpath(TEST_REPORT_DIR))
        self.acr_spatial_resolution_task.ACR_obj = ACRObject(
            [pydicom.read_file(os.path.join(TEST_DATA_DIR, 'acr', 'SiemensMTF', f'{i}')) for i in
             os.listdir(os.path.join(TEST_DATA_DIR, 'acr', 'SiemensMTF'))])

        self.dcm = self.acr_spatial_resolution_task.ACR_obj.dcm[0]
        self.crop_image = self.acr_spatial_resolution_task.crop_image(self.dcm.pixel_array, self.centre[0],
                                                                      self.y_ramp_pos, self.width)

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
    centre = (254, 255)
    rotation_angle = 0
    y_ramp_pos = 244
    width = 26
    edge_type = 'vertical', 'upward'
    edge_loc = [5, 7]
    slope = 0.037
    MTF50 = (0.72, 0.71)

    def setUp(self):
        ACR_DATA_GE = pathlib.Path(TEST_DATA_DIR / 'acr' / 'GE')
        ge_files = get_dicom_files(ACR_DATA_GE)

        self.acr_spatial_resolution_task = ACRSpatialResolution(
            input_data=ge_files,
            report_dir=pathlib.PurePath.joinpath(TEST_REPORT_DIR))
        self.acr_spatial_resolution_task.ACR_obj = ACRObject(
            [pydicom.read_file(os.path.join(ACR_DATA_GE, f'{i}')) for i in
             os.listdir(ACR_DATA_GE)])

        self.dcm = self.acr_spatial_resolution_task.ACR_obj.dcm[0]
        self.crop_image = self.acr_spatial_resolution_task.crop_image(self.dcm.pixel_array, self.centre[0],
                                                                      self.y_ramp_pos, self.width)

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
