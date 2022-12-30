import os
import unittest
import pathlib
import pydicom
import numpy as np

from hazenlib.tasks.acr_uniformity import ACRUniformity
from tests import TEST_DATA_DIR, TEST_REPORT_DIR


class TestACRUniformitySiemens(unittest.TestCase):
    ACR_UNIFORMITY_DATA = pathlib.Path(TEST_DATA_DIR / 'acr')
    centre = [129, 128]
    piu = 68.66
    array = np.zeros((256, 256), dtype=int)
    array[127][126] = 1
    array[126][127] = 1
    array[127][127] = 1
    array[128][127] = 1
    array[127][128] = 1

    def setUp(self):
        self.acr_uniformity_task = ACRUniformity(data_paths=[os.path.join(TEST_DATA_DIR, 'acr')],
                                                 report_dir=pathlib.PurePath.joinpath(TEST_REPORT_DIR))
        self.dcm = pydicom.read_file(os.path.join(TEST_DATA_DIR, 'acr', 'Siemens', '6.dcm'))

    def test_object_centre(self):
        assert self.acr_uniformity_task.centroid_com(self.dcm.pixel_array) == self.centre

    def test_circular_mask(self):
        test_circle = self.acr_uniformity_task.circular_mask([128, 128], 1, [256, 256]).astype(int)
        assert (self.array == test_circle).all() == True

    def test_uniformity(self):
        results = self.acr_uniformity_task.get_integral_uniformity(self.dcm)
        assert round(results, 2) == self.piu


# class TestACRUniformityPhilips(unittest.TestCase):
#
class TestACRUniformityGE(unittest.TestCase):
    ACR_UNIFORMITY_DATA = pathlib.Path(TEST_DATA_DIR / 'acr')
    centre = [253, 256]
    piu = 84.76

    def setUp(self):
        self.acr_uniformity_task = ACRUniformity(data_paths=[os.path.join(TEST_DATA_DIR, 'acr')],
                                                 report_dir=pathlib.PurePath.joinpath(TEST_REPORT_DIR))
        self.dcm = pydicom.read_file(os.path.join(TEST_DATA_DIR, 'acr', 'GE', '4.dcm'))

    def test_object_centre(self):
        assert self.acr_uniformity_task.centroid_com(self.dcm.pixel_array) == self.centre

    def test_uniformity(self):
        results = self.acr_uniformity_task.get_integral_uniformity(self.dcm)
        assert round(results, 2) == self.piu
