import unittest
import pathlib

import pydicom

import hazenlib.acr_uniformity as hazen_acr_uniformity
import numpy as np
from tests import TEST_DATA_DIR


class TestACRUniformitySiemens(unittest.TestCase):
    ACR_UNIFORMITY_DATA = pathlib.Path(TEST_DATA_DIR / 'acr')
    centre = [128, 129]
    piu = 67.91
    array = np.zeros((256,256), dtype=int)
    array[127][126] = 1
    array[126][127] = 1
    array[127][127] = 1
    array[128][127] = 1
    array[127][128] = 1

    def setUp(self):
        self.dcm_file = pydicom.read_file(str(self.ACR_UNIFORMITY_DATA / 'Siemens' / 'Test' / '6.dcm'))

    def test_object_centre(self):
        data = self.dcm_file.pixel_array
        assert hazen_acr_uniformity.centroid_com(data) == self.centre

    def test_circular_mask(self):
        test_circle = hazen_acr_uniformity.circular_mask([128,128], 1, [256,256]).astype(int)
        assert (self.array==test_circle).all() == True

    def test_uniformity(self):
        results = hazen_acr_uniformity.integral_uniformity(self.dcm_file, False)
        assert round(results,2) == self.piu


# class TestACRUniformityPhilips(unittest.TestCase):
#
class TestACRUniformityGE(unittest.TestCase):
    ACR_UNIFORMITY_DATA = pathlib.Path(TEST_DATA_DIR / 'acr')
    centre = [256, 253]
    piu = 85.69

    def setUp(self):
        self.dcm_file = pydicom.read_file(str(self.ACR_UNIFORMITY_DATA / 'GE' / 'Test' / '4.dcm'))

    def test_object_centre(self):
        data = self.dcm_file.pixel_array
        assert hazen_acr_uniformity.centroid_com(data) == self.centre

    def test_uniformity(self):
        results = hazen_acr_uniformity.integral_uniformity(self.dcm_file, False)
        assert round(results,2) == self.piu
