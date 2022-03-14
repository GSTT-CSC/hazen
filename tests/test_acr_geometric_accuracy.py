import unittest
import pathlib

import pydicom

import hazenlib.acr_uniformity as hazen_acr_uniformity
import numpy as np
from tests import TEST_DATA_DIR


class TestACRUniformitySiemens(unittest.TestCase):
    ACR_GEOMETRIC_ACCURACY_DATA = pathlib.Path(TEST_DATA_DIR / 'acr')
    centre = [128, 129]
    L = 190.43, 186.52, 191.41, 189.45, 190.43, 186.52

    def setUp(self):
        self.dcm_file = pydicom.read_file(str(self.ACR_GEOMETRIC_ACCURACY_DATA / 'Siemens' / 'Test' / '11.dcm'))
        self.dcm_file2 = pydicom.read_file(str(self.ACR_GEOMETRIC_ACCURACY_DATA / 'Siemens' / 'Test' / '7.dcm'))

    def test_object_centre(self):
        data = self.dcm_file.pixel_array
        assert hazen_acr_geometric_accuracy.centroid_com(data) == self.centre

    def test_geometric_accuracy(self):
        test_L = hazen_acr_geometric_accuracy.geo_accuracy(dcm_file,dcm_file2)
        assert (self.L==test_L).all() == True

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
