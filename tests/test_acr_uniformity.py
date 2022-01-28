import unittest
import pathlib

import pydicom

import hazenlib.acr_uniformity as hazen_acr_uniformity
from tests import TEST_DATA_DIR


class TestACRUniformitySiemens(unittest.TestCase):

    ACR_UNIFORMITY_DATA = pathlib.Path(TEST_DATA_DIR / 'acr_uniformity')
    centre = (128, 129)
    centre_2 = (128, 128)
    piu = 67.91
    piu_2 = 69.34

    def setUp(self):
        self.test_file = [pydicom.read_file(str(self.ACR_UNIFORMITY_DATA / 'Siemens' / '01_ACR_T1_TRA_HEAD_NORMOFF'), force=True)]
        self.test_file_2 = [pydicom.read_file(str(self.ACR_UNIFORMITY_DATA / 'Siemens' / '04b_ACR_T1_TRA_HEAD_8_33_NORMOFF'), force=True)]

    def test_object_centre(self):
        assert hazen_acr_uniformity.centroid_com(self.test_file) == self.centre
        assert hazen_acr_uniformity.centroid_com(self.test_file_2) == self.centre_2

    def test_uniformity_1(self):
        results = hazen_acr_uniformity.main(self.test_file)
        key = f"{self.test_file[0].SeriesDescription}_{self.test_file[0].SeriesNumber}_{self.test_file[0].InstanceNumber}"
        assert results[key]['percent']['integral']['uniformity'] == self.piu

    def test_uniformity_2(self):
        results = hazen_acr_uniformity.main(self.test_file_2)
        key = f"{self.test_file_2[0].SeriesDescription}_{self.test_file_2[0].SeriesNumber}_{self.test_file_2[0].InstanceNumber}"
        assert results[key]['percent']['integral']['uniformity'] == self.piu_2

# class TestACRUniformityPhilips(unittest.TestCase):
#
# class TestACRUniformityGE(unittest.TestCase):
