import unittest
import pathlib

import pydicom

import hazenlib.uniformity as hazen_uniformity
from tests import TEST_DATA_DIR


class TestUniformity(unittest.TestCase):

    UNIFORMITY_DATA = pathlib.Path(TEST_DATA_DIR / 'uniformity')
    IPEM_HORIZONTAL = 1.0
    IPEM_VERTICAL = 0.98125

    def setUp(self):
        self.test_file = [pydicom.read_file(str(self.UNIFORMITY_DATA / 'axial_oil.IMA'), force=True)]

    def test_uniformity(self):
        results = hazen_uniformity.main(self.test_file)
        key = f"{self.test_file[0].SeriesDescription}_{self.test_file[0].SeriesNumber}_{self.test_file[0].InstanceNumber}"
        assert results[key]['horizontal']['IPEM'] == self.IPEM_HORIZONTAL
        assert results[key]['vertical']['IPEM'] == self.IPEM_VERTICAL


class TestSagUniformity(TestUniformity):
    IPEM_HORIZONTAL = 0.46875
    IPEM_VERTICAL = 0.5125

    def setUp(self):
        self.test_file = [pydicom.read_file(str(self.UNIFORMITY_DATA / 'sag.dcm'), force=True)]


class TestCorUniformity(TestUniformity):
    IPEM_HORIZONTAL = 0.35
    IPEM_VERTICAL = 0.45

    def setUp(self):
        self.test_file = [pydicom.read_file(str(self.UNIFORMITY_DATA / 'cor.dcm'), force=True)]
