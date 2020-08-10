"""
Tests functions in the hazenlib.__init__.py file
"""
import pydicom

import hazenlib
from tests import TEST_DATA_DIR


class TestHazenlib:
    # Data by ImplementationVersionName
    PHILIPS_MR_53_1 = str(
        TEST_DATA_DIR / 'resolution' / 'philips' / 'IM-0004-0002.dcm')

    SIEMENS_MR_VE11C = str(
        TEST_DATA_DIR / 'resolution' / 'eastkent' / '256_sag.IMA')

    def test_get_bandwidth_philips(self):
        bw = hazenlib.get_bandwidth(pydicom.read_file(self.PHILIPS_MR_53_1))
        assert bw == 205.0

    def test_get_bandwidth_siemens(self):
        bw = hazenlib.get_bandwidth(pydicom.read_file(self.SIEMENS_MR_VE11C))
        assert bw == 130.0

