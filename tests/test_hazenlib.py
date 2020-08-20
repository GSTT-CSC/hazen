"""
Tests functions in the hazenlib.__init__.py file
"""
import pydicom

import hazenlib
from tests import TEST_DATA_DIR


class TestHazenlib:
    # Data by ImplementationVersionName
    PHILIPS_MR_53_1 = str(TEST_DATA_DIR / 'resolution' / 'philips' / 'IM-0004-0002.dcm')

    SIEMENS_MR_VE11C = str(TEST_DATA_DIR / 'resolution' / 'eastkent' / '256_sag.IMA')

    TOSHIBA_TM_MR_DCM_V3_0 = str(TEST_DATA_DIR / 'toshiba' / 'TOSHIBA_TM_MR_DCM_V3_0.dcm')

    GE_eFILM = str(TEST_DATA_DIR / 'ge' / 'ge_eFilm.dcm')

    def test_get_bandwidth_philips(self):
        bw = hazenlib.get_bandwidth(pydicom.read_file(self.PHILIPS_MR_53_1))
        assert bw == 205.0

    def test_get_bandwidth_siemens(self):
        bw = hazenlib.get_bandwidth(pydicom.read_file(self.SIEMENS_MR_VE11C))
        assert bw == 130.0

    def test_get_bandwidth_toshiba(self):
        bw = hazenlib.get_bandwidth(pydicom.read_file(self.TOSHIBA_TM_MR_DCM_V3_0))
        assert bw == 244.0

    def test_get_bandwidth_ge(self):
        bw = hazenlib.get_bandwidth(pydicom.read_file(self.GE_eFILM))
        assert bw == 156.25

