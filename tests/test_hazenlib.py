"""
Tests functions in the hazenlib.__init__.py file
"""
import unittest
import pydicom
import hazenlib




from tests import TEST_DATA_DIR


class TestHazenlib(unittest.TestCase):
    # Data by ImplementationVersionName
    # all test values are taken from DICOM headers

    ROWS = 512
    COLUMNS = 512
    TR_CHECK = 500
    BW = 205.0
    ENHANCED = False

    def setUp(self):
        self.file = str(TEST_DATA_DIR / 'resolution' / 'philips' / 'IM-0004-0002.dcm')
        self.dcm = pydicom.read_file(self.file)
    def test_get_bandwidth(self):
        bw = hazenlib.get_bandwidth(self.dcm)
        assert bw == self.BW

    def test_get_rows(self):
        rows = hazenlib.get_rows(self.dcm)
        assert rows == self.ROWS

    def test_get_columns(self):
        columns = hazenlib.get_columns(self.dcm)
        assert columns == self.COLUMNS

    def test_get_TR(self):
        TR = hazenlib.get_TR(self.dcm)
        assert TR == self.TR_CHECK

    def test_is_enhanced_dicom(self):
        enhanced = hazenlib.is_enhanced_dicom(self.dcm)
        assert enhanced == self.ENHANCED






class TestFactorsPhilipsMR531(TestHazenlib):
    #PHILIPS_MR_53_1

    ROWS = 512
    COLUMNS = 512
    TR_CHECK = 500
    BW = 205.0

    def setUp(self):
        self.file = str(TEST_DATA_DIR / 'resolution' / 'philips' / 'IM-0004-0002.dcm')
        self.dcm = pydicom.read_file(self.file)

class TestFactorsSiemensMRVE11C(TestHazenlib):
    #SIEMENS_MR_VE11C
    ROWS = 256
    COLUMNS = 256
    TR_CHECK = 500
    BW = 130.0

    def setUp(self):
        self.file = str(TEST_DATA_DIR / 'resolution' / 'eastkent' / '256_sag.IMA')
        self.dcm = pydicom.read_file(self.file)

class TestFactorsToshibaTMMRDCMV30(TestHazenlib):
    # TOSHIBA_TM_MR_DCM_V3_0
    ROWS = 256
    COLUMNS = 256
    TR_CHECK = 45.0
    BW = 244.0

    def setUp(self):
        self.file = str(TEST_DATA_DIR / 'toshiba' / 'TOSHIBA_TM_MR_DCM_V3_0.dcm')
        self.dcm = pydicom.read_file(self.file)

class TestFactorsGEeFilm(TestHazenlib):
    # GE_eFILM
    ROWS = 256
    COLUMNS = 256
    TR_CHECK = 1000.0
    BW = 156.25

    def setUp(self):
        self.file = str(TEST_DATA_DIR / 'ge' / 'ge_eFilm.dcm')
        self.dcm = pydicom.read_file(self.file)







if __name__ == "__main__":
    unittest.main()





