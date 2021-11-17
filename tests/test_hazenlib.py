"""
Tests functions in the hazenlib.__init__.py file
"""
import unittest
import pydicom
import hazenlib
import os
import numpy as np




from tests import TEST_DATA_DIR


class TestHazenlib(unittest.TestCase):
    # Data by ImplementationVersionName
    # all test values are taken from DICOM headers
    MANUFACTURER = "philips"
    ROWS = 512
    COLUMNS = 512
    TR_CHECK = 500
    BW = 205.0
    ENHANCED = False
    PIX_ARRAY = 1




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

    def test_get_num_of_frames(self):
        pix_arr = hazenlib.get_num_of_frames(self.dcm)
        assert pix_arr == self.PIX_ARRAY

    def test_get_slice_thickness(self):
        SLICE_THICK = self.dcm.SliceThickness
        slice_thick = hazenlib.get_slice_thickness(self.dcm)
        assert slice_thick == SLICE_THICK

    def test_get_pixel_size(self):
        PIX_SIZE = self.dcm.PixelSpacing
        PIX_SIZE=tuple(PIX_SIZE)
        pix_size = hazenlib.get_pixel_size(self.dcm)
        assert pix_size == PIX_SIZE

    def test_get_average(self):
        AVG = self.dcm.NumberOfAverages
        avg = hazenlib.get_average(self.dcm)
        assert avg == AVG

    def test_get_manufacturer(self):
        assert hazenlib.get_manufacturer(self.dcm) == self.MANUFACTURER









class TestFactorsPhilipsMR531(TestHazenlib):
    #PHILIPS_MR_53_1

    MANUFACTURER = 'philips'
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
    MANUFACTURER = 'siemens'

    def setUp(self):
        self.file = str(TEST_DATA_DIR / 'resolution' / 'eastkent' / '256_sag.IMA')
        self.dcm = pydicom.read_file(self.file)

class TestFactorsToshibaTMMRDCMV30(TestHazenlib):
    # TOSHIBA_TM_MR_DCM_V3_0
    ROWS = 256
    COLUMNS = 256
    TR_CHECK = 45.0
    BW = 244.0
    MANUFACTURER = 'toshiba'

    def setUp(self):
        self.file = str(TEST_DATA_DIR / 'toshiba' / 'TOSHIBA_TM_MR_DCM_V3_0.dcm')
        self.dcm = pydicom.read_file(self.file)

class TestFactorsGEeFilm(TestHazenlib):
    # GE_eFILM
    ROWS = 256
    COLUMNS = 256
    TR_CHECK = 1000.0
    BW = 156.25
    MANUFACTURER ='ge'

    def setUp(self):
        self.file = str(TEST_DATA_DIR / 'ge' / 'ge_eFilm.dcm')
        self.dcm = pydicom.read_file(self.file)



class Test(unittest.TestCase):

    def setUp(self):
        self.file = str(TEST_DATA_DIR / 'resolution' / 'eastkent' / '256_sag.IMA')
        self.dcm = pydicom.read_file(self.file)
        self.dcm = self.dcm.pixel_array

    def test_isupper(self):
        test_array = np.array([[1,2], [3,4]])
        TEST_OUT = np.array([[63,127],[191,255]])
        test_array = hazenlib.rescale_to_byte(test_array)
        test_array = test_array.tolist()
        TEST_OUT = TEST_OUT.tolist()
        self.assertListEqual(test_array, TEST_OUT)









if __name__ == "__main__":
    unittest.main()
