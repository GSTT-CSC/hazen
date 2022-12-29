import pydicom
from tests import TEST_DATA_DIR, TEST_REPORT_DIR
import hazenlib
import unittest
import numpy as np
from docopt import docopt
from hazenlib.logger import logger
from pprint import pprint
import sys
import ast
import os

TEST_DICOM = str(TEST_DATA_DIR / 'toshiba' / 'TOSHIBA_TM_MR_DCM_V3_0.dcm')
TEST_DICOM = pydicom.read_file(TEST_DICOM)
print(TEST_DICOM.Columns * TEST_DICOM.PixelSpacing[0])
test_dicoms = {'philips': {'file': str(TEST_DATA_DIR / 'resolution' / 'philips' / 'IM-0004-0002.dcm'),
                           'MANUFACTURER': 'philips',
                           'ROWS': 512,
                           'COLUMNS': 512,
                           'TR_CHECK': 500,
                           'BW': 205.0,
                           'ENHANCED': False,
                           'PIX_ARRAY': 1,
                           'SLICE_THICKNESS': 5,
                           'PIX_SIZE': [0.48828125, 0.48828125],
                           'AVERAGE': 1},
               'siemens': {'file': str(TEST_DATA_DIR / 'resolution' / 'eastkent' / '256_sag.IMA'),
                           'MANUFACTURER': 'siemens',
                           'ROWS': 256,
                           'COLUMNS': 256,
                           'TR_CHECK': 500,
                           'BW': 130.0,
                           'ENHANCED': False,
                           'PIX_ARRAY': 1,
                           'SLICE_THICKNESS': 5,
                           'PIX_SIZE': [0.9765625, 0.9765625],
                           'AVERAGE': 1},
               'toshiba': {'file': str(TEST_DATA_DIR / 'toshiba' / 'TOSHIBA_TM_MR_DCM_V3_0.dcm'),
                           'MANUFACTURER': 'toshiba',
                           'ROWS': 256,
                           'COLUMNS': 256,
                           'TR_CHECK': 45.0,
                           'BW': 244.0,
                           'ENHANCED': False,
                           'PIX_ARRAY': 1,
                           'SLICE_THICKNESS': 6,
                           'PIX_SIZE': [1.0, 1.0],
                           'AVERAGE': 1},
               'ge': {'file': str(TEST_DATA_DIR / 'ge' / 'ge_eFilm.dcm'),
                      'MANUFACTURER': 'ge',
                      'ROWS': 256,
                      'COLUMNS': 256,
                      'TR_CHECK': 1000.0,
                      'BW': 156.25,
                      'ENHANCED': False,
                      'PIX_ARRAY': 1,
                      'SLICE_THICKNESS': 5,
                      'PIX_SIZE': [0.625, 0.625],
                      'AVERAGE': 1}}


class TestHazenlib(unittest.TestCase):

    def test_get_manufacturer(self):
        for manufacturer in test_dicoms.keys():
            with pydicom.read_file(test_dicoms[manufacturer]['file']) as dcm:
                assert hazenlib.get_manufacturer(dcm) == test_dicoms[manufacturer]['MANUFACTURER']

    def test_get_rows(self):
        for manufacturer in test_dicoms.keys():
            with pydicom.read_file(test_dicoms[manufacturer]['file']) as dcm:
                rows = hazenlib.get_rows(dcm)
                assert rows == test_dicoms[manufacturer]['ROWS']

    def test_get_columns(self):
        for manufacturer in test_dicoms.keys():
            with pydicom.read_file(test_dicoms[manufacturer]['file']) as dcm:
                columns = hazenlib.get_columns(dcm)
                assert columns == test_dicoms[manufacturer]['COLUMNS']

    def test_get_TR(self):
        for manufacturer in test_dicoms.keys():
            with pydicom.read_file(test_dicoms[manufacturer]['file']) as dcm:
                TR = hazenlib.get_TR(dcm)
                assert TR == test_dicoms[manufacturer]['TR_CHECK']

    def test_get_bandwidth(self):
        for manufacturer in test_dicoms.keys():
            with pydicom.read_file(test_dicoms[manufacturer]['file']) as dcm:
                bw = hazenlib.get_bandwidth(dcm)
                assert bw == test_dicoms[manufacturer]['BW']

    def test_is_enhanced(self):
        for manufacturer in test_dicoms.keys():
            with pydicom.read_file(test_dicoms[manufacturer]['file']) as dcm:
                enhanced = hazenlib.is_enhanced_dicom(dcm)
                assert enhanced == test_dicoms[manufacturer]['ENHANCED']

    def test_get_num_of_frames(self):
        for manufacturer in test_dicoms.keys():
            with pydicom.read_file(test_dicoms[manufacturer]['file']) as dcm:
                pix_arr = hazenlib.get_num_of_frames(dcm)
                assert pix_arr == test_dicoms[manufacturer]['PIX_ARRAY']

    def test_get_slice_thickness(self):
        for manufacturer in test_dicoms.keys():
            with pydicom.read_file(test_dicoms[manufacturer]['file']) as dcm:
                slice_thick = hazenlib.get_slice_thickness(dcm)
                assert slice_thick == test_dicoms[manufacturer]['SLICE_THICKNESS']

    def get_average(self):
        for manufacturer in test_dicoms.keys():
            with pydicom.read_file(test_dicoms[manufacturer]['file']) as dcm:
                avg = hazenlib.get_average(dcm)
                assert avg == test_dicoms[manufacturer]['AVERAGE']

    def get_pixel_size(self):
        for manufacturer in test_dicoms.keys():
            with pydicom.read_file(test_dicoms[manufacturer]['file']) as dcm:
                pix_size = hazenlib.get_pixel_size(dcm)
                pix_size = list(pix_size)
                self.assertEqual(pix_size, test_dicoms[manufacturer]['PIX_SIZE'])

    def test_fov(self):
        test_dicoms = {'philips': {'file': str(TEST_DATA_DIR / 'resolution' / 'philips' / 'IM-0004-0002.dcm'),
                                   'MANUFACTURER': 'philips',
                                   'FOV': 250.0},
                       'siemens': {'file': str(TEST_DATA_DIR / 'resolution' / 'eastkent' / '256_sag.IMA'),
                                   'MANUFACTURER': 'siemens',
                                   'FOV': 250.0},
                       'toshiba': {'file': str(TEST_DATA_DIR / 'toshiba' / 'TOSHIBA_TM_MR_DCM_V3_0.dcm'),
                                   'MANUFACTURER': 'toshiba',
                                   'FOV': 256.0}}

        for manufacturer in test_dicoms.keys():
            with pydicom.read_file(test_dicoms[manufacturer]['file']) as dcm:
                # first test function
                fov = hazenlib.get_field_of_view(dcm)
                print(fov)
                assert fov == test_dicoms[manufacturer]['FOV']


class Test(unittest.TestCase):

    def setUp(self):
        self.file = str(TEST_DATA_DIR / 'resolution' / 'eastkent' / '256_sag.IMA')
        self.dcm = pydicom.read_file(self.file)
        self.dcm = self.dcm.pixel_array

    def test_rescale_to_byte(self):
        test_array = np.array([[1, 2], [3, 4]])
        TEST_OUT = np.array([[63, 127], [191, 255]])
        test_array = hazenlib.rescale_to_byte(test_array)
        test_array = test_array.tolist()
        TEST_OUT = TEST_OUT.tolist()
        self.assertListEqual(test_array, TEST_OUT)


class TestCliParser(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        self.file = str(TEST_DATA_DIR / 'resolution' / 'philips' / 'IM-0004-0002.dcm')
        self.dcm = pydicom.read_file(self.file)

    def test1_logger(self):
        sys.argv = ["hazen", "spatial_resolution", ".\\tests\\data\\resolution\\RESOLUTION\\", "--log", "warning"]

        sys.argv = [item.replace("\\", "/") for item in sys.argv]

        hazenlib.main()

        logging = hazenlib.logging

        self.assertEqual(logging.root.level, logging.WARNING)

    def test2_logger(self):
        sys.argv = ["hazen", "spatial_resolution", ".\\tests\\data\\resolution\\RESOLUTION\\"]

        sys.argv = [item.replace("\\", "/") for item in sys.argv]

        hazenlib.main()

        logging = hazenlib.logging

        self.assertEqual(logging.root.level, logging.INFO)

    def test_main_snr_exception(self):
        sys.argv = ["hazen", "spatial_resolution", ".\\tests\\data\\snr\\Siemens\\", "--measured_slice_width=10"]

        sys.argv = [item.replace("\\", "/") for item in sys.argv]

        self.assertRaises(Exception, hazenlib.main)

    def test_snr_measured_slice_width(self):
        sys.argv = ["hazen", "snr", ".\\tests\\data\\snr\\GE", "--measured_slice_width", "1"]

        sys.argv = [item.replace("\\", "/") for item in sys.argv]

        output = hazenlib.main()
        output_dict = ast.literal_eval(output)
        print(output_dict)
        dict1 = {'snr_subtraction_measured_SNR_SNR_SAG_MEAS1_23_1': 183.97,
                 'snr_subtraction_normalised_SNR_SNR_SAG_MEAS1_23_1': 7593.04,
                 'snr_smoothing_measured_SNR_SNR_SAG_MEAS2_24_1': 183.93,
                 'snr_smoothing_normalised_SNR_SNR_SAG_MEAS2_24_1': 7591.33,
                 'snr_smoothing_measured_SNR_SNR_SAG_MEAS1_23_1': 179.94,
                 'snr_smoothing_normalised_SNR_SNR_SAG_MEAS1_23_1': 7426.54}

        self.assertDictEqual(output_dict['SNR_SNR_SAG_MEAS1_23_1'], dict1)

    def test_relaxometry(self):
        sys.argv = ["hazen", "relaxometry", ".\\tests\\data\\relaxometry\\T1\\site3_ge\\plate4\\", "--plate_number",
                    "4", "--calc_t1"]

        sys.argv = [item.replace("\\", "/") for item in sys.argv]

        output = hazenlib.main()
        output_dict = ast.literal_eval(output)

        dict1 = {'Spin Echo_32_2_P4_t1': {'rms_frac_time_difference': 0.13499936644959437}}
        self.assertAlmostEqual(dict1['Spin Echo_32_2_P4_t1']['rms_frac_time_difference'],
                               output_dict['Spin Echo_32_2_P4_t1']['rms_frac_time_difference'], 4)
