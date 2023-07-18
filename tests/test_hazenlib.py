import pydicom
from tests import TEST_DATA_DIR, TEST_REPORT_DIR
import unittest
import numpy as np
from docopt import docopt
from hazenlib.logger import logger
import hazenlib.utils
from pprint import pprint
import sys
import ast
import os


class TestCliParser(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        self.file = str(TEST_DATA_DIR / 'resolution' / 'philips' / 'IM-0004-0002.dcm')
        self.dcm = pydicom.read_file(self.file)

    def test1_logger(self):
        path = str(TEST_DATA_DIR / 'resolution' / 'RESOLUTION')
        sys.argv = ["hazen", "spatial_resolution", path, "--log", "warning"]

        hazenlib.main()

        logging = hazenlib.logging

        self.assertEqual(logging.root.level, logging.WARNING)

    def test2_logger(self):
        path = str(TEST_DATA_DIR / 'resolution' / 'RESOLUTION')
        sys.argv = ["hazen", "spatial_resolution", path]

        hazenlib.main()

        logging = hazenlib.logging

        self.assertEqual(logging.root.level, logging.INFO)

    def test_main_snr_exception(self):
        path = str(TEST_DATA_DIR / 'snr' / 'Siemens')
        sys.argv = ["hazen", "spatial_resolution", path, "--measured_slice_width=10"]

        self.assertRaises(Exception, hazenlib.main)

    def test_snr_measured_slice_width(self):
        path = str(TEST_DATA_DIR / 'snr' / 'GE')
        sys.argv = ["hazen", "snr", path, "--measured_slice_width", "1"]

        output = hazenlib.main()
        output_dict = ast.literal_eval(output)
        dict1 = {'snr_subtraction_measured_SNR_SNR_SAG_MEAS1_23_1': 183.97,
                 'snr_subtraction_normalised_SNR_SNR_SAG_MEAS1_23_1': 7593.04,
                 'snr_smoothing_measured_SNR_SNR_SAG_MEAS2_24_1': 183.93,
                 'snr_smoothing_normalised_SNR_SNR_SAG_MEAS2_24_1': 7591.33,
                 'snr_smoothing_measured_SNR_SNR_SAG_MEAS1_23_1': 179.94,
                 'snr_smoothing_normalised_SNR_SNR_SAG_MEAS1_23_1': 7426.54}

        self.assertDictEqual(output_dict['SNR_SNR_SAG_MEAS1_23_1'], dict1)

    def test_relaxometry(self):
        path = str(TEST_DATA_DIR / 'relaxometry' / 'T1' / 'site3_ge' / 'plate4')
        sys.argv = ["hazen", "relaxometry", path, "--plate_number", "4", "--calc_t1"]

        output = hazenlib.main()
        output_dict = ast.literal_eval(output)

        dict1 = {'Spin Echo_32_2_P4_t1': {'rms_frac_time_difference': 0.13499936644959437}}
        self.assertAlmostEqual(dict1['Spin Echo_32_2_P4_t1']['rms_frac_time_difference'],
                               output_dict['Spin Echo_32_2_P4_t1']['rms_frac_time_difference'], 4)
