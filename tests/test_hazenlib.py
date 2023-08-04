import sys
from tests import TEST_DATA_DIR, TEST_REPORT_DIR
import unittest
import pydicom
import hazenlib
from hazenlib.utils import get_dicom_files, is_dicom_file
from hazenlib.tasks.snr import SNR
from hazenlib.tasks.relaxometry import Relaxometry


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

    def test_snr_measured_slice_width(self):
        path = str(TEST_DATA_DIR / 'snr' / 'GE')
        files = get_dicom_files(path)
        snr_task = SNR(data_paths=files, report=False)
        result = snr_task.run(measured_slice_width=1)

        dict1 = {'snr_subtraction_measured_SNR_SNR_SAG_MEAS1_23_1': 183.97,
                 'snr_subtraction_normalised_SNR_SNR_SAG_MEAS1_23_1': 7593.04,
                 'snr_smoothing_measured_SNR_SNR_SAG_MEAS1_23_1': 184.41,
                 'snr_smoothing_measured_SNR_SNR_SAG_MEAS2_24_1': 189.38,
                 'snr_smoothing_normalised_SNR_SNR_SAG_MEAS1_23_1': 7610.83,
                 'snr_smoothing_normalised_SNR_SNR_SAG_MEAS2_24_1': 7816.0}

        self.assertDictEqual(result['SNR_SNR_SAG_MEAS1_23_1'], dict1)

    def test_relaxometry(self):
        path = str(TEST_DATA_DIR / 'relaxometry' / 'T1' / 'site3_ge' / 'plate4')
        files = get_dicom_files(path)
        relaxometry_task = Relaxometry(data_paths=files, report=False)
        result = relaxometry_task.run(calc='T1', plate_number=4, verbose=False)

        dict1 = {'Spin Echo_32_2_P4_t1': {'rms_frac_time_difference': 0.13499936644959437}}
        self.assertEqual(dict1.keys(), result.keys())
        self.assertAlmostEqual(dict1['Spin Echo_32_2_P4_t1']['rms_frac_time_difference'],
                               result['Spin Echo_32_2_P4_t1']['rms_frac_time_difference'], 4)
