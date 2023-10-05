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
        snr_task = SNR(input_data=files, report=False, measured_slice_width=5)
        result = snr_task.run()

        dict1 = {
            "task": "SNR",
            "file": ["SNR_SAG_MEAS1_23_1", "SNR_SAG_MEAS2_24_1"],
            "measurement": {
                "snr by smoothing": {
                    "SNR_SAG_MEAS1_23_1": {
                        "measured": 184.41,
                        "normalised": 1522.17
                    },
                    "SNR_SAG_MEAS2_24_1": {
                        "measured": 189.38,
                        "normalised": 1563.2
                    }
                },
                "snr by subtraction": {
                    "measured": 183.97,
                    "normalised": 1518.61
                }
            }
        }

        self.assertDictEqual(result, dict1)

    def test_relaxometry(self):
        path = str(TEST_DATA_DIR / 'relaxometry' / 'T1' / 'site3_ge' / 'plate4')
        files = get_dicom_files(path)
        relaxometry_task = Relaxometry(input_data=files, report=False)
        result = relaxometry_task.run(calc='T1', plate_number=4, verbose=False)

        dict1 = {
            "task": "Relaxometry",
            "file": "Spin_Echo_34_2_4_t1",
            "measurement": {"rms_frac_time_difference": 0.135}
        }
        self.assertEqual(dict1.keys(), result.keys())
        self.assertAlmostEqual(dict1['measurement']['rms_frac_time_difference'],
                               result['measurement']['rms_frac_time_difference'], 4)
