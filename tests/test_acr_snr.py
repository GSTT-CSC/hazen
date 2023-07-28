import os
import unittest
import pathlib
import pydicom

from hazenlib.tasks.acr_snr import ACRSNR
from hazenlib.ACRObject import ACRObject
from tests import TEST_DATA_DIR, TEST_REPORT_DIR


class TestACRSNRSiemens(unittest.TestCase):
    ACR_SNR_DATA = pathlib.Path(TEST_DATA_DIR / 'acr')
    norm_factor = 9.761711312090041
    snr = 344.15
    sub_snr = 75.94

    def setUp(self):
        self.acr_snr_task = ACRSNR(data_paths=[os.path.join(TEST_DATA_DIR, 'acr')],
                                   report_dir=pathlib.PurePath.joinpath(TEST_REPORT_DIR))
        self.acr_snr_task.ACR_obj = [ACRObject(
            [pydicom.read_file(os.path.join(TEST_DATA_DIR, 'acr', 'Siemens', f'{i}')) for i in
             os.listdir(os.path.join(TEST_DATA_DIR, 'acr', 'Siemens'))])]
        self.acr_snr_task.ACR_obj.append(
            ACRObject([pydicom.read_file(os.path.join(TEST_DATA_DIR, 'acr', 'Siemens2', f'{i}')) for i in
                       os.listdir(os.path.join(TEST_DATA_DIR, 'acr', 'Siemens2'))]))

        self.dcm = [i.dcm[6] for i in self.acr_snr_task.ACR_obj]

    def test_normalisation_factor(self):
        SNR_factor = self.acr_snr_task.get_normalised_snr_factor(self.dcm[0])
        assert SNR_factor == self.norm_factor

    def test_snr_by_smoothing(self):
        snr, _ = self.acr_snr_task.snr_by_smoothing(self.dcm[0])
        assert round(snr, 2) == self.snr

    def test_snr_by_subtraction(self):
        snr, _ = self.acr_snr_task.snr_by_subtraction(self.dcm[0], self.dcm[1])
        rounded_snr = round(snr, 2)

        print("\ntest_snr_by_subtraction.py::TestSnrBySubtraction::test_snr_by_subtraction")
        print("new_release_value:", rounded_snr)
        print("fixed_value:", self.sub_snr)

        assert rounded_snr == self.sub_snr


class TestACRSNRGE(unittest.TestCase):
    ACR_SNR_DATA = pathlib.Path(TEST_DATA_DIR / 'acr')
    norm_factor = 57.12810400630368
    snr = 40.19

    def setUp(self):
        self.acr_snr_task = ACRSNR(data_paths=[os.path.join(TEST_DATA_DIR, 'acr')],
                                   report_dir=pathlib.PurePath.joinpath(TEST_REPORT_DIR))
        self.acr_snr_task.ACR_obj = [ACRObject(
            [pydicom.read_file(os.path.join(TEST_DATA_DIR, 'acr', 'GE', f'{i}')) for i in
             os.listdir(os.path.join(TEST_DATA_DIR, 'acr', 'GE'))])]

        self.dcm = self.acr_snr_task.ACR_obj[0].dcm[6]

    def test_normalisation_factor(self):
        SNR_factor = self.acr_snr_task.get_normalised_snr_factor(self.dcm)
        assert SNR_factor == self.norm_factor

    def test_snr_by_smoothing(self):
        snr, _ = self.acr_snr_task.snr_by_smoothing(self.dcm)
        assert round(snr, 2) == self.snr
