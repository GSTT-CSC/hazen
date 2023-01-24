import os
import unittest
import pathlib
import pydicom

from hazenlib.tasks.acr_snr import ACRSNR
from tests import TEST_DATA_DIR, TEST_REPORT_DIR


class TestACRSNRSiemens(unittest.TestCase):
    ACR_SNR_DATA = pathlib.Path(TEST_DATA_DIR / 'acr')
    centre = [129, 128]
    norm_factor = 9.761711312090041
    snr = 335.27
    sub_snr = 76.48

    def setUp(self):
        self.acr_snr_task = ACRSNR(data_paths=[os.path.join(TEST_DATA_DIR, 'acr')],
                                   report_dir=pathlib.PurePath.joinpath(TEST_REPORT_DIR))
        self.dcm = pydicom.read_file(os.path.join(TEST_DATA_DIR, 'acr', 'Siemens', '6.dcm'))
        self.dcm2 = pydicom.read_file(os.path.join(TEST_DATA_DIR, 'acr', 'Siemens2', '7.dcm'))

    def test_object_centre(self):
        assert self.acr_snr_task.centroid(self.dcm)[1] == self.centre

    def test_normalisation_factor(self):
        SNR_factor = self.acr_snr_task.get_normalised_snr_factor(self.dcm)
        assert SNR_factor == self.norm_factor

    def test_snr_by_smoothing(self):
        snr, _ = self.acr_snr_task.snr_by_smoothing(self.dcm)
        assert round(snr, 2) == self.snr

    def test_snr_by_subtraction(self):
        snr, _ = self.acr_snr_task.snr_by_subtraction(self.dcm, self.dcm2)
        assert round(snr, 2) == self.sub_snr


class TestACRSNRGE(unittest.TestCase):
    ACR_SNR_DATA = pathlib.Path(TEST_DATA_DIR / 'acr')
    centre = [253, 256]
    norm_factor = 57.12810400630368
    snr = 39.97

    def setUp(self):
        self.acr_snr_task = ACRSNR(data_paths=[os.path.join(TEST_DATA_DIR, 'acr')],
                                   report_dir=pathlib.PurePath.joinpath(TEST_REPORT_DIR))
        self.dcm = pydicom.read_file(os.path.join(TEST_DATA_DIR, 'acr', 'GE', '4.dcm'))

    def test_object_centre(self):
        assert self.acr_snr_task.centroid(self.dcm)[1] == self.centre

    def test_normalisation_factor(self):
        SNR_factor = self.acr_snr_task.get_normalised_snr_factor(self.dcm)
        assert SNR_factor == self.norm_factor

    def test_snr_by_smoothing(self):
        snr, _ = self.acr_snr_task.snr_by_smoothing(self.dcm)
        assert round(snr, 2) == self.snr
