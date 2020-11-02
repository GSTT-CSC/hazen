import unittest
import pathlib

import pytest
import pydicom

import hazenlib.snr
from tests import TEST_DATA_DIR


class TestSnr(unittest.TestCase):

    SNR_DATA = pathlib.Path(TEST_DATA_DIR / 'snr')
    ORIENTATION = 'Transverse'
    OBJECT_CENTRE = (131, 122) # note these coordinates are (x, y) ie. (COLUMN, ROW)
    SNR_NORM_FACTOR = 9.761711312090041 
    IMAGE_SMOOTHED_SNR = 1874.81 # this value from MATLAB for tra_250_2meas_1.IMA, single image smoothed, normalised
    IMAGE_SUBTRACT_SNR = 2130.93 # this value from MATLAB for tra_250_2meas_1.IMA and tra_250_2meas_2.IMA, subtract method, normalised

    #setting up and lower bands of 10% error while SNR code being cleaned up
    UPPER_SMOOTHED_SNR = IMAGE_SMOOTHED_SNR*1.1
    LOWER_SMOOTHED_SNR = IMAGE_SMOOTHED_SNR*0.9

    UPPER_SUBTRACT_SNR = IMAGE_SUBTRACT_SNR*1.1
    LOWER_SUBTRACT_SNR = IMAGE_SUBTRACT_SNR*0.9

    def setUp(self):
        self.test_file = pydicom.read_file(str(self.SNR_DATA / 'tra_250_2meas_1.IMA'), force=True)
        self.test_file_2 = pydicom.read_file(str(self.SNR_DATA / 'tra_250_2meas_2.IMA'), force=True)

    def test_get_object_centre(self):
        assert hazenlib.snr.get_object_centre(self.test_file) == self.OBJECT_CENTRE

    def test_image_snr(self):
        val = hazenlib.snr.main(data=[self.test_file, self.test_file_2])
        self.assertTrue(self.LOWER_SMOOTHED_SNR <= val["snr_smoothing_normalised_seFoV250_2meas_slice5mm_tra_repeat_PSN_noDC_2_1"] <= self.UPPER_SMOOTHED_SNR)
        self.assertTrue(self.LOWER_SUBTRACT_SNR <= val["snr_subtraction_normalised_seFoV250_2meas_slice5mm_tra_repeat_PSN_noDC_2_1"] <= self.UPPER_SUBTRACT_SNR)

    def test_SNR_factor(self):
        SNR_factor=hazenlib.snr.get_normalised_snr_factor(self.test_file)
        assert(SNR_factor) == self.SNR_NORM_FACTOR

