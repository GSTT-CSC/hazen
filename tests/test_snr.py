import unittest
import pathlib

import pytest
import pydicom

import hazenlib.snr
from tests import TEST_DATA_DIR


class TestSnr(unittest.TestCase):

    SNR_DATA = pathlib.Path(TEST_DATA_DIR / 'snr')
    ORIENTATION = 'Transverse'
    OBJECT_CENTRE = (131, 122)

    def setUp(self):
        self.test_file = pydicom.read_file(str(self.SNR_DATA / 'tra_250_2meas_1.IMA'), force=True)

    def test_get_object_centre(self):
        assert hazenlib.snr.get_object_centre(self.test_file) == self.OBJECT_CENTRE

    def test_image_snr(self):
        val = hazenlib.snr.main(data=[self.test_file])
        assert val["seFoV250_2meas_slice5mm_tra_repeat_PSN_noDC_2_1_normalised_snr_smoothing"] == 2509.1319217231753
