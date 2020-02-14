import unittest
import pathlib

import pytest
import pydicom

import hazenlib.snr
from tests import TEST_DATA_DIR


class TestSnr(unittest.TestCase):

    SNR_DATA = pathlib.Path(TEST_DATA_DIR / 'snr')

    def setUp(self):
        self.test_file = pydicom.read_file(str(self.SNR_DATA / 'uniform-circle.IMA'), force=True)

    def test_image_snr(self):
        val = hazenlib.snr.main(data=[self.test_file])
        assert val["normalised_snr_smoothing_method_0"] == 2509.1319217231753
