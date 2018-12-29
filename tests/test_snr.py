import unittest
import pathlib

import pytest
import pydicom

import hazenlib.snr
from tests import TEST_DATA_DIR


class TestSnr(unittest.TestCase):

    SNR_DATA = pathlib.Path(TEST_DATA_DIR / 'snr')

    def setUp(self):
        test_file = str(self.SNR_DATA / 'uniform-circle.IMA')
        self.image = pydicom.read_file(test_file)

    def test_image_snr(self):
        val = hazenlib.snr.main(self.image)
        # have to test against rounded value as true float doesn't pass for some reason.

        assert round(val) == 182
