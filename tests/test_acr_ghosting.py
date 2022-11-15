import unittest
import pathlib

import pydicom

import hazenlib.acr_ghosting as hazen_acr_ghosting
import numpy as np
from tests import TEST_DATA_DIR

class TestACRGhostingSiemens(unittest.TestCase):
    ACR_GHOSTING_DATA = pathlib.Path(TEST_DATA_DIR / 'acr')
    centre = [129, 128]
    psg = 0.056

    def setUp(self):
        self.dcm_file = pydicom.read_file(str(self.ACR_GHOSTING_DATA / 'Siemens' / 'Test' / '6.dcm'))

    def test_object_centre(self):
        data = self.dcm_file.pixel_array
        assert hazen_acr_ghosting.centroid_com(data)[1] == self.centre

    def test_ghosting(self):
        results = hazen_acr_ghosting.signal_ghosting(self.dcm_file, False)
        assert round(results, 3) == self.psg


class TestACRGhostingGE(unittest.TestCase):
    ACR_GHOSTING_DATA = pathlib.Path(TEST_DATA_DIR / 'acr')
    centre = [253, 256]
    psg = 0.487

    def setUp(self):
        self.dcm_file = pydicom.read_file(str(self.ACR_GHOSTING_DATA / 'GE' / 'Test' / '4.dcm'))

    def test_object_centre(self):
        data = self.dcm_file.pixel_array
        assert hazen_acr_ghosting.centroid_com(data)[1] == self.centre

    def test_ghosting(self):
        results = hazen_acr_ghosting.signal_ghosting(self.dcm_file, False)
        assert round(results, 3) == self.psg