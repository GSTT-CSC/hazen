import os
import unittest
import pathlib
import pydicom
import numpy as np

from hazenlib import HazenTask
from hazenlib.acr_tools import ACRTools
from tests import TEST_DATA_DIR, TEST_REPORT_DIR


class TestACRTools(unittest.TestCase):
    rotation = [-1.0, 0.0]
    centre = [(129, 130), (253, 255)]

    def setUp(self):
        self.Siemens_data = [pydicom.read_file(os.path.join(TEST_DATA_DIR, 'acr', 'Siemens', f'{i}')) for i in
                             os.listdir(os.path.join(TEST_DATA_DIR, 'acr', 'Siemens'))]
        self.GE_data = [pydicom.read_file(os.path.join(TEST_DATA_DIR, 'acr', 'GE', f'{i}')) for i in
                        os.listdir(os.path.join(TEST_DATA_DIR, 'acr', 'GE'))]

        self.Siemens_ACR_obj = ACRTools(self.Siemens_data)
        self.GE_ACR_obj = ACRTools(self.GE_data)

    def test_find_rotation(self):
        assert self.rotation[0] == np.round(self.Siemens_ACR_obj.determine_rotation(), 1)
        assert self.rotation[1] == np.round(self.GE_ACR_obj.determine_rotation(), 1)

    def test_find_centre(self):
        assert (self.centre[0] == np.round(self.Siemens_ACR_obj.centre, 1)).all() == True
        assert (self.centre[1] == np.round(self.GE_ACR_obj.centre, 1)).all() == True

    # def
