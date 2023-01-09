import os
import unittest
import pathlib
import pydicom

from hazenlib.tasks.acr_ghosting import ACRGhosting
from tests import TEST_DATA_DIR, TEST_REPORT_DIR


class TestACRGhostingSiemens(unittest.TestCase):
    ACR_GHOSTING_DATA = pathlib.Path(TEST_DATA_DIR / 'acr')
    centre = [129, 128]
    psg = 0.056

    def setUp(self):
        self.acr_ghosting_task = ACRGhosting(data_paths=[os.path.join(TEST_DATA_DIR, 'acr')],
                                             report_dir=pathlib.PurePath.joinpath(TEST_REPORT_DIR))
        self.dcm = pydicom.read_file(os.path.join(TEST_DATA_DIR, 'acr', 'Siemens', '6.dcm'))

    def test_object_centre(self):
        assert self.acr_ghosting_task.centroid_com(self.dcm.pixel_array)[1] == self.centre

    def test_ghosting(self):
        assert round(self.acr_ghosting_task.get_signal_ghosting(self.dcm), 3) == self.psg


class TestACRGhostingGE(unittest.TestCase):
    ACR_GHOSTING_DATA = pathlib.Path(TEST_DATA_DIR / 'acr')
    centre = [253, 256]
    psg = 0.487

    def setUp(self):
        self.acr_ghosting_task = ACRGhosting(data_paths=[os.path.join(TEST_DATA_DIR, 'acr')],
                                             report_dir=pathlib.PurePath.joinpath(TEST_REPORT_DIR))
        self.dcm = pydicom.read_file(os.path.join(TEST_DATA_DIR, 'acr', 'GE', '4.dcm'))

    def test_object_centre(self):
        assert self.acr_ghosting_task.centroid_com(self.dcm.pixel_array)[1] == self.centre

    def test_ghosting(self):
        assert round(self.acr_ghosting_task.get_signal_ghosting(self.dcm), 3) == self.psg
