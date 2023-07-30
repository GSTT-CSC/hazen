import os
import unittest
import pathlib
import pydicom

from hazenlib.tasks.acr_ghosting import ACRGhosting
from hazenlib.ACRObject import ACRObject
from tests import TEST_DATA_DIR, TEST_REPORT_DIR


class TestACRGhostingSiemens(unittest.TestCase):
    ACR_GHOSTING_DATA = pathlib.Path(TEST_DATA_DIR / 'acr')
    centre = [129, 128]
    psg = 0.035

    def setUp(self):
        self.acr_ghosting_task = ACRGhosting(data_paths=[os.path.join(TEST_DATA_DIR, 'acr')],
                                             report_dir=pathlib.PurePath.joinpath(TEST_REPORT_DIR))

        self.acr_ghosting_task.ACR_obj = ACRObject(
            [pydicom.read_file(os.path.join(TEST_DATA_DIR, 'acr', 'Siemens', f'{i}')) for i in
             os.listdir(os.path.join(TEST_DATA_DIR, 'acr', 'Siemens'))])

        self.dcm = self.acr_ghosting_task.ACR_obj.dcm[6]

    def test_ghosting(self):
        ghosting_val = round(self.acr_ghosting_task.get_signal_ghosting(self.dcm), 3)

        print("\ntest_ghosting.py::TestGhosting::test_ghosting")
        print("new_release_value:", ghosting_val)
        print("fixed_value:", self.psg)

        assert ghosting_val == self.psg


class TestACRGhostingGE(unittest.TestCase):
    ACR_GHOSTING_DATA = pathlib.Path(TEST_DATA_DIR / 'acr')
    centre = [253, 256]
    psg = 0.471

    def setUp(self):
        self.acr_ghosting_task = ACRGhosting(data_paths=[os.path.join(TEST_DATA_DIR, 'acr')],
                                             report_dir=pathlib.PurePath.joinpath(TEST_REPORT_DIR))
        self.acr_ghosting_task.ACR_obj = ACRObject(
            [pydicom.read_file(os.path.join(TEST_DATA_DIR, 'acr', 'GE', f'{i}')) for i in
             os.listdir(os.path.join(TEST_DATA_DIR, 'acr', 'GE'))])

        self.dcm = self.acr_ghosting_task.ACR_obj.dcm[6]

    def test_ghosting(self):
        assert round(self.acr_ghosting_task.get_signal_ghosting(self.dcm), 3) == self.psg
