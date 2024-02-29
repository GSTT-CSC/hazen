import os
import unittest
import pathlib
import pydicom

from hazenlib.utils import get_dicom_files
from hazenlib.tasks.acr_ghosting import ACRGhosting
from hazenlib.ACRObject import ACRObject
from tests import TEST_DATA_DIR, TEST_REPORT_DIR


class TestACRGhostingSiemens(unittest.TestCase):
    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "Siemens")
    psg = 0.034

    def setUp(self):
        input_files = get_dicom_files(self.ACR_DATA)
        self.acr_ghosting_task = ACRGhosting(
            input_data=input_files,
            report_dir=pathlib.PurePath.joinpath(TEST_REPORT_DIR),
        )

    def test_ghosting(self):
        ghosting_val = round(
            self.acr_ghosting_task.get_signal_ghosting(
                self.acr_ghosting_task.ACR_obj.slice_stack[6]
            ),
            3,
        )

        print("\ntest_ghosting.py::TestGhosting::test_ghosting")
        print("new_release_value:", ghosting_val)
        print("fixed_value:", self.psg)

        assert ghosting_val == self.psg


class TestACRGhostingGE(TestACRGhostingSiemens):
    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "GE")
    psg = 0.489
