import os
import unittest
import pathlib
import pydicom

from hazenlib.utils import get_dicom_files
from hazenlib.tasks.acr_uniformity import ACRUniformity
from hazenlib.ACRObject import ACRObject
from tests import TEST_DATA_DIR, TEST_REPORT_DIR


class TestACRUniformitySiemens(unittest.TestCase):
    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "Siemens")
    piu = 68.04

    def setUp(self):
        input_files = get_dicom_files(self.ACR_DATA)

        self.acr_uniformity_task = ACRUniformity(
            input_data=input_files,
            report_dir=pathlib.PurePath.joinpath(TEST_REPORT_DIR),
        )

    def test_uniformity(self):
        results = self.acr_uniformity_task.get_integral_uniformity(
            self.acr_uniformity_task.ACR_obj.slice_stack[6]
        )
        rounded_results = round(results, 2)

        print("\ntest_uniformity.py::TestUniformity::test_uniformity")
        print("new_release_values:", rounded_results)
        print("fixed_values:", self.piu)

        assert rounded_results == self.piu


# TODO: Add unit tests for Philips datasets.


class TestACRUniformityGE(TestACRUniformitySiemens):
    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "GE")
    piu = 84.23
