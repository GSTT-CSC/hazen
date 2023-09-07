import os
import unittest
import pathlib
import pydicom

from hazenlib.utils import get_dicom_files
from hazenlib.tasks.acr_uniformity import ACRUniformity
from hazenlib.ACRObject import ACRObject
from tests import TEST_DATA_DIR, TEST_REPORT_DIR


class TestACRUniformitySiemens(unittest.TestCase):
    piu = 67.95

    def setUp(self):
        ACR_DATA_SIEMENS = pathlib.Path(TEST_DATA_DIR / 'acr' / 'Siemens')
        siemens_files = get_dicom_files(ACR_DATA_SIEMENS)

        self.acr_uniformity_task = ACRUniformity(
            input_data=siemens_files,
            report_dir=pathlib.PurePath.joinpath(TEST_REPORT_DIR))

        self.acr_uniformity_task.ACR_obj = ACRObject(
            [pydicom.read_file(os.path.join(ACR_DATA_SIEMENS, f'{i}')) for i in
             os.listdir(ACR_DATA_SIEMENS)])

        self.dcm = self.acr_uniformity_task.ACR_obj.dcm[6]

    def test_uniformity(self):
        results = self.acr_uniformity_task.get_integral_uniformity(self.dcm)
        rounded_results = round(results, 2)

        print("\ntest_uniformity.py::TestUniformity::test_uniformity")
        print("new_release_values:", rounded_results)
        print("fixed_values:", self.piu)

        assert rounded_results == self.piu


# TODO: Add unit tests for Philips datasets.

class TestACRUniformityGE(unittest.TestCase):
    piu = 85.17

    def setUp(self):
        ACR_DATA_GE = pathlib.Path(TEST_DATA_DIR / 'acr' / 'GE')
        ge_files = get_dicom_files(ACR_DATA_GE)

        self.acr_uniformity_task = ACRUniformity(
            input_data=ge_files,
            report_dir=pathlib.PurePath.joinpath(TEST_REPORT_DIR))
        self.acr_uniformity_task.ACR_obj = ACRObject(
            [pydicom.read_file(os.path.join(ACR_DATA_GE, f'{i}')) for i in
             os.listdir(ACR_DATA_GE)])

        self.dcm = self.acr_uniformity_task.ACR_obj.dcm[6]

    def test_uniformity(self):
        results = self.acr_uniformity_task.get_integral_uniformity(self.dcm)
        assert round(results, 2) == self.piu
