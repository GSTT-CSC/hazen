import os
import pathlib
import unittest
import pytest

from hazenlib.tasks.uniformity import Uniformity
from tests import TEST_DATA_DIR, TEST_REPORT_DIR


class TestUniformity(unittest.TestCase):
    UNIFORMITY_DATA = pathlib.Path(TEST_DATA_DIR / 'uniformity')
    IPEM_HORIZONTAL = 1.0
    IPEM_VERTICAL = 0.98125

    def setUp(self):
        self.uniformity_task = Uniformity(
            input_data=[os.path.join(self.UNIFORMITY_DATA, 'axial_oil.IMA')],
            report_dir=pathlib.PurePath.joinpath(TEST_REPORT_DIR))

    def test_uniformity(self):
        results = self.uniformity_task.run()
        horizontal_ipem = results['measurement']['horizontal %']
        vertical_ipem = results['measurement']['vertical %']

        print("\ntest_uniformity.py::TestUniformity::test_uniformity")
        print("new_release_value:", vertical_ipem)
        print("fixed_value:", self.IPEM_VERTICAL)

        assert horizontal_ipem == pytest.approx(self.IPEM_HORIZONTAL, abs=0.005)
        assert vertical_ipem == pytest.approx(self.IPEM_VERTICAL, abs=0.005)

class TestSagUniformity(TestUniformity):
    IPEM_HORIZONTAL = 0.46875
    IPEM_VERTICAL = 0.5125

    def setUp(self):
        self.uniformity_task = Uniformity(
            input_data=[os.path.join(self.UNIFORMITY_DATA, 'sag.dcm')],
            report_dir=pathlib.PurePath.joinpath(TEST_REPORT_DIR))


class TestCorUniformity(TestUniformity):
    IPEM_HORIZONTAL = 0.35
    IPEM_VERTICAL = 0.45

    def setUp(self):
        self.uniformity_task = Uniformity(
            input_data=[os.path.join(self.UNIFORMITY_DATA, 'cor.dcm')],
            report_dir=pathlib.PurePath.joinpath(TEST_REPORT_DIR))
