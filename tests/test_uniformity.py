import os
import pathlib
import unittest

from hazenlib.tasks.uniformity import Uniformity
from tests import TEST_DATA_DIR, TEST_REPORT_DIR


class TestUniformity(unittest.TestCase):
    UNIFORMITY_DATA = pathlib.Path(TEST_DATA_DIR / 'uniformity')
    IPEM_HORIZONTAL = 1.0
    IPEM_VERTICAL = 0.98125

    def setUp(self):
        self.uniformity_task = Uniformity(data_paths=[os.path.join(self.UNIFORMITY_DATA, 'axial_oil.IMA')],
                                          report_dir=pathlib.PurePath.joinpath(TEST_REPORT_DIR))

    def test_uniformity(self):
        results = self.uniformity_task.run()
        assert results[self.uniformity_task.key(self.uniformity_task.data[0])]['horizontal'][
                   'IPEM'] == self.IPEM_HORIZONTAL
        assert results[self.uniformity_task.key(self.uniformity_task.data[0])]['vertical']['IPEM'] == self.IPEM_VERTICAL


class TestSagUniformity(TestUniformity):
    IPEM_HORIZONTAL = 0.46875
    IPEM_VERTICAL = 0.5125

    def setUp(self):
        self.uniformity_task = Uniformity(data_paths=[os.path.join(self.UNIFORMITY_DATA, 'sag.dcm')],
                                          report_dir=pathlib.PurePath.joinpath(TEST_REPORT_DIR))


class TestCorUniformity(TestUniformity):
    IPEM_HORIZONTAL = 0.35
    IPEM_VERTICAL = 0.45

    def setUp(self):
        self.uniformity_task = Uniformity(data_paths=[os.path.join(self.UNIFORMITY_DATA, 'cor.dcm')],
                                          report_dir=pathlib.PurePath.joinpath(TEST_REPORT_DIR))
