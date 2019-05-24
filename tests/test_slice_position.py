import unittest
import pathlib

import pytest
import pydicom

import hazenlib.slice_position as hazen_slice_position
from tests import TEST_DATA_DIR


class TestSlicePosition(unittest.TestCase):

    SLICE_POS_DATA = pathlib.Path(TEST_DATA_DIR / 'slicepos')

    def setUp(self):
        self.test_files = [str(i) for i in (self.SLICE_POS_DATA / 'SLICEPOSITION').iterdir()]

    def test_slice_position(self):
        assert hazen_slice_position.main(self.test_files) == [
            '0.259', '0.324', '0.377', '0.0450', '0.0210', '0.573', '0.151', '0.216', '0.282', '0.347', '0.412',
            '0.477', '1.03', '0.108', '0.173', '0.239', '0.304', '0.369', '0.434', '0', '0.0650', '0.131', '0.196',
            '0.226', '0.326', '0.108', '0.457', '0.522', '0.100', '0.153', '0.769', '0.216', '0.138', '0.414', '0.479',
            '0.442', '0.377', '0.811', '0.240', '0.181']
