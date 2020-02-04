import unittest
import pathlib

import pydicom

import hazenlib.slice_position as hazen_slice_position
from tests import TEST_DATA_DIR


class TestSlicePosition(unittest.TestCase):

    SLICE_POS_DATA = pathlib.Path(TEST_DATA_DIR / 'slicepos')

    def setUp(self):
        self.test_files = [pydicom.read_file(str(i), force=True) for i in (self.SLICE_POS_DATA / 'SLICEPOSITION').iterdir()]

    def test_slice_position(self):
        results = hazen_slice_position.main(self.test_files)

        assert results == ['0.244', '0.310', '0.363', '0.0580', '0.00800', '0.561', '0.140', '0.207', '0.273', '0.339', '0.405', '0.471',
         '1.02', '0.103', '0.169', '0.235', '0.301', '0.368', '0.434', '0', '0.0660', '0.132', '0.198', '0.264',
         '0.330', '0.103', '0.462', '0.529', '0.392', '0.161', '0.759', '0.207', '0.128', '0.425', '0.491', '0.429',
         '0.363', '0.797', '0.256', '0.165']
