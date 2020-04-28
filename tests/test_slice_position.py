import unittest
import pathlib

import pydicom

import hazenlib.slice_position as hazen_slice_position
from tests import TEST_DATA_DIR


class TestSlicePosition(unittest.TestCase):
    SLICE_POS = pathlib.Path(TEST_DATA_DIR / 'slicepos')
    ROD_COORDS = (122.22222222222223, 83.0, 132.88888888888889, 180.33333333333334)
    SLICE_POSITION_OUTPUT = ['0.151', '0.0964',  '0.237',  '0.101',  '0.224',  '0.103',  '0.0873',  '0.0386',  '0.0458',
                           '0.0484',  '0.110',  '0.0951', '0.141',  '0.00585',  '0.164',  '0.135',  '0.0185',  '0.0449',
                           '0',  '0.0523',  '0.0818',  '0.165',  '0.000858',  '0.139',  '0.00411', '0.247', '0.0931',
                           '0.127', '0.169', '0.240', '0.135', '0.265', '0.0125', '0.222', '0.139', '0.237', '0.125',
                           '0.152', '0.232', '0.0944']

    def setUp(self):
        self.test_files = [pydicom.read_file(str(i), force=True) for i in (self.SLICE_POS / 'SLICEPOSITION').iterdir()]
        self.test_files.sort(key=lambda x: x.SliceLocation)

    def test_get_rods_coords(self):
        # test just one file first, eventually test all 40 files

        test_dcm = self.test_files[10]  # first ten (0-9) are dropped

        lx, ly, rx, ry = hazen_slice_position.get_rods_coords(test_dcm)
        # import matplotlib.pyplot as plt
        # plt.imshow(test_dcm.pixel_array, cmap='gray')
        # plt.scatter([lx, rx], [ly, ry], 3, c='green')
        # plt.savefig('test_get_rods_coords.png')

        assert (lx, ly, rx, ry) == self.ROD_COORDS

    def test_slice_position(self):
        results = hazen_slice_position.main(self.test_files)

        assert results == self.SLICE_POSITION_OUTPUT



#now test on canon data
class CanonTestSlicePosition(TestSlicePosition):

    SLICE_POS = pathlib.Path(TEST_DATA_DIR / 'slicepos')
    ROD_COORDS = (123.0, 77.5, 130.88888888888889, 171.66666666666666)
    SLICE_POSITION_OUTPUT = ['1.49','1.35','1.29','1.12','0.928','0.885','0.729','0.557','0.531','0.635','0.0882','0.437','0.215','0.238','0.196','0.0415','0.117','0.131','0','0.194','0.361','0.0941','0.424','0.265','0.446','0.464','0.587','0.646','0.672','0.779','1.07','0.880','0.940','0.948','1.23','1.29','1.58','1.41','1.71','1.95']

    def setUp(self):
        self.test_files = [pydicom.read_file(str(i), force=True) for i in (self.SLICE_POS / 'canon').iterdir()]
        self.test_files.sort(key=lambda x: x.SliceLocation)

