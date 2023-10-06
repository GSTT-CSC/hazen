import unittest
import pathlib
import os
import numpy as np

from tests import TEST_DATA_DIR, TEST_REPORT_DIR
from hazenlib.tasks.slice_position import SlicePosition
from hazenlib.utils import get_dicom_files
import copy


class TestSlicePosition(unittest.TestCase):
    SLICE_POS = pathlib.Path(TEST_DATA_DIR / 'slicepos')
    ROD_COORDS = (122.22222222222223, 83.0, 132.88888888888889, 180.33333333333334)
    # SLICE_POSITION_OUTPUT = ['0.151', '0.0964', '0.237', '0.101', '0.224', '0.103', '0.0873', '0.0386', '0.0458',
    #                          '0.0484', '0.110', '0.0951', '0.141', '0.00585', '0.164', '0.135', '0.0185', '0.0449',
    #                          '0', '0.0523', '0.0818', '0.165', '0.000858', '0.139', '0.00411', '0.247', '0.0931',
    #                          '0.127', '0.169', '0.240', '0.135', '0.265', '0.0125', '0.222', '0.139', '0.237', '0.125',
    #                          '0.152', '0.232', '0.0944']
    SLICE_POSITION_OUTPUT = [0.151, 0.0964, 0.098, 0.101, 0.224, 0.103, 0.0873, 0.0386, 0.0459,
                             0.0484, 0.11, 0.0952, 0.141, 0.00583, 0.164, 0.135, 0.0186, 0.0449, 0,
                             0.0523, 0.0818, 0.165, 0.000847, 0.139, 0.0040, 0.247, 0.093, 0.127,
                             0.17, 0.24, 0.135, 0.266, 0.0125, 0.222, 0.139, 0.237, 0.125, 0.152,
                             0.233, 0.0945]

    def setUp(self):
        self.hazen_slice_position = SlicePosition(
            input_data=get_dicom_files(os.path.join(self.SLICE_POS, 'SLICEPOSITION')),
            report_dir=pathlib.PurePath.joinpath(TEST_REPORT_DIR))
        self.sorted_slices = copy.deepcopy(self.hazen_slice_position.dcm_list)
        self.sorted_slices.sort(key=lambda x: x.SliceLocation)  # sort by slice location

    def test_get_rods_coords(self):
        # test just one file first, eventually test all 40 files

        test_dcm = self.sorted_slices[10]  # first ten (0-9) are dropped

        lx, ly, rx, ry = self.hazen_slice_position.get_rods_coords(test_dcm)
        # import matplotlib.pyplot as plt
        # plt.imshow(test_dcm.pixel_array, cmap='gray')
        # plt.scatter([lx, rx], [ly, ry], 3, c='green')
        # plt.savefig('test_get_rods_coords.png')

        assert (lx, ly, rx, ry) == self.ROD_COORDS

    def test_slice_position_errors(self):
        slice_positions = self.hazen_slice_position.slice_position_error(
            self.sorted_slices[10:50]
        )

        print("\ntest_slice_position.py::TestSlicePosition::test_slice_position")
        print("new_release_value:", slice_positions)
        print("fixed_value", self.SLICE_POSITION_OUTPUT)

        np.testing.assert_allclose(
            self.SLICE_POSITION_OUTPUT, slice_positions, atol=0.005)


# now test on canon data
class CanonTestSlicePosition(TestSlicePosition):
    SLICE_POS = pathlib.Path(TEST_DATA_DIR / 'slicepos')
    ROD_COORDS = (123.0, 77.5, 130.88888888888889, 171.66666666666666)
    SLICE_POSITION_OUTPUT = [1.49, 1.35, 1.29, 1.12, 0.928, 0.885, 0.729, 0.557, 0.531, 0.635,
                             0.0882, 0.437, 0.215, 0.238, 0.196, 0.0415, 0.117, 0.131, 0, 0.194,
                             0.361, 0.0941, 0.424, 0.265, 0.446, 0.464, 0.587, 0.646, 0.607, 0.779,
                             1.07, 0.88, 0.94, 0.948, 1.23, 1.29, 1.58, 1.41, 1.71, 1.95]

    def setUp(self):
        # self.test_files = [pydicom.read_file(str(i), force=True) for i in (self.SLICE_POS / 'canon').iterdir()]
        # self.test_files.sort(key=lambda x: x.SliceLocation)
        self.hazen_slice_position = SlicePosition(input_data=get_dicom_files(os.path.join(self.SLICE_POS, 'canon')),
                                                  report_dir=pathlib.PurePath.joinpath(TEST_REPORT_DIR))
        self.sorted_slices = copy.deepcopy(self.hazen_slice_position.dcm_list)
        self.sorted_slices.sort(key=lambda x: x.SliceLocation)  # sort by slice location
