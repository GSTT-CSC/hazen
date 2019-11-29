import unittest
import pathlib

import pydicom

import hazenlib.slice_width as hazen_slice_width
from tests import TEST_DATA_DIR


class TestSliceWidth(unittest.TestCase):
    SLICE_WIDTH_DATA = pathlib.Path(TEST_DATA_DIR / 'slicewidth')

    # 789
    # 456
    # 123
    MATLAB_RODS = [[71.2857, 191.5000], [130.5000, 190.5000], [190.6000, 189.4000],
                   [69.5000, 131.5000], [128.7778, 130.5000], [189.1111, 129.2778],
                   [69.0000, 71.5000], [128.1176, 70.4118], [188.5000, 69.2222]]

    # 789
    # 456
    # 123
    rods = [
        hazen_slice_width.Rod(69.16751269035532, 191.18274111675126),
        hazen_slice_width.Rod(131.0, 189.06060606060606),
        hazen_slice_width.Rod(189.5857142857143, 188.125),
        hazen_slice_width.Rod(68.45833333333333, 129.79166666666666),
        hazen_slice_width.Rod(127.26158445440957, 128.7982062780269),
        hazen_slice_width.Rod(188.93866666666668, 127.104),
        hazen_slice_width.Rod(68.62729124236253, 70.13034623217922),
        hazen_slice_width.Rod(126.66222961730449, 68.3144758735441),
        hazen_slice_width.Rod(188.20809898762656, 67.90438695163104)]

    def setUp(self):
        self.test_files = [str(i) for i in (self.SLICE_WIDTH_DATA / 'SLICEWIDTH').iterdir()]

    def test_get_rods(self):
        dcm = pydicom.read_file(self.test_files[0])
        rods = hazen_slice_width.get_rods(dcm)

        assert rods[0] == self.rods[0]

    def test_rod_distortions(self):
        dcm = pydicom.read_file(self.test_files[0])
        result = hazen_slice_width.get_rod_distortions(self.rods, dcm)
        print(result)
        assert result == (0.3464633436804712, 0.2880737989705986)

    def test_trapezoid(self):

        assert hazen_slice_width.trapezoid([]*100, 50, 10, 90, 30, 70) == 10

    def test_slice_width(self):
        results = hazen_slice_width.main(self.test_files)
        print(results)
        assert results == 5.48
