import unittest
import pathlib

import numpy as np
import pydicom

import hazenlib.slice_width as hazen_slice_width
from tests import TEST_DATA_DIR


class TestSliceWidth(unittest.TestCase):
    SLICE_WIDTH_DATA = pathlib.Path(TEST_DATA_DIR / 'slicewidth')

    # 789
    # 456
    # 123
    matlab_rods = [hazen_slice_width.Rod(71.2857, 191.5000),
                   hazen_slice_width.Rod(130.5000, 190.5000),
                   hazen_slice_width.Rod(190.6000, 189.4000),
                   hazen_slice_width.Rod(69.5000, 131.5000),
                   hazen_slice_width.Rod(128.7778, 130.5000),
                   hazen_slice_width.Rod(189.1111, 129.2778),
                   hazen_slice_width.Rod(69.0000, 71.5000),
                   hazen_slice_width.Rod(128.1176, 70.4118),
                   hazen_slice_width.Rod(188.5000, 69.2222)]

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

        assert rods == self.rods

    def test_get_rod_distances(self):
        # From MATLAB Rods
        # Horizontal_Distance_Pixels = 119.3328 (rod3 - rod1) 119.6318 (rod6 - rod4)  119.5217 (rod9 - rod7)
        # Vertical_Distance_Pixels = 120.0218 (rod1 - rod7)  120.1119 (rod2 - rod8)  120.1961 (rod3 - rod9)

        distances = hazen_slice_width.get_rod_distances(self.matlab_rods)
        print(distances)
        assert distances == ([119.333, 119.632, 119.522], [120.022, 120.112, 120.196])

    def test_get_rod_distortion_correction_coefficients(self):
        distances = hazen_slice_width.get_rod_distances(self.matlab_rods)
        print(hazen_slice_width.get_rod_distortion_correction_coefficients(distances[0]))
        assert hazen_slice_width.get_rod_distortion_correction_coefficients(distances[0]) == {"top": 0.9965,
                                                                                              "bottom": 0.9957}

    def test_rod_distortions(self):
        dcm = pydicom.read_file(self.test_files[0])
        result = hazen_slice_width.get_rod_distortions(self.rods, dcm)
        assert result == (0.3464633436804712, 0.2880737989705986)

    def test_get_profiles(self):
        pass

    def test_baseline_correction(self):
        # matlab top 0.0215   -2.9668  602.4568
        # matlab bottom [0.0239, -2.9349,  694.9520]

        dcm = pydicom.read_file(self.test_files[0])
        ramps = hazen_slice_width.get_ramp_profiles(dcm.pixel_array, self.matlab_rods)
        assert hazen_slice_width.baseline_correction(np.mean(ramps["bottom"], axis=0), sample_spacing=0.25)["baseline_fit"] == [0.0239, -2.9349,  694.9520]

    def test_trapezoid(self):
        # variables from one iteration of the original matlab script
        #  n_ramp, n_plateau, n_left_baseline, n_right_baseline, plateau_amplitude =
        #  55.0000   58.0000  156.0000  153.0000 -136.6194 and fwhm 113

        assert hazen_slice_width.trapezoid(55, 58, 156, 153, -136.6194)[1] == 113

    def test_get_initial_trapezoid_fit_and_coefficients(self):
        """
        Trapezoid_Fit_Coefficients_Initial = 48.0000   56.0000  153.0000  172.0000 -116.4920
        fwhm = 104
        Returns:
        """

    # def test_slice_width(self):
    #     results = hazen_slice_width.main(self.test_files)
    #
    #     assert results == 5.48
