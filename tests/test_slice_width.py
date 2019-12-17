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
        self.file = str(self.SLICE_WIDTH_DATA / 'SLICEWIDTH' / 'ANNUALQA.MR.HEAD_GENERAL.tra.slice_width.IMA')
        self.dcm = pydicom.read_file(self.file)

    def test_get_rods(self):
        rods = hazen_slice_width.get_rods(self.dcm)

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
        horizontal_distortion, vertical_distortion = hazen_slice_width.get_rod_distortions(self.matlab_rods, self.dcm)
        assert (round(horizontal_distortion, 2), round(vertical_distortion, 2)) == (0.13, 0.07)

    def test_get_ramp_profiles(self):
        top_centre = 102
        bottom_centre = 162
        matlab_profile = [693, 694.75, 688.6, 685.05, 682.2, 677.65, 677.35, 675.5, 668, 665.9, 663.05, 662.05, 662.1,
                          657.75, 653.65, 654.75, 650.1, 648.65, 646.1, 645.9, 643.1, 639.4, 641.35, 639.1, 635.95,
                          633.75, 633.9, 630.7, 627.25, 628.2, 628.05, 623.4, 624.35, 622.4, 621.15, 622.45, 615, 615.8,
                          611.15, 605.2, 602.15, 595.75, 582.75, 573.5, 561.25, 550.4, 538.25, 524.9, 511.65, 499.9,
                          489.7, 487.3, 477, 476.1, 469.55, 466.9, 467.7, 466, 465.05, 468, 464.7, 467.9, 464.65,
                          468.85, 467.55, 468, 469.05, 473.25, 479.6, 483.8, 492.3, 500, 512.5, 523, 535.95, 548.3,
                          561.8, 571.95, 585.35, 593.5, 601.55, 605.45, 609.3, 612.75, 617.1, 618.8, 620.35, 623.15,
                          621.7, 623.35, 625.25, 628.4, 628.8, 631.55, 632.55, 633.4, 634.35, 638.1, 639.35, 640.3,
                          641.75, 646.7, 648.1, 647.2, 648.95, 650.2, 653.95, 659.55, 660.8, 661.95, 663, 665.6, 666.15,
                          670.75, 672.6, 674.15, 675.9, 677.5, 682.45, 684.3]

        ramp_profiles = hazen_slice_width.get_ramp_profiles(self.dcm.pixel_array, self.matlab_rods)
        bottom_profiles = ramp_profiles["bottom"]
        mean_bottom_profile = np.mean(bottom_profiles, axis=0).tolist()

        assert ramp_profiles["bottom-centre"] == bottom_centre
        assert ramp_profiles["top-centre"] == top_centre
        assert mean_bottom_profile == matlab_profile

    def test_baseline_correction(self):
        # matlab top 0.0215   -2.9668  602.4568
        # matlab bottom [0.0239, -2.9349,  694.9520]

        ramps = hazen_slice_width.get_ramp_profiles(self.dcm.pixel_array, self.matlab_rods)

        top_mean_ramp = np.mean(ramps["top"], axis=0)
        top_coefficients = list(hazen_slice_width.baseline_correction(top_mean_ramp, sample_spacing=0.25)["f"])
        assert round(top_coefficients[0], 4) == 0.0215

        bottom_mean_ramp = np.mean(ramps["bottom"], axis=0)
        bottom_coefficients = list(hazen_slice_width.baseline_correction(bottom_mean_ramp, sample_spacing=0.25)["f"])
        assert round(bottom_coefficients[0], 4) == 0.0239

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
        sample_spacing = 0.25
        slice_thickness = self.dcm.SliceThickness
        ramps = hazen_slice_width.get_ramp_profiles(self.dcm.pixel_array, self.matlab_rods)
        top_mean_ramp = np.mean(ramps["top"], axis=0)
        bottom_mean_ramp = np.mean(ramps["bottom"], axis=0)
        ramps_baseline_corrected = {
            "top": hazen_slice_width.baseline_correction(top_mean_ramp, sample_spacing),
            "bottom": hazen_slice_width.baseline_correction(bottom_mean_ramp, sample_spacing)
        }

        trapezoid_fit, trapezoid_fit_coefficients = hazen_slice_width.get_initial_trapezoid_fit_and_coefficients(
            ramps_baseline_corrected["top"]["profile_corrected_interpolated"], slice_thickness)
        assert trapezoid_fit_coefficients[:4] == [47, 55, 153, 178]

        trapezoid_fit, trapezoid_fit_coefficients = hazen_slice_width.get_initial_trapezoid_fit_and_coefficients(
            ramps_baseline_corrected["bottom"]["profile_corrected_interpolated"], slice_thickness)

        assert trapezoid_fit_coefficients[:4] == [47, 55, 164, 167]

    def test_fit_trapezoid(self):
        sample_spacing = 0.25
        slice_thickness = self.dcm.SliceThickness

        ramps = hazen_slice_width.get_ramp_profiles(self.dcm.pixel_array, self.matlab_rods)

        top_mean_ramp = np.mean(ramps["top"], axis=0)
        bottom_mean_ramp = np.mean(ramps["bottom"], axis=0)
        ramps_baseline_corrected = {
            "top": hazen_slice_width.baseline_correction(top_mean_ramp, sample_spacing),
            "bottom": hazen_slice_width.baseline_correction(bottom_mean_ramp, sample_spacing)
        }
        trapezoid_fit_coefficients, baseline_fit_coefficients = hazen_slice_width.fit_trapezoid(
            profiles=ramps_baseline_corrected["top"], slice_thickness=slice_thickness)

        # check top profile first
        matlab_trapezoid_fit_coefficients = [50, 54, 153, 170, -111.7920]
        matlab_baseline_fit_coefficients = [0.0216, -2.9658, 602.2568]

        for idx, value in enumerate(trapezoid_fit_coefficients):
            assert abs(value - matlab_trapezoid_fit_coefficients[idx]) <= 5

        for idx, value in enumerate(matlab_baseline_fit_coefficients):
            assert abs(value - matlab_baseline_fit_coefficients[idx]) <= 5
        # residual error is 3735.6

        ## check bottom profile now
        trapezoid_fit_coefficients, baseline_fit_coefficients = hazen_slice_width.fit_trapezoid(
            profiles=ramps_baseline_corrected["bottom"], slice_thickness=slice_thickness)

        matlab_trapezoid_fit_coefficients = [55.0000, 60.0000, 155.0000, 152.0000, -136.6194]
        matlab_baseline_fit_coefficients = [0.0239, -2.9389, 694.9520]
        for idx, value in enumerate(trapezoid_fit_coefficients):
            assert abs(value - matlab_trapezoid_fit_coefficients[idx]) <= 5

        for idx, value in enumerate(matlab_baseline_fit_coefficients):
            assert abs(value - matlab_baseline_fit_coefficients[idx]) <= 5

    def test_slice_width(self):
        results = hazen_slice_width.main([self.file])
        assert abs(results[0] - 5.48) < 0.1
