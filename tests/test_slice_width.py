import unittest
import pathlib

import numpy as np
import pydicom

# import hazenlib.slice_width as hazen_slice_width
from tests import TEST_DATA_DIR, TEST_REPORT_DIR
from hazenlib.tasks.slice_width import SliceWidth
from hazenlib.utils import get_dicom_files, Rod
import os


class TestSliceWidth(unittest.TestCase):
    SLICE_WIDTH_DATA = pathlib.Path(TEST_DATA_DIR / 'slicewidth')

    """
    Notes
    -----
    The rod indices are ordered as:
        789
        456
        123 
    """
    matlab_rods = [Rod(71.2857, 191.5000),
                   Rod(130.5000, 190.5000),
                   Rod(190.6000, 189.4000),
                   Rod(69.5000, 131.5000),
                   Rod(128.7778, 130.5000),
                   Rod(189.1111, 129.2778),
                   Rod(69.0000, 71.5000),
                   Rod(128.1176, 70.4118),
                   Rod(188.5000, 69.2222)]

    rods = [Rod(70.26906602941604, 190.52291430040833),
            Rod(129.38344648450575, 189.5252799358382),
            Rod(189.6494724536544, 188.32774808447635),
            Rod(68.53084886954112, 130.56732921648214),
            Rod(127.86240947286896, 129.4605302262616),
            Rod(188.01124565987345, 128.2832650316875),
            Rod(67.97926729691507, 70.61103769200058),
            Rod(127.2060664869085, 69.42672715143607),
            Rod(187.49797283835656, 68.2890101413575)]

    DISTANCES = ([119.333, 119.632, 119.522], [120.022, 120.112, 120.196])
    DIST_CORR_COEFF = {"top": 0.9965, "bottom": 0.9957}
    ROD_DIST = (0.13, 0.07)

    TOP_CENTRE = 102
    BOTTOM_CENTRE = 162
    MATLAB_PROFILE = [693, 694.75, 688.6, 685.05, 682.2, 677.65, 677.35, 675.5, 668, 665.9, 663.05, 662.05, 662.1,
                      657.75, 653.65, 654.75, 650.1, 648.65, 646.1, 645.9, 643.1, 639.4, 641.35, 639.1, 635.95,
                      633.75, 633.9, 630.7, 627.25, 628.2, 628.05, 623.4, 624.35, 622.4, 621.15, 622.45, 615, 615.8,
                      611.15, 605.2, 602.15, 595.75, 582.75, 573.5, 561.25, 550.4, 538.25, 524.9, 511.65, 499.9,
                      489.7, 487.3, 477, 476.1, 469.55, 466.9, 467.7, 466, 465.05, 468, 464.7, 467.9, 464.65,
                      468.85, 467.55, 468, 469.05, 473.25, 479.6, 483.8, 492.3, 500, 512.5, 523, 535.95, 548.3,
                      561.8, 571.95, 585.35, 593.5, 601.55, 605.45, 609.3, 612.75, 617.1, 618.8, 620.35, 623.15,
                      621.7, 623.35, 625.25, 628.4, 628.8, 631.55, 632.55, 633.4, 634.35, 638.1, 639.35, 640.3,
                      641.75, 646.7, 648.1, 647.2, 648.95, 650.2, 653.95, 659.55, 660.8, 661.95, 663, 665.6, 666.15,
                      670.75, 672.6, 674.15, 675.9, 677.5, 682.45, 684.3]

    BLINE_TOP = 0.0215
    BLINE_BOT = 0.0239

    TRAP_FIT_COEFF_TOP = [47, 55, 153, 178]
    TRAP_FIT_COEFF_BOT = [47, 55, 164, 167]

    MATLAB_TRAP_FIT_COEFF = [50, 54, 153, 170, -111.7920]
    MATLAB_BLINE_FIT_COEFF = [0.0216, -2.9658, 602.2568]
    MATLAB_TRAP_FIT_COEFF_BOT = [55.0000, 60.0000, 155.0000, 152.0000, -136.6194]
    MATLAB_BLINE_FIT_COEFF_BOT = [0.0239, -2.9389, 694.9520]

    SW_MATLAB = 5.48

    def setUp(self):
        # self.file = str(self.SLICE_WIDTH_DATA / 'SLICEWIDTH' / 'ANNUALQA.MR.HEAD_GENERAL.tra.slice_width.IMA')
        # self.dcm = pydicom.read_file(self.file)
        self.slice_width = SliceWidth(
            input_data=[os.path.join(self.SLICE_WIDTH_DATA, 'SLICEWIDTH',
                            'ANNUALQA.MR.HEAD_GENERAL.tra.slice_width.IMA')],
            report_dir=pathlib.PurePath.joinpath(TEST_REPORT_DIR))

    def test_get_rods(self):
        arr = self.slice_width.single_dcm.pixel_array
        rods, _ = self.slice_width.get_rods(arr)
        # print("rods")
        # print(rods)
        for n in range(len(rods)):
            np.testing.assert_almost_equal(self.rods[n].centroid, rods[n].centroid, 3)

    def test_get_rod_distances(self):
        # From MATLAB Rods
        distances = self.slice_width.get_rod_distances(self.matlab_rods)
        np.testing.assert_almost_equal(distances, self.DISTANCES, 3)

    def test_get_rod_distortion_correction_coefficients(self):
        distances = self.slice_width.get_rod_distances(self.matlab_rods)
        # print("rod distortion correction coefficient")
        # print(hazen_slice_width.get_rod_distortion_correction_coefficients(distances[0], self.dcm.PixelSpacing[0]))
        dist_corr_coeff = self.slice_width.get_rod_distortion_correction_coefficients(
            distances[0])

        np.testing.assert_almost_equal(
            dist_corr_coeff['top'], self.DIST_CORR_COEFF['top'], 3)
        np.testing.assert_almost_equal(
            dist_corr_coeff['bottom'], self.DIST_CORR_COEFF['bottom'], 3)

    def test_rod_distortions(self):
        horz_dist, vert_dist = self.slice_width.get_rod_distances(self.matlab_rods)
        horizontal_distortion, vertical_distortion = self.slice_width.get_rod_distortions(horz_dist, vert_dist)
        # print("rod distortion")
        # print(horizontal_distortion, vertical_distortion)
        assert (round(horizontal_distortion, 2), round(vertical_distortion, 2)) == self.ROD_DIST

    def test_get_ramp_profiles(self):
        ramp_profiles = self.slice_width.get_ramp_profiles(
            self.slice_width.single_dcm.pixel_array, self.matlab_rods)
        bottom_profiles = ramp_profiles["bottom"]
        mean_bottom_profile = np.mean(bottom_profiles, axis=0).tolist()
        # print("bottom centre ramp profile")
        # print(ramp_profiles["bottom-centre"])
        # print("top centre ramp profile")
        # print(ramp_profiles["top-centre"])
        # print("mean bottom profile")
        # print(mean_bottom_profile)
        assert ramp_profiles["bottom-centre"] == self.BOTTOM_CENTRE
        assert ramp_profiles["top-centre"] == self.TOP_CENTRE
        assert mean_bottom_profile == self.MATLAB_PROFILE

    def test_baseline_correction(self):
        # matlab top 0.0215   -2.9668  602.4568
        # matlab bottom [0.0239, -2.9349,  694.9520]

        ramps = self.slice_width.get_ramp_profiles(
            self.slice_width.single_dcm.pixel_array, self.matlab_rods)

        top_mean_ramp = np.mean(ramps["top"], axis=0)
        top_coefficients = list(self.slice_width.baseline_correction(top_mean_ramp, sample_spacing=0.25)["f"])
        # print("top bline  corr coeff")
        # print(round(top_coefficients[0], 4))
        assert round(top_coefficients[0], 4) == self.BLINE_TOP

        bottom_mean_ramp = np.mean(ramps["bottom"], axis=0)
        bottom_coefficients = list(self.slice_width.baseline_correction(bottom_mean_ramp, sample_spacing=0.25)["f"])
        # print("bottom bline corr coeff")
        # print(round(bottom_coefficients[0], 4))
        assert round(bottom_coefficients[0], 4) == self.BLINE_BOT

    def test_trapezoid(self):
        """
        Notes
        -----
        variables from one iteration of the original matlab script
        n_ramp, n_plateau, n_left_baseline, n_right_baseline, plateau_amplitude =
        55.0000   58.0000  156.0000  153.0000 -136.6194 and fwhm 113
        """

        assert self.slice_width.trapezoid(55, 58, 156, 153, -136.6194)[1] == 113

    def test_get_initial_trapezoid_fit_and_coefficients(self):
        """
        Notes
        -----
        Trapezoid_Fit_Coefficients_Initial = 48.0000   56.0000  153.0000  172.0000 -116.4920
        fwhm = 104

        """
        sample_spacing = 0.25
        slice_thickness = self.slice_width.single_dcm.SliceThickness
        ramps = self.slice_width.get_ramp_profiles(
            self.slice_width.single_dcm.pixel_array, self.matlab_rods)
        top_mean_ramp = np.mean(ramps["top"], axis=0)
        bottom_mean_ramp = np.mean(ramps["bottom"], axis=0)
        ramps_baseline_corrected = {
            "top": self.slice_width.baseline_correction(top_mean_ramp, sample_spacing),
            "bottom": self.slice_width.baseline_correction(bottom_mean_ramp, sample_spacing)
        }

        trapezoid_fit, trapezoid_fit_coefficients = self.slice_width.get_initial_trapezoid_fit_and_coefficients(
            ramps_baseline_corrected["top"]["profile_corrected_interpolated"], slice_thickness)
        # print("trap fit coeff top initial")
        # print(trapezoid_fit_coefficients[:4])
        assert trapezoid_fit_coefficients[:4] == self.TRAP_FIT_COEFF_TOP

        trapezoid_fit, trapezoid_fit_coefficients = self.slice_width.get_initial_trapezoid_fit_and_coefficients(
            ramps_baseline_corrected["bottom"]["profile_corrected_interpolated"], slice_thickness)
        # print("trap fit coeff bottom initial")
        # print(trapezoid_fit_coefficients[:4])
        assert trapezoid_fit_coefficients[:4] == self.TRAP_FIT_COEFF_BOT

    def test_fit_trapezoid(self):
        sample_spacing = 0.25
        slice_thickness = self.slice_width.single_dcm.SliceThickness

        ramps = self.slice_width.get_ramp_profiles(
            self.slice_width.single_dcm.pixel_array, self.matlab_rods)

        top_mean_ramp = np.mean(ramps["top"], axis=0)
        bottom_mean_ramp = np.mean(ramps["bottom"], axis=0)
        ramps_baseline_corrected = {
            "top": self.slice_width.baseline_correction(top_mean_ramp, sample_spacing),
            "bottom": self.slice_width.baseline_correction(bottom_mean_ramp, sample_spacing)
        }
        trapezoid_fit_coefficients, baseline_fit_coefficients = self.slice_width.fit_trapezoid(
            profiles=ramps_baseline_corrected["top"], slice_thickness=slice_thickness)
        # print("top trap fit coeff")
        # print(trapezoid_fit_coefficients)

        # print("top bline fit coeff")
        # print(baseline_fit_coefficients)

        # check top profile first
        matlab_trapezoid_fit_coefficients = self.MATLAB_TRAP_FIT_COEFF
        matlab_baseline_fit_coefficients = self.MATLAB_BLINE_FIT_COEFF

        for idx, value in enumerate(trapezoid_fit_coefficients):
            assert abs(value - matlab_trapezoid_fit_coefficients[idx]) <= 5

        for idx, value in enumerate(matlab_baseline_fit_coefficients):
            assert abs(value - matlab_baseline_fit_coefficients[idx]) <= 5
        # residual error is 3735.6

        ## check bottom profile now
        trapezoid_fit_coefficients, baseline_fit_coefficients = self.slice_width.fit_trapezoid(
            profiles=ramps_baseline_corrected["bottom"], slice_thickness=slice_thickness)

        # print("bottom trap fit coeff")
        # print(trapezoid_fit_coefficients)
        # print("bottom bline fit coeff")
        # print(baseline_fit_coefficients)

        matlab_trapezoid_fit_coefficients = self.MATLAB_TRAP_FIT_COEFF_BOT
        matlab_baseline_fit_coefficients = self.MATLAB_BLINE_FIT_COEFF_BOT
        for idx, value in enumerate(trapezoid_fit_coefficients):
            assert abs(value - matlab_trapezoid_fit_coefficients[idx]) <= 5

        for idx, value in enumerate(matlab_baseline_fit_coefficients):
            assert abs(value - matlab_baseline_fit_coefficients[idx]) <= 5

    def test_slice_width(self):
        result = self.slice_width.run()
        slice_width_mm = result['measurement']['slice width mm']

        print("\ntest_slice_width.py::TestSliceWidth::test_slice_width")
        print("new_release_value:", slice_width_mm)
        print("fixed_value:", self.SW_MATLAB)

        assert abs(slice_width_mm - self.SW_MATLAB) < 0.1


class Test512Matrix(TestSliceWidth):
    SLICE_WIDTH_DATA = pathlib.Path(TEST_DATA_DIR / 'slicewidth')

    """
    Notes
    -----
    The rod indices are ordered as:
        789
        456
        123
    """
    matlab_rods = [Rod(134.19422395407386, 376.3297092734241),
                   Rod(255.5191606119133, 374.7890799840584),
                   Rod(376.45583336255567, 373.7479398253673),
                   Rod(133.73482492069846, 255.9592867481753),
                   Rod(254.49068286441252, 254.33374023796569),
                   Rod(375.15134878569734, 254.0979036468634),
                   Rod(133.07301997760993, 136.86070859809843),
                   Rod(253.53888071406627, 135.75358569280274),
                   Rod(374.00513210724, 135.12370483098044)]

    rods = [Rod(134.50715482125025, 376.39901305569873),
            Rod(255.46730860444976, 374.7645374009015),
            Rod(376.3508021809186, 373.6769436835617),
            Rod(133.58289022243835, 256.06611582258665),
            Rod(254.51562676421784, 254.61914241989848),
            Rod(375.3779965519839, 253.83655980142507),
            Rod(132.7380060479853, 137.19733440032368),
            Rod(253.45972063728271, 135.83081591213042),
            Rod(374.31705777525144, 135.3055747931858)]

    DISTANCES = ([242.275, 241.424, 240.938], [239.472, 239.044, 238.637])
    DIST_CORR_COEFF = {'top': 1.0049, 'bottom': 1.0077}
    ROD_DIST = (0.28, 0.17)

    TOP_CENTRE = 196
    BOTTOM_CENTRE = 316
    MATLAB_PROFILE = [7970.0, 7912.7, 7884.8, 7858.8, 7812.95, 7759.8, 7717.925, 7686.575, 7658.275, 7625.375, 7578.45,
                      7522.275, 7472.325, 7439.35, 7408.325, 7364.875, 7327.475, 7311.0, 7299.025, 7265.475, 7213.525,
                      7166.7, 7122.625, 7070.875, 7022.25, 6998.95, 6990.05, 6965.325, 6923.2, 6891.6, 6880.05,
                      6856.425, 6809.975, 6770.95, 6759.55, 6752.15, 6721.55, 6682.175, 6658.075, 6642.175, 6611.75,
                      6562.325, 6512.85, 6485.575, 6479.45, 6478.3, 6467.8, 6439.45, 6396.675, 6351.825, 6320.7,
                      6309.525, 6310.175, 6303.0, 6267.3, 6214.1, 6173.725, 6167.0, 6176.825, 6171.4, 6152.0, 6146.4,
                      6147.425, 6127.85, 6090.75, 6065.175, 6055.675, 6031.15, 5991.975, 5977.125, 5996.25, 5998.8,
                      5951.0, 5895.325, 5883.475, 5903.65, 5899.7, 5861.825, 5843.075, 5860.825, 5872.95, 5857.625,
                      5835.2, 5828.275, 5816.95, 5783.475, 5752.975, 5753.025, 5766.5, 5768.75, 5773.775, 5794.775,
                      5802.7, 5757.25, 5684.375, 5638.1, 5624.1, 5593.575, 5519.275, 5428.525, 5351.5, 5274.45,
                      5169.925, 5045.75, 4932.65, 4839.15, 4757.825, 4684.975, 4621.675, 4577.3, 4547.775, 4529.55,
                      4515.15, 4495.7, 4479.775, 4484.425, 4501.25, 4487.35, 4437.025, 4392.075, 4386.55, 4401.175,
                      4401.925, 4386.75, 4388.975, 4402.075, 4395.4, 4376.25, 4384.05, 4419.075, 4443.875, 4436.9,
                      4423.475, 4437.25, 4467.7, 4485.775, 4490.65, 4499.55, 4522.525, 4556.525, 4594.4, 4634.55,
                      4662.325, 4658.45, 4645.775, 4675.325, 4746.475, 4802.05, 4809.95, 4813.95, 4874.025, 4968.925,
                      5048.425, 5119.975, 5236.2, 5398.2, 5542.675, 5632.375, 5706.4, 5813.375, 5943.575, 6056.1,
                      6135.025, 6193.7, 6246.75, 6291.075, 6325.35, 6353.55, 6371.325, 6382.85, 6403.925, 6445.9,
                      6493.45, 6522.275, 6531.15, 6535.175, 6541.625, 6544.275, 6557.0, 6593.725, 6651.325, 6703.0,
                      6725.95, 6732.325, 6745.5, 6774.625, 6811.225, 6843.275, 6863.175, 6874.65, 6893.3, 6934.7,
                      6985.0, 7019.45, 7049.125, 7095.475, 7143.35, 7165.65, 7169.85, 7197.3, 7255.65, 7304.35,
                      7319.925, 7333.55, 7385.35, 7466.05, 7530.7, 7559.15, 7579.95, 7614.85, 7657.725, 7699.05,
                      7747.975, 7805.1, 7855.1, 7888.925, 7913.6, 7947.425, 7996.65, 8051.65, 8105.075, 8160.575,
                      8215.6, 8258.975, 8284.975, 8312.85, 8364.925, 8438.1, 8501.675, 8547.175, 8597.825, 8665.9,
                      8731.175, 8776.475, 8814.125, 8864.925, 8922.575, 8972.85, 9023.675, 9093.4, 9167.25, 9216.7]

    BLINE_TOP = 0.1686
    BLINE_BOT = 0.2072

    TRAP_FIT_COEFF_TOP = [47, 55, 398, 421]
    TRAP_FIT_COEFF_BOT = [47, 55, 442, 377]

    MATLAB_TRAP_FIT_COEFF = [81, 125, 331, 350, -1448.2731999293467]
    MATLAB_BLINE_FIT_COEFF = [0.16890305013478935, -39.18378191113812, 8256.387981752052]
    MATLAB_TRAP_FIT_COEFF_BOT = [80, 115, 382, 311, -1331.3180036312333]
    MATLAB_BLINE_FIT_COEFF_BOT = [0.20707462357475487, -44.91849057042588, 8014.213383072912]

    SW_MATLAB = 4.972852917690252

    def setUp(self):
        self.slice_width = SliceWidth(
            input_data=[os.path.join(self.SLICE_WIDTH_DATA, '512_matrix', '512_matrix')],
            report_dir=pathlib.PurePath.joinpath(TEST_REPORT_DIR))
        # self.file = str(TEST_DATA_DIR / 'slicewidth' / 'SLICEWIDTH' / '512_matrix')
        # self.dcm = pydicom.read_file(self.file)
