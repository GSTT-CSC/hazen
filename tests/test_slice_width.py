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
        # top = [-595.173, -594.86, -594.225, -593.362, -592.369, -591.34, -590.371, -589.56, -589, -588.747, -588.686,
        #        -588.661, -588.518, -588.137, -587.554, -586.841, -586.071, -585.308, -584.581, -583.908, -583.311,
        #        -582.795, -582.311, -581.796, -581.187, -580.451, -579.677, -578.986, -578.499, -578.288, -578.24,
        #        -578.196, -577.996, -577.518, -576.788, -575.871, -574.83, -573.736, -572.673, -571.731, -571, -570.538,
        #        -570.272, -570.096, -569.906, -569.615, -569.211, -568.703, -568.098, -567.413, -566.7, -566.018,
        #        -565.426, -564.974, -564.659, -564.469, -564.39, -564.4, -564.43, -564.402, -564.24, -563.875, -563.274,
        #        -562.415, -561.277, -559.878, -558.41, -557.106, -556.199, -555.845, -555.89, -556.105, -556.257,
        #        -556.155, -555.761, -555.076, -554.101, -552.883, -551.655, -550.695, -550.282, -550.578, -551.277,
        #        -551.958, -552.198, -551.704, -550.703, -549.55, -548.6, -548.105, -547.895, -547.698, -547.239,
        #        -546.326, -545.092, -543.75, -542.513, -541.564, -540.957, -540.719, -540.874, -541.391, -542.018,
        #        -542.447, -542.37, -541.577, -540.238, -538.624, -537.003, -535.613, -534.571, -533.962, -533.872,
        #        -534.32, -535.061, -535.784, -536.176, -536.014, -535.419, -534.6, -533.767, -533.092, -532.598,
        #        -532.27, -532.094, -532.026, -531.912, -531.567, -530.807, -529.542, -528.059, -526.736, -525.956,
        #        -525.967, -526.506, -527.179, -527.591, -527.451, -526.883, -526.111, -525.362, -524.814, -524.46,
        #        -524.247, -524.119, -524.037, -524.02, -524.1, -524.312, -524.676, -525.173, -525.771, -526.441,
        #        -527.146, -525.359, -523.501, -521.52, -519.406, -517.335, -515.528, -514.207, -513.496, -513.125,
        #        -512.725, -511.93, -510.492, -508.644, -506.742, -505.139, -504.119, -503.678, -503.742, -504.234,
        #        -505.052, -505.978, -506.764, -507.165, -507.024, -506.543, -506.015, -505.733, -505.905, -506.414,
        #        -507.058, -507.636, -507.975, -508.026, -507.766, -507.175, -506.258, -505.125, -503.91, -502.75,
        #        -501.75, -500.886, -500.108, -499.362, -498.588, -497.7, -496.605, -495.209, -495.939, -496.43, -496.834,
        #        -497.307, -497.969, -498.808, -499.78, -500.84, -501.922, -502.87, -503.509, -503.66, -503.229, -502.44,
        #        -501.6, -501.016, -500.898, -501.076, -501.285, -501.258, -500.811, -500.09, -499.323, -498.736,
        #        -498.488, -498.469, -498.499, -498.399, -498.045, -497.534, -497.018, -496.649, -496.533, -496.587,
        #        -496.681, -496.685, -496.516, -496.273, -496.105, -496.157, -496.517, -497.031, -497.485, -497.665,
        #        -497.426, -496.897, -496.275, -495.759, -495.482, -495.315, -495.065, -494.54, -493.606, -492.38,
        #        -491.038, -489.756, -488.659, -490.144, -491.6, -492.894, -493.943, -494.871, -495.847, -497.046,
        #        -498.561, -500.171, -501.579, -502.485, -502.69, -502.39, -501.881, -501.459, -501.342, -501.441,
        #        -501.589, -501.62, -501.415, -501.044, -500.622, -500.267, -500.07, -500.034, -500.134, -500.349,
        #        -500.633, -500.847, -500.829, -500.418, -499.545, -498.515, -497.725, -497.573, -498.313, -499.631,
        #        -501.067, -502.164, -502.598, -502.597, -502.524, -502.74, -503.509, -504.682, -506.013, -507.253,
        #        -505.76, -504.182, -502.774, -501.788, -501.395, -501.427, -501.632, -501.759, -501.61, -501.209,
        #        -500.635, -499.966, -499.279, -498.649, -498.151, -497.859, -497.82, -497.976, -498.244, -498.537,
        #        -498.788, -498.981, -499.119, -499.202, -499.233, -499.21, -499.134, -499.003, -498.837, -498.729,
        #        -498.793, -499.14, -499.836, -500.75, -501.702, -502.514, -503.052, -503.378, -503.599, -503.823,
        #        -504.137, -504.555, -505.068, -505.668, -506.325, -506.92, -507.309, -507.349, -506.968, -506.369,
        #        -505.827, -505.616, -505.922, -506.577, -507.328, -507.92, -508.164, -508.138, -507.989, -507.859,
        #        -507.877, -508.107, -508.594, -509.384, -510.47, -511.622, -512.559, -512.996, -512.752, -512.058,
        #        -511.245, -510.643, -510.509, -510.795, -511.375, -512.127, -512.93, -513.693, -514.327, -514.746,
        #        -514.902, -514.904, -514.904, -515.052, -515.457, -516.074, -516.815, -517.593, -518.331, -518.981,
        #        -519.507, -519.871, -520.057, -520.135, -520.197, -520.335, -520.62, -521.048, -521.594, -522.234,
        #        -522.932, -523.597, -524.127, -524.42, -524.425, -524.294, -524.232, -524.442, -525.067, -525.999,
        #        -527.066, -528.1, -528.96, -529.634, -530.14, -530.494, -530.722, -530.879, -531.026, -531.224,
        #        -531.53, -531.98, -532.606, -533.44, -534.474, -535.545, -536.452, -536.992, -537.04, -536.775,
        #        -536.453, -536.33, -536.588, -537.112, -537.713, -538.204, -538.456, -538.589, -538.782, -539.214,
        #        -540.006, -541.034, -542.113, -543.06, -543.736, -544.178, -544.47, -544.693, -544.922, -545.196,
        #        -545.549, -546.011, -546.614, -547.387, -548.361, -549.565, -550.99, -552.467, -553.791, -554.756,
        #        -555.223, -555.335, -555.301, -555.332, -555.592, -556.054, -556.646, -557.295, -557.927, -558.469,
        #        -558.849, -558.993]

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
        assert trapezoid_fit_coefficients == []
        assert baseline_fit_coefficients == []

        trapezoid_fit_coefficients, baseline_fit_coefficients = hazen_slice_width.fit_trapezoid(
            profiles=ramps_baseline_corrected["bottom"], slice_thickness=slice_thickness)
        assert trapezoid_fit_coefficients == []
        assert baseline_fit_coefficients == []

    def test_slice_width(self):
        results = hazen_slice_width.main([self.file])

        assert results == 5.48
