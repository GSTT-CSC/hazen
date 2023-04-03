import pathlib
import unittest
import pydicom
import numpy as np
import os.path
import matplotlib

from hazenlib.tasks.snr_map import SNRMap
from tests import TEST_DATA_DIR, TEST_REPORT_DIR


class TestSnrMap(unittest.TestCase):
    siemens_1 = os.path.join(TEST_DATA_DIR, 'snr', 'Siemens', 'tra_250_2meas_1.IMA')

    ROI_CORNERS_TEST = [np.array([114, 121]), np.array([74, 81]), np.array([154, 81]),
                        np.array([74, 161]), np.array([154, 161])]
    IMAGE_CENTRE_TEST = np.array([123.7456188, 131.21848254])

    def __init__(self, methodName: str = ...):
        super().__init__(methodName)
        self.images = None

    def setUp(self):
        dcms = [self.siemens_1]  # Test on single SNR image
        self.snr_map = SNRMap(data_paths=dcms, report=True)
        self.results = self.snr_map.run()

    def test_snr_value(self):
        np.testing.assert_almost_equal(192.88188017908504,
                                       self.results['SNRMap_seFoV250_2meas_slice5mm_tra_repeat_PSN_noDC_2_1'])

    def test_smooth(self):
        np.testing.assert_almost_equal(
            self.snr_map.original_image.cumsum().sum(), 1484467722691)
        np.testing.assert_almost_equal(
            self.snr_map.smooth_image.cumsum().sum(), 1484468146211.5635)
        np.testing.assert_almost_equal(
            abs(self.snr_map.noise_image).sum(), 2147755.9753086423)

    def test_get_rois(self):
        np.testing.assert_array_almost_equal(
            self.snr_map.roi_corners, self.ROI_CORNERS_TEST)
        np.testing.assert_array_almost_equal(
            self.snr_map.image_centre, self.IMAGE_CENTRE_TEST)
        assert self.snr_map.mask.sum() == 29444

    def test_calc_snr(self):
        np.testing.assert_approx_equal(
            self.snr_map.snr, 192.8818801790859)

    def test_calc_snr_map(self):
        np.testing.assert_almost_equal(
            self.snr_map.snr_map.cumsum().sum(), 128077116718.40483)

    def test_plot_detailed(self):
        # Just check a valid figure handle is returned
        fig = self.snr_map.plot_detailed()
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_plot_summary(self):
        # Just check a valid figure handle is returned
        fig = self.snr_map.plot_summary()
        assert isinstance(fig, matplotlib.figure.Figure)


if __name__ == '__main__':
    unittest.main()
