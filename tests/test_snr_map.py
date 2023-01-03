import pathlib
import unittest
import pydicom
import numpy as np
import os.path
import matplotlib

import hazenlib.snr_map as hazen_snr_map
from tests import TEST_DATA_DIR, TEST_REPORT_DIR


class TestSnrMap(unittest.TestCase):
    siemens_1 = os.path.join(TEST_DATA_DIR, 'snr', 'Siemens', 'tra_250_2meas_1.IMA')
    RANDOM_DATA = np.array([-39.5905228, 9.56115628, 21.34564697,
                            65.52582681, 49.27813827, 14.395829,
                            77.88067287, 78.14606583, -55.1427819,
                            16.77284764, -45.36429801, -41.69026038])

    ROI_CORNERS_TEST = [np.array([114, 121]), np.array([74, 81]), np.array([154, 81]),
                        np.array([74, 161]), np.array([154, 161])]
    IMAGE_CENTRE_TEST = np.array([123.7456188, 131.21848254])

    def setUp(self):
        # run `smooth` function and save images for testing
        self.DCM = pydicom.read_file(self.siemens_1)

        original, smooth, noise = hazen_snr_map.smooth(self.DCM)
        self.images = {'original': original,
                       'smooth': smooth,
                       'noise': noise}

    def test_snr_value(self):
        dcms = [pydicom.dcmread(self.siemens_1)]
        results = hazen_snr_map.main(dcms, report_path=True, report_dir=pathlib.Path.joinpath(TEST_REPORT_DIR, 'SNRMap'))
        np.testing.assert_almost_equal(192.88188017908504,
                                       results[
                                           'snr_map_snr_seFoV250_2meas_slice5mm_tra_repeat_PSN_noDC_2_1_tra_250_2meas_1.IMA'])

    def test_sample_std(self):
        np.testing.assert_almost_equal(49.01842637544699,
                                       hazen_snr_map.sample_std(self.RANDOM_DATA))

    def test_smooth(self):
        # original, smooth, noise = hazen_snr_map.smooth(self.DCM)
        assert self.images['original'].cumsum().sum() == 1484467722691
        np.testing.assert_almost_equal(
            self.images['smooth'].cumsum().sum(), 1484468146211.5635)
        np.testing.assert_almost_equal(
            abs(self.images['noise']).sum(), 2147755.9753086423)

    def test_get_rois(self):
        # original, smooth, noise = hazen_snr_map.smooth(self.DCM)
        mask, roi_corners, image_centre = \
            hazen_snr_map.get_rois(self.images['smooth'], 40, 20)

        np.testing.assert_array_almost_equal(roi_corners, self.ROI_CORNERS_TEST)
        np.testing.assert_array_almost_equal(image_centre, self.IMAGE_CENTRE_TEST)
        assert mask.sum() == 29444

    def test_calc_snr(self):
        # original, smooth, noise = hazen_snr_map.smooth(self.DCM)
        snr = hazen_snr_map.calc_snr(
            self.images['original'], self.images['noise'], self.ROI_CORNERS_TEST, 20)
        np.testing.assert_approx_equal(snr, 192.8818801790859)

    def test_calc_snr_map(self):
        # original, smooth, noise = hazen_snr_map.smooth(self.DCM)
        snr_map = hazen_snr_map.calc_snr_map(
            self.images['original'], self.images['noise'], 20)
        np.testing.assert_almost_equal(snr_map.cumsum().sum(), 128077116718.40483)

    def test_plot_detailed(self):
        # Just check a valid figure handle is returned
        mask, roi_corners, image_centre = \
            hazen_snr_map.get_rois(self.images['smooth'], 40, 20)

        snr_map = hazen_snr_map.calc_snr_map(
            self.images['original'], self.images['noise'], 20)

        fig = hazen_snr_map.plot_detailed(
            self.DCM, self.images['original'], self.images['smooth'],
            self.images['noise'], snr_map, mask, image_centre, roi_corners,
            20, 999.99)

        assert isinstance(fig, matplotlib.figure.Figure)

    def test_plot_summary(self):
        # Just check a valid figure handle is returned
        snr_map = hazen_snr_map.calc_snr_map(
            self.images['original'], self.images['noise'], 20)
        fig = hazen_snr_map.plot_summary(
            self.images['original'], snr_map, self.ROI_CORNERS_TEST, 20)
        assert isinstance(fig, matplotlib.figure.Figure)


if __name__ == '__main__':
    unittest.main()
