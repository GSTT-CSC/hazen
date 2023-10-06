import unittest
import numpy as np
import os.path
import matplotlib

from hazenlib.tasks.snr_map import SNRMap
from tests import TEST_DATA_DIR, TEST_REPORT_DIR


class TestSnrMap(unittest.TestCase):
    siemens_1 = [os.path.join(TEST_DATA_DIR, 'snr', 'Siemens', 'tra_250_2meas_1.IMA')]

    ROI_CORNERS_TEST = [np.array([114, 121]), np.array([74, 81]), np.array([154, 81]),
                        np.array([74, 161]), np.array([154, 161])]
    IMAGE_CENTRE_TEST = np.array([123.7456188, 131.21848254])

    def __init__(self, methodName: str = ...):
        super().__init__(methodName)

    def setUp(self):
        self.snr_map_task = SNRMap(input_data=self.siemens_1, report=True)
        self.results = self.snr_map_task.run()
        self.original, self.smoothed, self.noise = self.snr_map_task.smooth(
            dcm=self.snr_map_task.single_dcm,
            kernel=self.snr_map_task.kernel_len)
        self.image_centre, self.roi_corners = self.snr_map_task.get_rois(self.smoothed)
        self.snr = self.snr_map_task.calc_snr(self.original, self.noise, self.roi_corners)
        self.snr_map = self.snr_map_task.calc_snr_map(self.original, self.noise)
        self.detailed_fig = self.snr_map_task.plot_detailed(self.original, self.smoothed, self.noise,
                    self.snr, self.snr_map, self.image_centre, self.roi_corners)
        self.summary_fig = self.snr_map_task.plot_summary(self.snr_map, self.original, self.roi_corners)


    def test_snr_value(self):
        np.testing.assert_almost_equal(
            192.88188017908504,
            self.results['measurement']['snr by smoothing'],
            2)

    def test_smooth(self):
        np.testing.assert_almost_equal(
            self.original.cumsum().sum(), 1484467722691)
        np.testing.assert_almost_equal(
            self.smoothed.cumsum().sum(), 1484468146211.5635)
        np.testing.assert_almost_equal(
            abs(self.noise).sum(), 2147755.9753086423)

    def test_get_rois(self):
        np.testing.assert_array_almost_equal(
            self.roi_corners, self.ROI_CORNERS_TEST)
        np.testing.assert_array_almost_equal(
            self.image_centre, self.IMAGE_CENTRE_TEST)
        assert self.snr_map_task.mask.sum() == 29444

    def test_calc_snr(self):
        np.testing.assert_approx_equal(
            self.snr, 192.8818801790859)

    def test_calc_snr_map(self):
        snr_map_cumsum = self.snr_map.cumsum().sum()

        print("\ntest_calc_snr_map.py::TestCalcSnrMap::test_calc_snr_map")
        print("new_release_value:", snr_map_cumsum)
        print("fixed_value:", 128077116718.40483)

        np.testing.assert_almost_equal(snr_map_cumsum, 128077116718.40483)

    def test_plot_detailed(self):
        # Just check a valid figure handle is returned
        assert isinstance(self.detailed_fig, matplotlib.figure.Figure)

    def test_plot_summary(self):
        # Just check a valid figure handle is returned
        assert isinstance(self.summary_fig, matplotlib.figure.Figure)


if __name__ == '__main__':
    unittest.main()
