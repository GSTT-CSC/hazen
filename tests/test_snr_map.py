import unittest
import pydicom
import numpy as np
import os.path

import hazenlib.snr_map as hazen_snr_map
from tests import TEST_DATA_DIR


class TestSnrMap(unittest.TestCase):

    siemens_1 = os.path.join(TEST_DATA_DIR, 'snr', r'tra_250_2meas_1.ima')
    siemens_2 = os.path.join(TEST_DATA_DIR, 'snr', 'tra_250_2meas_1.ima')
    siemens_3 = os.path.join(TEST_DATA_DIR, 'snr', r'tra_250_2meas_1.IMA')


    def test_snr_value(self):

        dcms = [pydicom.dcmread(self.siemens_1)]
        results = hazen_snr_map.main(dcms, report_path=True)
        np.testing.assert_almost_equal(192.88188017908504,
            results['snr_map_snr_seFoV250_2meas_slice5mm_tra_repeat_PSN_noDC_2_1_tra_250_2meas_1.ima'])

    def test_snr_value2(self):

        dcms = [pydicom.dcmread(self.siemens_2)]
        results = hazen_snr_map.main(dcms, report_path=True)
        np.testing.assert_almost_equal(192.88188017908504,
            results['snr_map_snr_seFoV250_2meas_slice5mm_tra_repeat_PSN_noDC_2_1_tra_250_2meas_1.ima'])

    def test_snr_value3(self):

        dcms = [pydicom.dcmread(self.siemens_3)]
        results = hazen_snr_map.main(dcms, report_path=True)
        np.testing.assert_almost_equal(192.88188017908504,
            results['snr_map_snr_seFoV250_2meas_slice5mm_tra_repeat_PSN_noDC_2_1_tra_250_2meas_1.IMA'])


if __name__ == '__main__':
    unittest.main()
