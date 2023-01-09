import unittest
import pathlib
import pydicom
import os
from tests import TEST_DATA_DIR, TEST_REPORT_DIR
from hazenlib.tools import get_dicom_files
from hazenlib.tasks.snr import SNR


# Note all SNR tests assume 5mm slice thickness
class TestSnr(unittest.TestCase):
    # SIEMENS MR_VE11C
    # 1.5T

    SNR_DATA = pathlib.Path(TEST_DATA_DIR / 'snr')
    ORIENTATION = 'Transverse'

    OBJECT_CENTRE = (131, 122)  # note these coordinates are (x, y) ie. (COLUMN, ROW)
    SNR_NORM_FACTOR = 9.761711312090041  # checked manually
    IMAGE_SMOOTHED_SNR = 1874.81  # this value from MATLAB for tra_250_2meas_1.IMA, single image smoothed, normalised
    IMAGE_SUBTRACT_SNR = 2130.93  # this value from MATLAB for tra_250_2meas_1.IMA and tra_250_2meas_2.IMA, subtract method, normalised

    # setting +/- 3% range for SNR results
    UPPER_SMOOTHED_SNR = IMAGE_SMOOTHED_SNR * 1.02
    LOWER_SMOOTHED_SNR = IMAGE_SMOOTHED_SNR * 0.90

    UPPER_SUBTRACT_SNR = IMAGE_SUBTRACT_SNR * 1.02
    LOWER_SUBTRACT_SNR = IMAGE_SUBTRACT_SNR * 0.98

    def setUp(self):
        self.snr = SNR(data_paths=get_dicom_files(os.path.join(TEST_DATA_DIR, 'snr', 'Siemens'), sort=True),
                       report_dir=pathlib.PurePath.joinpath(TEST_REPORT_DIR))

    def test_get_object_centre(self):
        assert self.snr.get_object_centre(self.snr.data[0]) == self.OBJECT_CENTRE

    def test_image_snr(self):
        val = self.snr.run()
        self.assertTrue(self.LOWER_SMOOTHED_SNR <= val[self.snr.key(self.snr.data[0])][
            f"snr_smoothing_normalised_{self.snr.key(self.snr.data[0])}"] <= self.UPPER_SMOOTHED_SNR)
        self.assertTrue(self.LOWER_SUBTRACT_SNR <= val[self.snr.key(self.snr.data[0])][
            f"snr_subtraction_normalised_{self.snr.key(self.snr.data[0])}"] <= self.UPPER_SUBTRACT_SNR)

    def test_SNR_factor(self):
        SNR_factor = self.snr.get_normalised_snr_factor(self.snr.data[0])
        assert (SNR_factor) == self.SNR_NORM_FACTOR


class TestSnrPhilips(TestSnr):
    # PHILIPS_MR_53_1
    # 1.5T

    SNR_DATA = pathlib.Path(TEST_DATA_DIR / 'snr')
    ORIENTATION = 'Coronal'

    OBJECT_CENTRE = (127,
                     129)  # note these coordinates are (x, y) ie. (COLUMN, ROW) taken from Hazen, but checked in close proximity to Matlab
    SNR_NORM_FACTOR = 14.35183536242098  # value taken from Hazen, but checked manually.
    IMAGE_SMOOTHED_SNR = 5684.08  # this value from MATLAB for Philips_IM-0011-0005.dcm, single image smoothed, normalised
    IMAGE_SUBTRACT_SNR = 5472.44  # this value from MATLAB for Philips_IM-0011-0005.dcm and Philips_IM-0011-0006.dcm, subtract method, normalised

    # setting +/- 2% range for SNR results
    UPPER_SMOOTHED_SNR = IMAGE_SMOOTHED_SNR * 1.02
    LOWER_SMOOTHED_SNR = IMAGE_SMOOTHED_SNR * 0.98

    UPPER_SUBTRACT_SNR = IMAGE_SUBTRACT_SNR * 1.02
    LOWER_SUBTRACT_SNR = IMAGE_SUBTRACT_SNR * 0.98

    def setUp(self):
        # self.test_file = pydicom.read_file(str(self.SNR_DATA / 'Philips' / 'Philips_IM-0011-0005.dcm'), force=True)
        # self.test_file_2 = pydicom.read_file(str(self.SNR_DATA / 'Philips' / 'Philips_IM-0011-0006.dcm'), force=True)
        self.snr = SNR(data_paths=get_dicom_files(os.path.join(TEST_DATA_DIR, 'snr', 'Philips'), sort=True),
                       report_dir=pathlib.PurePath.joinpath(TEST_REPORT_DIR))

    def test_image_snr(self):
        val = self.snr.run()
        self.assertTrue(self.LOWER_SMOOTHED_SNR <= val[self.snr.key(self.snr.data[0])][
            f"snr_smoothing_normalised_{self.snr.key(self.snr.data[0])}"] <= self.UPPER_SMOOTHED_SNR)
        self.assertTrue(self.LOWER_SUBTRACT_SNR <= val[self.snr.key(self.snr.data[0])][
            f"snr_subtraction_normalised_{self.snr.key(self.snr.data[0])}"] <= self.UPPER_SUBTRACT_SNR)


class TestSnrGE(TestSnr):
    # GE ACCESSNET69-45B
    # 1.5T

    SNR_DATA = pathlib.Path(TEST_DATA_DIR / 'snr')
    ORIENTATION = 'Sagittal'

    OBJECT_CENTRE = (127,
                     129)  # note these coordinates are (x, y) ie. (COLUMN, ROW) taken from Hazen, but checked in close proximity to Matlab
    SNR_NORM_FACTOR = 8.254476647778304  # value taken from Hazen, but checked manually
    IMAGE_SMOOTHED_SNR = 1551.19  # this value from MATLAB for GE_IM-0003-0001.dcm, single image smoothed, normalised
    IMAGE_SUBTRACT_SNR = 1517.88  # this value from MATLAB for GE_IM-0003-0001.dcm and Philips_IM-0004-0001.dcm, subtract method, normalised

    # setting +/- 2% range for SNR results
    UPPER_SMOOTHED_SNR = IMAGE_SMOOTHED_SNR * 1.02
    LOWER_SMOOTHED_SNR = IMAGE_SMOOTHED_SNR * 0.90

    UPPER_SUBTRACT_SNR = IMAGE_SUBTRACT_SNR * 1.02
    LOWER_SUBTRACT_SNR = IMAGE_SUBTRACT_SNR * 0.98

    def setUp(self):
        # self.test_file = pydicom.read_file(str(self.SNR_DATA / 'GE' / 'IM-0003-0001.dcm'), force=True)
        # self.test_file_2 = pydicom.read_file(str(self.SNR_DATA / 'GE' / 'IM-0004-0001.dcm'), force=True)
        self.snr = SNR(data_paths=get_dicom_files(os.path.join(TEST_DATA_DIR, 'snr', 'GE'), sort=True),
                       report_dir=pathlib.PurePath.joinpath(TEST_REPORT_DIR))

    def test_image_snr(self):
        # val = self.snr.run(data=[self.test_file, self.test_file_2])
        val = self.snr.run()
        self.assertTrue(
            self.LOWER_SMOOTHED_SNR <= val[self.snr.key(self.snr.data[0])][
                f"snr_smoothing_normalised_{self.snr.key(self.snr.data[0])}"] <= self.UPPER_SMOOTHED_SNR)
        self.assertTrue(
            self.LOWER_SUBTRACT_SNR <= val[self.snr.key(self.snr.data[0])][
                f"snr_subtraction_normalised_{self.snr.key(self.snr.data[0])}"] <= self.UPPER_SUBTRACT_SNR)


class TestSnrThreshold(TestSnr):
    # SIEMENS VIDA
    # Example of shape detection failure

    SNR_DATA = pathlib.Path(TEST_DATA_DIR / 'snr_threshold')

    OBJECT_CENTRE = (129, 126)
    SNR_NORM_FACTOR = 13.537071812733949  # value taken from Hazen
    IMAGE_SMOOTHED_SNR = 5508.12  # TODO: get this value from Matlab, as in other tests (currently using Hazen value)
    IMAGE_SUBTRACT_SNR = 4809.91  # TODO: get this value from Matlab, as in other tests (currently using Hazen value)

    # setting +/- 2% range for SNR results
    UPPER_SMOOTHED_SNR = IMAGE_SMOOTHED_SNR * 1.02
    LOWER_SMOOTHED_SNR = IMAGE_SMOOTHED_SNR * 0.84

    UPPER_SUBTRACT_SNR = IMAGE_SUBTRACT_SNR * 1.02
    LOWER_SUBTRACT_SNR = IMAGE_SUBTRACT_SNR * 0.98

    def setUp(self):
        # self.test_file = pydicom.read_file(str(self.SNR_DATA / 'VIDA' / 'HC_SNR_SAG_1.dcm'), force=True)
        # self.test_file_2 = pydicom.read_file(str(self.SNR_DATA / 'VIDA' / 'HC_SNR_SAG_2.dcm'), force=True)
        self.snr = SNR(data_paths=get_dicom_files(os.path.join(TEST_DATA_DIR, 'snr_threshold', 'VIDA'), sort=True),
                       report_dir=pathlib.PurePath.joinpath(TEST_REPORT_DIR))

    def test_get_object_centre(self):
        assert self.snr.get_object_centre(self.snr.data[0]) == self.OBJECT_CENTRE

    def test_image_snr(self):
        val = self.snr.run()
        self.assertTrue(self.LOWER_SMOOTHED_SNR <= val[self.snr.key(self.snr.data[0])][
            f"snr_smoothing_normalised_{self.snr.key(self.snr.data[0])}"] <= self.UPPER_SMOOTHED_SNR)
        self.assertTrue(self.LOWER_SUBTRACT_SNR <= val[self.snr.key(self.snr.data[0])][
            f"snr_subtraction_normalised_{self.snr.key(self.snr.data[0])}"] <= self.UPPER_SUBTRACT_SNR)
