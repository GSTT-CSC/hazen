import os
import unittest
import pathlib
import pydicom

from hazenlib.utils import get_dicom_files
from hazenlib.tasks.acr_snr import ACRSNR
from hazenlib.ACRObject import ACRObject
from tests import TEST_DATA_DIR, TEST_REPORT_DIR


class TestACRSNRGE(unittest.TestCase):
    norm_factor = 57.12810400630368
    snr = 39.76

    def setUp(self):
        ACR_DATA_GE = pathlib.Path(TEST_DATA_DIR / "acr" / "GE")
        ge_files = get_dicom_files(ACR_DATA_GE)

        self.acr_snr_task = ACRSNR(
            input_data=ge_files, report_dir=pathlib.PurePath.joinpath(TEST_REPORT_DIR)
        )

        self.snr_dcm = self.acr_snr_task.ACR_obj.slice_stack[6]

    def test_normalisation_factor(self):
        SNR_factor = self.acr_snr_task.get_normalised_snr_factor(self.snr_dcm)
        assert SNR_factor == self.norm_factor

    def test_snr_by_smoothing(self):
        snr, _ = self.acr_snr_task.snr_by_smoothing(self.snr_dcm)
        assert round(snr, 2) == self.snr


class TestACRSNRSiemens(TestACRSNRGE):
    norm_factor = 9.761711312090041
    snr = 351.22
    sub_snr = 76.06

    def setUp(self):
        ACR_DATA_SIEMENS = pathlib.Path(TEST_DATA_DIR / "acr" / "Siemens")
        siemens_files = get_dicom_files(ACR_DATA_SIEMENS)
        ACR_DATA_SIEMENS2 = pathlib.Path(TEST_DATA_DIR / "acr" / "Siemens")
        # siemens_files2 = get_dicom_files(ACR_DATA_SIEMENS2)

        self.acr_snr_task = ACRSNR(
            input_data=siemens_files,
            subtract=ACR_DATA_SIEMENS2,
            report_dir=pathlib.PurePath.joinpath(TEST_REPORT_DIR),
        )

        self.snr_dcm = self.acr_snr_task.ACR_obj.slice_stack[6]
        self.snr_dcm2 = ACRObject(
            [
                pydicom.read_file(
                    os.path.join(TEST_DATA_DIR, "acr", "Siemens2", f"{i}")
                )
                for i in os.listdir(os.path.join(TEST_DATA_DIR, "acr", "Siemens2"))
            ]
        ).slice_stack[6]

    def test_snr_by_subtraction(self):
        snr, _ = self.acr_snr_task.snr_by_subtraction(self.snr_dcm, self.snr_dcm2)
        rounded_snr = round(snr, 2)

        print(
            "\ntest_snr_by_subtraction.py::TestSnrBySubtraction::test_snr_by_subtraction"
        )
        print("new_release_value:", rounded_snr)
        print("fixed_value:", self.sub_snr)

        assert rounded_snr == self.sub_snr
