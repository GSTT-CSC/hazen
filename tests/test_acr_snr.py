import copy
import os
import pathlib
import unittest

import numpy as np

from hazenlib.ACRObject import ACRObject
from hazenlib.logger import logger
from hazenlib.tasks.acr_snr import ACRSNR
from hazenlib.utils import dcmread, get_dicom_files

from tests import TEST_DATA_DIR, TEST_REPORT_DIR, transform_series

SNR_TOLERANCE = 0.01


class TestACRSNRGE(unittest.TestCase):
    """GE data with a 512 matrix (0.51 mm pixels).

    The ROIs are now 20 mm squares 40 mm from the centre whatever the pixel
    size; they used to be 20 x 20 pixels 40 pixels from the centre, i.e.
    10 mm squares 20 mm from the centre here, where the noise is higher
    (SNR was 39.98).
    """

    norm_factor = 57.12810400630368
    snr = 49.78

    def setUp(self):
        ACR_DATA_GE = pathlib.Path(TEST_DATA_DIR / "acr" / "GE")
        ge_files = get_dicom_files(ACR_DATA_GE)

        self.acr_snr_task = ACRSNR(
            input_data=ge_files,
            report_dir=pathlib.PurePath.joinpath(TEST_REPORT_DIR),
        )

        self.snr_dcm = self.acr_snr_task.ACR_obj.slice_stack[6]

    def test_normalisation_factor(self):
        SNR_factor = self.acr_snr_task.get_normalised_snr_factor(self.snr_dcm)
        self.assertAlmostEqual(SNR_factor, self.norm_factor, places=9)

    def test_snr_by_smoothing(self):
        snr, normalised = self.acr_snr_task.snr_by_smoothing(self.snr_dcm)
        self.assertAlmostEqual(snr, self.snr, delta=SNR_TOLERANCE)
        self.assertAlmostEqual(normalised, snr * self.norm_factor, places=6)


class TestACRSNRSiemens(TestACRSNRGE):
    norm_factor = 9.761711312090041
    snr = 348.85
    sub_snr = 76.59

    def setUp(self):
        ACR_DATA_SIEMENS = pathlib.Path(TEST_DATA_DIR / "acr" / "Siemens")
        siemens_files = get_dicom_files(ACR_DATA_SIEMENS)
        ACR_DATA_SIEMENS2 = pathlib.Path(TEST_DATA_DIR / "acr" / "Siemens2")

        self.acr_snr_task = ACRSNR(
            input_data=siemens_files,
            subtract=ACR_DATA_SIEMENS2,
            report_dir=pathlib.PurePath.joinpath(TEST_REPORT_DIR),
        )

        self.snr_dcm = self.acr_snr_task.ACR_obj.slice_stack[6]
        self.acr_obj2 = ACRObject(
            [
                dcmread(os.path.join(TEST_DATA_DIR, "acr", "Siemens2", f"{i}"))
                for i in os.listdir(
                    os.path.join(TEST_DATA_DIR, "acr", "Siemens2")
                )
            ]
        )
        self.snr_dcm2 = self.acr_obj2.slice_stack[6]

    def test_snr_by_subtraction(self):
        snr, _ = self.acr_snr_task.snr_by_subtraction(
            self.snr_dcm, self.snr_dcm2
        )
        rounded_snr = round(snr, 2)

        print(
            "\ntest_snr_by_subtraction.py::TestSnrBySubtraction::test_snr_by_subtraction"
        )
        print("new_release_value:", rounded_snr)
        print("fixed_value:", self.sub_snr)

        self.assertAlmostEqual(snr, self.sub_snr, delta=SNR_TOLERANCE)

    def test_run_subtraction(self):
        result = self.acr_snr_task.run()
        values = [m.value for m in result.measurements]
        assert len(values) == 2
        self.assertAlmostEqual(values[0], self.sub_snr, delta=SNR_TOLERANCE)

    def test_subtraction_with_second_series_rotated(self):
        """Each series is reoriented using its own slice 1."""
        transform_series(self.acr_obj2, 1, True)
        snr, _ = self.acr_snr_task.snr_by_subtraction(
            self.snr_dcm,
            self.acr_obj2.slice_stack[6],
            acr_obj2=self.acr_obj2,
        )
        self.assertAlmostEqual(snr, self.sub_snr, delta=SNR_TOLERANCE)


class TestACRSNRNoManufacturer(unittest.TestCase):
    """Data without a Manufacturer tag used to fail in the normalisation."""

    def test_snr(self):
        task = ACRSNR(
            input_data=get_dicom_files(
                TEST_DATA_DIR
                / "acr"
                / "PhilipsSliceThicknessLineProfileLocalMinimaIssue"
            ),
        )
        result = task.run()
        values = [m.value for m in result.measurements]
        assert len(values) == 2
        self.assertAlmostEqual(values[0], 188.07, delta=SNR_TOLERANCE)


class TestACRSNREncodingSteps(unittest.TestCase):
    """The normalisation uses the phase encoding direction to tell the
    frequency and phase encoding axes apart."""

    def setUp(self):
        task = ACRSNR(
            input_data=get_dicom_files(TEST_DATA_DIR / "acr" / "Siemens")
        )
        self.dcm = copy.deepcopy(task.ACR_obj.slice_stack[6])
        self.dcm.Rows, self.dcm.Columns = 256, 192

    def test_column_direction(self):
        self.dcm.InPlanePhaseEncodingDirection = "COL"
        assert ACRSNR.encoding_steps(self.dcm) == (192, 256)

    def test_row_direction(self):
        self.dcm.InPlanePhaseEncodingDirection = "ROW"
        assert ACRSNR.encoding_steps(self.dcm) == (256, 192)

    def test_unknown_direction(self):
        del self.dcm.InPlanePhaseEncodingDirection
        with self.assertLogs(logger, level="WARNING") as logs:
            steps = ACRSNR.encoding_steps(self.dcm)
        assert steps == (192, 256)
        assert any("direction not found" in m for m in logs.output)


class TestACRSNRSynthetic(unittest.TestCase):
    """SNR of a uniform disc (signal 1000) in Gaussian noise of SD 10.

    The SNR must be about 100 (unbiased over many noise seeds), with the
    ROIs fixed in mm whatever the pixel size.
    """

    def setUp(self):
        self.task = ACRSNR(
            input_data=get_dicom_files(TEST_DATA_DIR / "acr" / "Siemens"),
            orientation=(0, False),
        )
        self.rng = np.random.default_rng(1)

    def make_slice(self, spacing):
        row_sp, col_sp = spacing
        ny, nx = int(250 / row_sp), int(250 / col_sp)
        y = (np.arange(ny) - ny / 2) * row_sp
        x = (np.arange(nx) - nx / 2) * col_sp
        xx, yy = np.meshgrid(x, y)
        return np.where(np.hypot(xx, yy) < 95, 1000.0, 0.0)

    def smoothing_snr(self, spacing):
        self.task.ACR_obj.dx, self.task.ACR_obj.dy = spacing
        img = self.make_slice(spacing)
        img = img + self.rng.normal(0, 10, img.shape)
        snr, centre, *_ = self.task.smoothed_snr(img)
        masks = self.task.roi_masks(img.shape, centre)
        areas = [m.sum() * spacing[0] * spacing[1] for m in masks]
        return snr, areas

    def test_square_pixels(self):
        snr, areas = self.smoothing_snr((1.0, 1.0))
        assert abs(snr - 100) < 5  # about 3 SDs of the estimate
        assert all(abs(a - 400) <= 40 for a in areas)

    def test_half_millimetre_pixels(self):
        snr, areas = self.smoothing_snr((0.5, 0.5))
        assert abs(snr - 100) < 5  # about 3 SDs of the estimate
        assert all(abs(a - 400) <= 20 for a in areas)

    def test_non_square_pixels(self):
        snr, areas = self.smoothing_snr((1.0, 0.8))
        assert abs(snr - 100) < 5  # about 3 SDs of the estimate
        assert all(abs(a - 400) <= 40 for a in areas)
