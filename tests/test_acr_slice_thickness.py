"""Testing module for the ACR Slice thickness."""

# ruff: noqa: PT009

# Python imports
import logging
import pathlib
import unittest

import numpy as np
from scipy.special import erf

# Local imports
from hazenlib.tasks.acr_slice_thickness import ACRSliceThickness
from hazenlib.utils import get_dicom_files
from tests import TEST_DATA_DIR

logger = logging.getLogger(__name__)

THICKNESS_TOLERANCE = 0.02  # mm
RAMP_TOLERANCE = 0.1  # mm


class TestACRSliceThicknessSiemens(unittest.TestCase):
    """Base class for ACR Slice Thickness tests.

    Values were verified against the report images: the ramp ends sit at
    the half-maximum points of the ramp profiles.
    """

    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "Siemens")
    dz = 4.85
    ramps = {"top": 53.16, "bottom": 44.57}

    def setUp(self) -> None:
        """Set up each test."""
        input_files = get_dicom_files(self.ACR_DATA)

        self.acr_slice_thickness_task = ACRSliceThickness(
            input_data=input_files,
        )

        self.dcm = self.acr_slice_thickness_task.ACR_obj.slice_stack[0]
        self.results = self.acr_slice_thickness_task.get_slice_thickness(
            self.dcm,
        )

    def test_slice_thickness(self) -> None:
        """Test the slice thickness."""
        slice_thickness_val = round(self.results["thickness"], 2)

        logger.info(
            "test_slice_thickness.py::TestSliceThickness::test_slice_thickness\n"
            "new_release_value: %s\n"
            "fixed_value: %s",
            slice_thickness_val,
            self.dz,
        )
        self.assertAlmostEqual(
            slice_thickness_val, self.dz, delta=THICKNESS_TOLERANCE
        )

    def test_ramp_lengths(self) -> None:
        """Test the length (FWHM, in mm) of each ramp."""
        for name, expected in self.ramps.items():
            self.assertAlmostEqual(
                self.results["ramps"][name]["width"],
                expected,
                delta=RAMP_TOLERANCE,
                msg=f"{name} ramp",
            )


class TestACRSliceThicknessPhilipsAchieva(TestACRSliceThicknessSiemens):
    """Tests for the Philips Achieva data.

    The two ramps differ a lot in length here, which checks that "top"
    and "bottom" are labelled correctly (the upper ramp is the short one).
    """

    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "PhilipsAchieva")
    dz = 5.17
    ramps = {"top": 44.31, "bottom": 62.02}


class TestACRSliceThicknessPhilipsAchieva2(TestACRSliceThicknessSiemens):
    """Tests for the Philips Achieva 2 data."""

    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "PhilipsAchieva2")
    dz = 4.84
    ramps = {"top": 49.28, "bottom": 47.51}


class TestACRSliceThicknessSiemensSolaFit(TestACRSliceThicknessSiemens):
    """Tests for the Siemens Sola Fit."""

    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "SiemensSolaFit")
    dz = 5.17
    ramps = {"top": 49.95, "bottom": 53.68}


class TestACRSliceThicknessSiemensLargeSliceLocationDelta(
    TestACRSliceThicknessSiemens,
):
    """Tests for the Siemens Large Slice Location Delta."""

    ACR_DATA = pathlib.Path(
        TEST_DATA_DIR / "acr" / "SiemensLargeSliceLocationDelta",
    )
    dz = 4.47
    ramps = {"top": 44.82, "bottom": 44.52}


class TestACRSliceThicknessPhilips3TDStream(TestACRSliceThicknessSiemens):
    """Tests for the Philips 3TD Stream data."""

    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "Philips3TdStream")
    dz = 5.09
    ramps = {"top": 49.23, "bottom": 52.66}


class TestACRSliceThicknessPhilips3TDStream2(TestACRSliceThicknessSiemens):
    """Tests for the Philips 3TD 2 Stream data."""

    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "Philips3TdStream2")
    dz = 5.17
    ramps = {"top": 50.74, "bottom": 52.72}


class TestACRPhilipsSliceThicknessLineProfileLocalMinimaIssue(
    TestACRSliceThicknessSiemens,
):
    """Tests for the Philips Line Profile Local Minima Issue data."""

    ACR_DATA = pathlib.Path(
        TEST_DATA_DIR
        / "acr"
        / "PhilipsSliceThicknessLineProfileLocalMinimaIssue",
    )
    dz = 5.01
    ramps = {"top": 49.12, "bottom": 51.17}


class TestACRGESignaSliceThickness(TestACRSliceThicknessSiemens):
    """Tests for the GE Signa data."""

    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "GE_Signa_1.5T_T1")
    dz = 5.41
    ramps = {"top": 60.28, "bottom": 49.14}


class TestACRGESliceThicknessHalfMillimetrePixels(
    TestACRSliceThicknessSiemens,
):
    """GE data with 0.51 mm pixels.

    Lengths must be in mm, not pixels: the previous version reported
    9.99 mm here because it did not apply the pixel spacing.
    """

    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "GE")
    dz = 5.08
    ramps = {"top": 49.29, "bottom": 52.45}


class TestACRSliceThicknessSiemens2(TestACRSliceThicknessSiemens):
    """A slightly modified copy of the Siemens data must give the same
    result (previously 5.89 and 4.75 mm)."""

    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "Siemens2")
    dz = 4.84
    ramps = {"top": 53.10, "bottom": 44.48}


class TestACRSliceThicknessSiemensTilted(TestACRSliceThicknessSiemens):
    """Phantom tilted by about 8 degrees (the orientation was previously
    misdetected, so the insert appeared tilted by 80 degrees)."""

    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "SiemensMTF")
    dz = 5.09
    ramps = {"top": 49.38, "bottom": 52.56}


class TestACRSliceThicknessRun(unittest.TestCase):
    """The task reports the thickness and both ramp lengths in mm."""

    def test_run(self) -> None:
        """Run the whole task on one dataset."""
        task = ACRSliceThickness(
            input_data=get_dicom_files(TEST_DATA_DIR / "acr" / "Siemens"),
        )
        result = task.run()
        values = {
            (m.subtype, m.description): m.value for m in result.measurements
        }
        expected = TestACRSliceThicknessSiemens
        self.assertAlmostEqual(
            values[("slice width", "")], expected.dz, delta=THICKNESS_TOLERANCE
        )
        for name, length in expected.ramps.items():
            self.assertAlmostEqual(
                values[("Ramps", f"{name} ramp length")],
                length,
                delta=RAMP_TOLERANCE,
            )

    def test_localizer_is_rejected(self) -> None:
        """A sagittal localizer has no ramps: no value must be reported."""
        task = ACRSliceThickness(
            input_data=get_dicom_files(
                TEST_DATA_DIR / "acr" / "SiemensSolaFitLocalizer",
            ),
        )
        result = task.run()
        self.assertEqual(len(result.measurements), 0)


class TestACRSliceThicknessOrientation(unittest.TestCase):
    """A series exported rotated by a multiple of 90 degrees, or mirrored,
    must give the same result, with "top" and "bottom" still referring to
    the phantom in its standard view."""

    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "PhilipsAchieva")
    expected = TestACRSliceThicknessPhilipsAchieva

    def test_rotated_and_mirrored(self) -> None:
        """All eight 90 degree rotations / mirrorings."""
        files = get_dicom_files(self.ACR_DATA)
        for k in range(4):
            for flip in (False, True):
                task = ACRSliceThickness(input_data=files)
                for dcm in task.ACR_obj.slice_stack:
                    arr = np.rot90(dcm.pixel_array, k)
                    arr = np.ascontiguousarray(
                        np.fliplr(arr) if flip else arr,
                    )
                    dcm.Rows, dcm.Columns = arr.shape
                    dcm.PixelData = arr.tobytes()
                    dcm._pixel_array = None  # drop pydicom's cached array
                    dcm._pixel_id = {}
                if k % 2:
                    task.ACR_obj.dx, task.ACR_obj.dy = (
                        task.ACR_obj.dy,
                        task.ACR_obj.dx,
                    )
                results = task.get_slice_thickness(task.ACR_obj.slice_stack[0])
                self.assertAlmostEqual(
                    results["thickness"],
                    self.expected.dz,
                    delta=THICKNESS_TOLERANCE,
                    msg=f"k={k} flip={flip}",
                )
                for name, length in self.expected.ramps.items():
                    self.assertAlmostEqual(
                        results["ramps"][name]["width"],
                        length,
                        delta=RAMP_TOLERANCE,
                        msg=f"{name} ramp, k={k} flip={flip}",
                    )


class TestACRSliceThicknessSynthetic(unittest.TestCase):
    """Measurements on synthetic slice 1 images of known slice thickness.

    The insert is a dark 180 x 10 mm bar in a water disc. Each ramp is the
    slice profile stretched 10 times (a rectangle blurred by a Gaussian);
    right-left misalignment ``delta`` lengthens one ramp to
    10 T / (1 - delta) and shortens the other to 10 T / (1 + delta), so the
    ACR formula gives back exactly T.
    """

    def setUp(self) -> None:
        """Use a task object only for its geometry and spacing."""
        self.task = ACRSliceThickness(
            input_data=get_dicom_files(TEST_DATA_DIR / "acr" / "Siemens"),
        )
        self.task._orientation = (0, False)

    @staticmethod
    def make_slice(thickness, delta, spacing, tilt_deg=0.0, blur_mm=1.5):
        row_sp, col_sp = spacing
        ny, nx = int(250 / row_sp), int(250 / col_sp)
        y = (np.arange(ny) - ny / 2) * row_sp
        x = (np.arange(nx) - nx / 2) * col_sp
        xx, yy = np.meshgrid(x, y)
        a = np.deg2rad(tilt_deg)
        uu = xx * np.cos(a) + yy * np.sin(a)
        vv = -xx * np.sin(a) + yy * np.cos(a)
        img = np.where(np.hypot(xx, yy) < 95, 1000.0, 0.0)
        insert = (np.abs(uu) < 90) & (np.abs(vv) < 5)
        img[insert] = 20.0
        sigma = blur_mm * 10 / 2.355

        def ramp(length, shift):
            return 0.5 * (
                erf((uu - shift + length / 2) / (np.sqrt(2) * sigma))
                - erf((uu - shift - length / 2) / (np.sqrt(2) * sigma))
            )

        top = insert & (vv > -4.6) & (vv < -0.4)
        bottom = insert & (vv > 0.4) & (vv < 4.6)
        top_len = 10 * thickness / (1 + delta)
        bottom_len = 10 * thickness / (1 - delta)
        img[top] += 250 * ramp(top_len, -8)[top]
        img[bottom] += 250 * ramp(bottom_len, 8)[bottom]
        return img

    def measure(self, img, spacing):
        self.task.ACR_obj.dx, self.task.ACR_obj.dy = spacing
        insert = self.task.find_insert(img)
        top, bottom = self.task.measure_ramps(img, insert)
        return 0.2 * top.width * bottom.width / (top.width + bottom.width)

    def check(self, thickness, delta, spacing, tilt_deg=0.0):
        img = self.make_slice(thickness, delta, spacing, tilt_deg)
        self.assertAlmostEqual(
            self.measure(img, spacing),
            thickness,
            delta=THICKNESS_TOLERANCE,
            msg=f"T={thickness} delta={delta} spacing={spacing} "
            f"tilt={tilt_deg}",
        )

    def test_aligned(self) -> None:
        """5 mm slice, phantom aligned."""
        self.check(5.0, 0.0, (1.0, 1.0))

    def test_misaligned(self) -> None:
        """Right-left misalignment is corrected by the crossed ramps."""
        self.check(5.0, 0.1, (1.0, 1.0))

    def test_half_millimetre_pixels(self) -> None:
        """The result is in mm whatever the pixel size."""
        self.check(5.0, 0.1, (0.5, 0.5))

    def test_non_square_pixels(self) -> None:
        """Non-square pixels."""
        self.check(5.0, 0.1, (1.0, 0.8))

    def test_tilted(self) -> None:
        """A small residual tilt of the phantom."""
        self.check(5.0, 0.1, (0.98, 0.98), tilt_deg=9.0)

    def test_thick_slice(self) -> None:
        """A 10 mm slice, whose ramps fill most of the insert."""
        self.check(10.0, -0.08, (0.98, 0.98), tilt_deg=-6.0)
