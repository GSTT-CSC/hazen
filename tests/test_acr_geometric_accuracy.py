"""Test suite for the Geometric Accuracy."""

# ruff: noqa: T201 S101

import logging
import pathlib
import unittest

import numpy as np
import scipy.ndimage as ndi
from hazenlib.tasks.acr_geometric_accuracy import (
    DIRECTIONS,
    ACRGeometricAccuracy,
    determine_phantom_orientation,
    measure_phantom,
    reorient,
    reorient_spacing,
)
from hazenlib.utils import get_dicom_files

from tests import TEST_DATA_DIR, TEST_REPORT_DIR

logger = logging.getLogger(__name__)

# Lengths are checked to this tolerance (mm). Values were verified against
# the report images: every end point sits on the phantom wall.
TOLERANCE = 0.05


###########
# Siemens #
###########


class TestACRGeometricAccuracySiemens(unittest.TestCase):
    """Geometric accuracy for legacy Siemens test data.

    The vertical diameter on this dataset used to be measured to the bottom
    of the wedge housing at the top of the phantom (184.1 mm on slice 1 and
    174.9 mm on slice 5) instead of to the phantom wall.
    """

    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "Siemens")
    L1 = (191.1, 190.5)
    L5 = (191.26, 190.46, 190.67, 191.03)
    distortion_metrics = (0.84, 1.26, 0.16)

    def setUp(self) -> None:
        """Set up the test object."""
        input_files = get_dicom_files(self.ACR_DATA)

        self.acr_geometric_accuracy_task = ACRGeometricAccuracy(
            input_data=input_files,
            report_dir=pathlib.PurePath.joinpath(TEST_REPORT_DIR),
        )

        self.dcm_1 = self.acr_geometric_accuracy_task.ACR_obj.slice_stack[0]
        self.dcm_5 = self.acr_geometric_accuracy_task.ACR_obj.slice_stack[4]

    def test_geometric_accuracy_slice_1(self) -> None:
        """Test Geometric Accuracy with Slice 1."""
        slice1_vals = np.array(
            self.acr_geometric_accuracy_task.get_geometric_accuracy(0),
        )

        logger.info(
            "Slice 1:\nnew_release: %s\nfixed value: %s",
            np.round(slice1_vals, 2),
            self.L1,
        )

        np.testing.assert_allclose(slice1_vals, self.L1, atol=TOLERANCE)

    def test_geometric_accuracy_slice_5(self) -> None:
        """Test Geometric Accuracy with Slice 5."""
        slice5_vals = np.array(
            self.acr_geometric_accuracy_task.get_geometric_accuracy(4),
        )

        logger.info(
            "Slice 5:\nnew_release: %s\nfixed value: %s",
            np.round(slice5_vals, 2),
            self.L5,
        )

        np.testing.assert_allclose(slice5_vals, self.L5, atol=TOLERANCE)

    def test_distortion_metrics(self) -> None:
        """Test distortion metrics."""
        metrics = np.array(
            self.acr_geometric_accuracy_task.get_distortion_metrics(
                self.L1 + self.L5,
            ),
        )
        metrics = np.round(metrics, 2)
        assert (metrics == self.distortion_metrics).all()


class TestACRGeometricAccuracySiemensAeraT1(TestACRGeometricAccuracySiemens):
    """Test Data for the Siemens_Aera_1.5T_T1 dataset."""

    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "Siemens_Aera_1.5T_T1")
    L1 = (189.42, 190.94)
    L5 = (189.36, 190.78, 190.01, 189.99)
    distortion_metrics = (0.08, 0.94, 0.32)


class TestACRGeometricAccuracySiemensAeraT2(
    TestACRGeometricAccuracySiemensAeraT1,
):
    """Test Data for the Siemens_Aera_1.5T_T2 dataset."""

    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "Siemens_Aera_1.5T_T2")
    L1 = (189.4, 190.92)
    L5 = (189.34, 190.76, 190.01, 189.99)
    distortion_metrics = (0.07, 0.92, 0.32)


class TestACRGeometricAccuracySiemensMagnetomSkyraT1(
    TestACRGeometricAccuracySiemens,
):
    """Test Data for the Siemens_MagnetomSkyra_3T_T1 dataset."""

    ACR_DATA = (
        pathlib.Path(TEST_DATA_DIR) / "acr" / "Siemens_MagnetomSkyra_3T_T1"
    )
    L1 = (191.81, 191.85)
    L5 = (191.35, 191.4, 191.39, 191.25)
    distortion_metrics = (1.51, 1.85, 0.12)


class TestACRGeometricAccuracySiemensMagnetomSkyraT2(
    TestACRGeometricAccuracySiemensMagnetomSkyraT1,
):
    """Test Data for the Siemens_MagnetomSkyra_3T_T2 dataset."""

    ACR_DATA = (
        pathlib.Path(TEST_DATA_DIR) / "acr" / "Siemens_MagnetomSkyra_3T_T2"
    )
    L1 = (191.82, 191.79)
    L5 = (191.34, 191.28, 191.34, 191.17)
    distortion_metrics = (1.46, 1.82, 0.13)


class TestACRGeometricAccuracySiemensSolaT1(TestACRGeometricAccuracySiemens):
    """Test Data for the Siemens_Sola_1.5T_T1 dataset."""

    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "Siemens_Sola_1.5T_T1")
    L1 = (189.93, 189.38)
    L5 = (190.06, 189.51, 190.1, 189.55)
    distortion_metrics = (-0.24, 0.62, 0.15)


class TestACRGeometricAccuracySiemensSolaT2(
    TestACRGeometricAccuracySiemensSolaT1,
):
    """Test Data for the Siemens_Sola_1.5T_T2 dataset."""

    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "Siemens_Sola_1.5T_T2")
    L1 = (189.96, 189.4)
    L5 = (190.06, 189.52, 190.11, 189.55)
    distortion_metrics = (-0.23, 0.6, 0.15)


class TestACRGeometricAccuracySiemensSolaFit(TestACRGeometricAccuracySiemens):
    """Test Data for the Siemens Sola Fit dataset."""

    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "SiemensSolaFit")
    L1 = (190.5, 190.46)
    L5 = (190.33, 190.28, 190.37, 190.0)
    distortion_metrics = (0.32, 0.5, 0.09)


######
# GE #
######


class TestACRGeometricAccuracyGE(TestACRGeometricAccuracySiemens):
    """Test Data for the GE dataset."""

    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "GE")
    L1 = (190.45, 190.25)
    L5 = (190.57, 190.39, 190.72, 190.07)
    distortion_metrics = (0.41, 0.72, 0.11)


class TestACRGeometricAccuracyGEArtistT1(TestACRGeometricAccuracyGE):
    """Test Data for the GE Artist 1.5T T1 dataset."""

    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "GE_Artist_1.5T_T1")
    L1 = (190.09, 190.2)
    L5 = (189.95, 190.1, 190.23, 190.19)
    distortion_metrics = (0.13, 0.23, 0.05)


class TestACRGeometricAccuracyGEArtistT2(TestACRGeometricAccuracyGEArtistT1):
    """Test Data for the GE Artist 1.5T T2 dataset."""

    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "GE_Artist_1.5T_T2")
    L1 = (190.12, 190.22)
    L5 = (189.97, 190.1, 190.24, 190.21)
    distortion_metrics = (0.14, 0.24, 0.05)


class TestACRGeometricAccuracyGEMR450WT1(TestACRGeometricAccuracyGE):
    """Test Data for the GE MR450W 1.5T T1 dataset."""

    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "GE_MR450W_1.5T_T1")
    L1 = (189.1, 190.41)
    L5 = (189.07, 190.4, 190.06, 189.82)
    distortion_metrics = (-0.19, 0.93, 0.29)


class TestACRGeometricAccuracyGEMR450WT2(TestACRGeometricAccuracyGEMR450WT1):
    """Test Data for the GE MR450W 1.5T T2 dataset."""

    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "GE_MR450W_1.5T_T2")
    L1 = (189.11, 190.42)
    L5 = (189.1, 190.39, 190.07, 189.8)
    distortion_metrics = (-0.19, 0.9, 0.29)


class TestACRGeometricAccuracyGESignaT1(TestACRGeometricAccuracyGE):
    """Test Data for the GE Signa 3T T1 dataset."""

    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "GE_Signa_3T_T1")
    L1 = (190.33, 191.16)
    L5 = (190.02, 190.63, 190.71, 191.08)
    distortion_metrics = (0.66, 1.16, 0.21)


class TestACRGeometricAccuracyGESignaT2(TestACRGeometricAccuracyGESignaT1):
    """Test Data for the GE Signa 3T T2 dataset."""

    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "GE_Signa_3T_T2")
    L1 = (190.3, 191.14)
    L5 = (189.99, 190.96, 190.61, 191.03)
    distortion_metrics = (0.67, 1.14, 0.22)


###########
# Philips #
###########


class TestACRGeometricAccuracyPhilipsAchieva(TestACRGeometricAccuracySiemens):
    """Test data for the Philips Achieva."""

    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "PhilipsAchieva")
    L1 = (189.96, 190.2)
    L5 = (190.15, 189.98, 190.42, 189.66)
    distortion_metrics = (0.06, 0.42, 0.12)


################
# Whole task   #
################


class TestACRGeometricAccuracyRun(unittest.TestCase):
    """The task reports every length plus the three distortion metrics."""

    def test_run(self) -> None:
        """Run the whole task on one dataset."""
        task = ACRGeometricAccuracy(
            input_data=get_dicom_files(TEST_DATA_DIR / "acr" / "Siemens"),
        )
        result = task.run()
        values = [m.value for m in result.measurements]
        assert len(values) == 9  # 2 + 4 lengths, 3 distortion metrics
        expected = (
            TestACRGeometricAccuracySiemens.L1
            + TestACRGeometricAccuracySiemens.L5
            + TestACRGeometricAccuracySiemens.distortion_metrics
        )
        np.testing.assert_allclose(values, expected, atol=TOLERANCE)


class TestACRGeometricAccuracyOrientation(unittest.TestCase):
    """A series exported rotated by a multiple of 90 degrees, or mirrored,
    must give the same lengths, with the direction labels still referring to
    the phantom in its standard view."""

    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "GE")
    expected = TestACRGeometricAccuracyGE.L1 + TestACRGeometricAccuracyGE.L5

    def test_rotated_and_mirrored(self) -> None:
        """All eight 90 degree rotations / mirrorings."""
        files = get_dicom_files(self.ACR_DATA)
        for k in range(4):
            for flip in (False, True):
                task = ACRGeometricAccuracy(input_data=files)
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
                assert task.orientation == ((-k) % 4 if not flip else k, flip)
                values = task.get_geometric_accuracy(
                    0,
                ) + task.get_geometric_accuracy(4)
                np.testing.assert_allclose(
                    values,
                    self.expected,
                    atol=TOLERANCE,
                    err_msg=f"k={k} flip={flip}",
                )

    def test_tilted(self) -> None:
        """A phantom tilted in-plane (here up to 20 degrees) on top of any
        90 degree rotation / mirroring. The orientation features are sampled
        in narrow bands, which a tilt of about 8 degrees used to defeat."""
        task = ACRGeometricAccuracy(input_data=get_dicom_files(self.ACR_DATA))
        img = task.ACR_obj.slice_stack[0].pixel_array.astype(float)
        spacing = task.pixel_spacing
        for tilt in (-20, -8, 8, 20):
            tilted = ndi.rotate(img, tilt, reshape=False, order=1)
            for k in range(4):
                for flip in (False, True):
                    found = determine_phantom_orientation(
                        reorient(tilted, k, flip), reorient_spacing(spacing, k)
                    )
                    assert found == ((-k) % 4 if not flip else k, flip), (
                        f"tilt={tilt} k={k} flip={flip}: {found}"
                    )

    def test_tilted_scan(self) -> None:
        """Scan of a phantom tilted by about 8 degrees, in the standard view."""
        task = ACRGeometricAccuracy(
            input_data=get_dicom_files(TEST_DATA_DIR / "acr" / "SiemensMTF")
        )
        assert task.orientation == (0, False)


class TestMeasurePhantomSynthetic(unittest.TestCase):
    """Measurements on a synthetic phantom of known size.

    The disc has a dark block and two notch cut-outs where the wall meets
    the top of the phantom, like the wedge housing on the ACR phantom. The
    vertical diameter must still be measured to the wall.
    """

    RADIUS_MM = 95.0

    def make_phantom(self, spacing, obstructed=True):
        row_sp, col_sp = spacing
        shape = (int(260 / row_sp), int(260 / col_sp))
        yy, xx = np.indices(shape, dtype=float)
        cy, cx = (shape[0] - 1) / 2, (shape[1] - 1) / 2
        y = (yy - cy) * row_sp
        x = (xx - cx) * col_sp
        # Supersampled edge for a smooth, sub-pixel accurate wall
        r = np.hypot(x, y)
        img = np.clip(self.RADIUS_MM - r + 0.5, 0, 1) * 1000.0
        if obstructed:
            housing = (np.abs(x) < 6) & (y < -self.RADIUS_MM + 9)
            notches = (np.hypot(np.abs(x) - 12, y + self.RADIUS_MM) < 6)
            img[housing | notches] = 0.0
        return ndi.gaussian_filter(img, 0.7) + 20.0

    def check(self, spacing):
        img = self.make_phantom(spacing)
        _, _, lengths = measure_phantom(img, spacing, tuple(DIRECTIONS))
        for name, d in lengths.items():
            assert abs(d["Distance"] - 2 * self.RADIUS_MM) < 0.2, (
                f"{name}: {d['Distance']:.2f} mm with spacing {spacing}"
            )

    def test_obstructed_top_wall(self) -> None:
        """Vertical diameter is measured to the wall, not to the block."""
        self.check((1.0, 1.0))

    def test_non_square_pixels(self) -> None:
        """Lengths are in mm, including the diagonals."""
        self.check((1.0, 0.8))
