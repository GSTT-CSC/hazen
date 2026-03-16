"""Test suite for the Geometric Accuracy."""

# ruff: noqa: T201 S101

import logging
import pathlib
import unittest

import numpy as np
from hazenlib.tasks.acr_geometric_accuracy import ACRGeometricAccuracy
from hazenlib.utils import get_dicom_files

from tests import TEST_DATA_DIR, TEST_REPORT_DIR

logger = logging.getLogger(__name__)


###########
# Siemens #
###########


class TestACRGeometricAccuracySiemens(unittest.TestCase):
    """Geometric accuracy for legacy Siemens test data."""

    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "Siemens")
    L1 = (191.41, 187.5)
    L5 = (191.41, 187.5, 191.41, 191.41)
    distortion_metrics = (0.11, 2.5, 0.97)

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
        slice1_vals = self.acr_geometric_accuracy_task.get_geometric_accuracy(
            0
        )
        slice1_vals = np.round(slice1_vals, 2)

        logger.info(
            "Slice 1:\nnew_release: %s\nfixed value: %s",
            slice1_vals,
            self.L1,
        )

        assert (slice1_vals == self.L1).all()

    def test_geometric_accuracy_slice_5(self) -> None:
        """Test Geometric Accuracy with Slice 5."""
        slice5_vals = np.array(
            self.acr_geometric_accuracy_task.get_geometric_accuracy(4),
        )

        slice5_vals = np.round(slice5_vals, 2)

        logger.info(
            "Slice 5:\nnew_release: %s\nfixed value: %s",
            slice5_vals,
            self.L5,
        )

        assert (slice5_vals == self.L5).all()

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
    L1 = (189.45, 190.43)
    L5 = (190.43, 191.41, 189.45, 189.45)
    distortion_metrics = (0.1, 1.41, 0.38)


class TestACRGeometricAccuracySiemensAeraT2(
    TestACRGeometricAccuracySiemensAeraT1,
):
    """Test Data for the Siemens_Aera_1.5T_T2 dataset."""

    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "Siemens_Aera_1.5T_T2")


class TestACRGeometricAccuracySiemensMagnetomSkyraT1(
    TestACRGeometricAccuracySiemens,
):
    """Test Data for the Siemens_MagnetomSkyra_3T_T1 dataset."""

    ACR_DATA = (
        pathlib.Path(TEST_DATA_DIR) / "acr" / "Siemens_MagnetomSkyra_3T_T1"
    )
    L1 = (192.38, 192.38)
    L5 = (191.41, 191.41, 191.41, 191.41)
    distortion_metrics = (1.73, 2.38, 0.24)


class TestACRGeometricAccuracySiemensMagnetomSkyraT2(
    TestACRGeometricAccuracySiemensMagnetomSkyraT1,
):
    """Test Data for the Siemens_MagnetomSkyra_3T_T2 dataset."""

    ACR_DATA = (
        pathlib.Path(TEST_DATA_DIR) / "acr" / "Siemens_MagnetomSkyra_3T_T2"
    )


class TestACRGeometricAccuracySiemensSolaT1(TestACRGeometricAccuracySiemens):
    """Test Data for the Siemens_Sola_1.5T_T1 dataset."""

    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "Siemens_Sola_1.5T_T1")
    L1 = (190.43, 189.45)
    L5 = (189.45, 190.43, 191.41, 189.45)
    distortion_metrics = (0.1, 1.41, 0.38)


class TestACRGeometricAccuracySiemensSolaT2(
    TestACRGeometricAccuracySiemensSolaT1,
):
    """Test Data for the Siemens_Sola_1.5T_T2 dataset."""

    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "Siemens_Sola_1.5T_T2")
    L5 = (189.45, 190.43, 189.45, 189.45)
    distortion_metrics = (-0.22, 0.55, 0.24)


class TestACRGeometricAccuracySiemensSolaFit(TestACRGeometricAccuracySiemens):
    """Test Data for the Siemens Sola Fit dataset."""

    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "SiemensSolaFit")
    L1 = (190.43, 190.43)
    L5 = (189.45, 190.43, 191.41, 191.41)
    distortion_metrics = (0.59, 1.41, 0.35)


######
# GE #
######


class TestACRGeometricAccuracyGE(TestACRGeometricAccuracySiemens):
    """Test Data for the GE dataset."""

    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "GE")
    L1 = (190.93, 188.9)
    L5 = (190.42, 189.41, 190.43, 189.41)
    distortion_metrics = (-0.08, 1.1, 0.38)


class TestACRGeometricAccuracyGEArtistT1(TestACRGeometricAccuracyGE):
    """Test Data for the GE Artist 1.5T T1 dataset."""

    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "GE_Artist_1.5T_T1")
    L1 = (190.44, 190.44)
    L5 = (190.44, 190.44, 189.46, 191.41)
    distortion_metrics = (0.44, 1.41, 0.30)


class TestACRGeometricAccuracyGEArtistT2(TestACRGeometricAccuracyGEArtistT1):
    """Test Data for the GE Artist 1.5T T2 dataset."""

    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "GE_Artist_1.5T_T2")
    L5 = (190.44, 190.44, 189.46, 190.44)
    distortion_metrics = (0.28, 0.54, 0.19)


class TestACRGeometricAccuracyGEMR450WT1(TestACRGeometricAccuracyGE):
    """Test Data for the GE MR450W 1.5T T1 dataset."""

    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "GE_MR450W_1.5T_T1")
    L1 = (188.48, 188.48)
    L5 = (188.48, 188.48, 190.44, 191.41)
    distortion_metrics = (-0.71, 1.52, 0.63)


class TestACRGeometricAccuracyGEMR450WT2(TestACRGeometricAccuracyGEMR450WT1):
    """Test Data for the GE MR450W 1.5T T2 dataset."""

    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "GE_MR450W_1.5T_T2")
    L1 = (189.46, 188.48)
    L5 = (188.48, 188.48, 191.41, 189.46)
    distortion_metrics = (-0.71, 1.52, 0.55)


class TestACRGeometricAccuracyGESignaT1(TestACRGeometricAccuracyGE):
    """Test Data for the GE Signa 3T T1 dataset."""

    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "GE_Signa_3T_T1")
    L1 = (189.46, 189.46)
    L5 = (190.44, 188.48, 191.41, 191.41)
    distortion_metrics = (0.11, 1.52, 0.57)


class TestACRGeometricAccuracyGESignaT2(TestACRGeometricAccuracyGESignaT1):
    """Test Data for the GE Signa 3T T2 dataset."""

    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "GE_Signa_3T_T2")
    L1 = (189.46, 189.46)
    L5 = (190.44, 188.48, 191.41, 191.41)
    distortion_metrics = (0.11, 1.52, 0.57)


###########
# Philips #
###########


class TestACRGeometricAccuracyPhilipsAchieva(TestACRGeometricAccuracySiemens):
    """Test data for the Philips Achieva."""

    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "PhilipsAchieva")
    L1 = 190.43, 189.45
    L5 = 190.43, 189.45, 189.45, 189.45
    distortion_metrics = (-0.22, 0.55, 0.24)
