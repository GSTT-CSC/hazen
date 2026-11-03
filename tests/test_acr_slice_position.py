# ruff: noqa: S101
import pathlib
import unittest

import numpy as np
import scipy.ndimage
from hazenlib.tasks.acr_slice_position import ACRSlicePosition
from hazenlib.utils import get_dicom_files
from tests import TEST_DATA_DIR


def set_pixels(dcm, arr):
    """Replace the pixel data of a pydicom dataset in place.

    ``arr`` must already have the dataset's pixel dtype and shape.
    """
    dcm.PixelData = np.ascontiguousarray(arr).tobytes()
    # Drop pydicom's cached array so the new data is decoded.
    dcm._pixel_array = None
    dcm._pixel_id = {}


class TestACRSlicePositionSiemens(unittest.TestCase):
    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "Siemens")
    # Bar length difference (right - left, mm) for slices 1 and 11,
    # verified visually against the report images.
    dL = -0.15, -1.91
    TOLERANCE = 0.05
    MAX_TILT = 5  # deg, in-plane tilt of the phantom in the test data

    def setUp(self):
        input_files = get_dicom_files(self.ACR_DATA)
        self.task = ACRSlicePosition(input_data=input_files)
        self.dcms = (
            self.task.ACR_obj.slice_stack[0],
            self.task.ACR_obj.slice_stack[-1],
        )

    def geometry(self, dcm):
        _, img, _ = self.task.ACR_obj.get_presentation_pixels(dcm)
        img = np.asarray(img, dtype=float)
        centre, radius = self.task.ACR_obj.find_phantom_center(
            img, self.task.ACR_obj.dx, self.task.ACR_obj.dy
        )
        return self.task.find_wedges(img, centre, radius)

    def test_find_wedges(self):
        for dcm in self.dcms:
            geo = self.geometry(dcm)
            # The test data are acquired roughly upright, with the wedges
            # at the top of the image (270 deg from +x, y down).
            assert abs(np.rad2deg(geo.theta) % 360 - 270) < self.MAX_TILT
            # Physical size of the wedge pair on the ACR large phantom
            assert 10 < geo.width < 14
            assert abs(geo.u_mid) < 5
            assert 90 < geo.radius < 100

    def test_slice_position(self):
        for dcm, expected in zip(self.dcms, self.dL, strict=True):
            value = self.task.get_slice_position(dcm)
            print(
                f"\n{type(self).__name__}: new_release_value {value:.2f}, "
                f"fixed_value {expected}"
            )
            assert abs(value - expected) < self.TOLERANCE

    def test_run(self):
        result = self.task.run()
        values = [m.value for m in result.measurements]
        assert len(values) == 2
        for value, expected in zip(values, self.dL, strict=True):
            assert abs(value - expected) < self.TOLERANCE


class TestACRSlicePositionGE(TestACRSlicePositionSiemens):
    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "GE")
    dL = 0.37, 0.34


class TestACRSlicePositionPhilipsAchieva(TestACRSlicePositionSiemens):
    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "PhilipsAchieva")
    dL = 1.14, -3.39


class TestACRSlicePositionSiemensSolaFit(TestACRSlicePositionSiemens):
    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "SiemensSolaFit")
    dL = -0.02, -1.28


class TestACRSlicePositionSiemensLargeSliceLocationDelta(
    TestACRSlicePositionSiemens
):
    ACR_DATA = pathlib.Path(
        TEST_DATA_DIR / "acr" / "SiemensLargeSliceLocationDelta"
    )
    dL = -6.82, -6.96


class TestACRSlicePositionPhilips3TDStream(TestACRSlicePositionSiemens):
    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "Philips3TdStream")
    dL = 1.65, -2.29


class TestACRSlicePositionPhilips3TDStream2(TestACRSlicePositionSiemens):
    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "Philips3TdStream2")
    dL = 2.84, -1.02


# triggers mispositioning of line profiles under a prior algorithm revision.
class TestACRSlicePositionSiemensSolaFit2(TestACRSlicePositionSiemens):
    ACR_DATA = pathlib.Path(
        TEST_DATA_DIR / "acr" / "SiemensSolaFitSlicePosition"
    )
    dL = -1.28, -2.95


# Large bar length difference that the previous cross-correlation
# algorithm under-reported (-0.98 mm on slice 11).
class TestACRSlicePositionGESigna3T(TestACRSlicePositionSiemens):
    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "GE_Signa_3T_T1")
    dL = 0.42, -6.71


class TestACRSlicePositionGEMR450W(TestACRSlicePositionSiemens):
    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "GE_MR450W_1.5T_T2")
    dL = -0.95, -3.35


class TestACRSlicePositionSiemensSolaT2(TestACRSlicePositionSiemens):
    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "Siemens_Sola_1.5T_T2")
    dL = 2.27, 0.35


class TestACRSlicePositionSiemensSolaT1(TestACRSlicePositionSiemens):
    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "Siemens_Sola_1.5T_T1")
    dL = 2.25, 0.37


# Phantom tilted by about 8 degrees: the orientation was previously
# misdetected, and on slice 1 the notch in the phantom edge merges with the
# wedge box.
class TestACRSlicePositionSiemensTilted(TestACRSlicePositionSiemens):
    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "SiemensMTF")
    dL = -1.02, -3.75
    MAX_TILT = 12


class TestACRSlicePositionOrientation(unittest.TestCase):
    """A series exported rotated by a multiple of 90 degrees, or mirrored,
    must give the same result, including the sign of the bar length
    difference."""

    DATASETS = {"Siemens": (-0.15, -1.91), "GE_Signa_3T_T1": (0.42, -6.71)}
    TOLERANCE = 0.05

    @staticmethod
    def transformed_task(name, k, flip, tilt=0.0):
        """Task whose every slice is (tilted, then) rotated by k * 90
        degrees and optionally mirrored, as if exported that way."""
        task = ACRSlicePosition(
            input_data=get_dicom_files(TEST_DATA_DIR / "acr" / name)
        )
        for dcm in task.ACR_obj.slice_stack:
            arr = dcm.pixel_array
            if tilt:
                arr = scipy.ndimage.rotate(
                    arr.astype(float), tilt, reshape=False, order=3
                )
                arr = (
                    np.clip(arr, 0, None).round().astype(dcm.pixel_array.dtype)
                )
            arr = np.rot90(arr, k)
            arr = np.ascontiguousarray(np.fliplr(arr) if flip else arr)
            dcm.Rows, dcm.Columns = arr.shape
            set_pixels(dcm, arr)
        if k % 2:
            # rows and columns swap, so does the pixel spacing
            task.ACR_obj.dx, task.ACR_obj.dy = (
                task.ACR_obj.dy,
                task.ACR_obj.dx,
            )
        return task

    def check(self, tilt):
        for name, expected in self.DATASETS.items():
            for k in range(4):
                for flip in (False, True):
                    task = self.transformed_task(name, k, flip, tilt)
                    for dcm, exp in zip(
                        (
                            task.ACR_obj.slice_stack[0],
                            task.ACR_obj.slice_stack[-1],
                        ),
                        expected,
                        strict=True,
                    ):
                        value = task.get_slice_position(dcm)
                        assert abs(value - exp) < self.TOLERANCE, (
                            f"{name} k={k} flip={flip} tilt={tilt}: "
                            f"{value:.2f} vs {exp}"
                        )

    def test_rotated_and_mirrored(self):
        self.check(tilt=0.0)

    def test_rotated_with_small_tilt(self):
        self.check(tilt=7.0)
