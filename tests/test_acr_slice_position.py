import unittest
import pathlib

from hazenlib.utils import get_dicom_files
from hazenlib.tasks.acr_slice_position import ACRSlicePosition
from tests import TEST_DATA_DIR


class TestACRSlicePositionSiemens(unittest.TestCase):
    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "Siemens")
    slice_1_x_pts = [123, 129]
    slice_11_x_pts = [123, 129]
    slice_1_y_pts = [44, 84]
    slice_11_y_pts = [43, 83]
    dL = -0.59, -1.56

    def setUp(self):
        input_files = get_dicom_files(self.ACR_DATA)
        self.acr_slice_position_task = ACRSlicePosition(input_data=input_files)

        self.dcm_1 = self.acr_slice_position_task.ACR_obj.slice_stack[0]
        img_1 = self.dcm_1.pixel_array
        cxy_1, _ = self.acr_slice_position_task.ACR_obj.find_phantom_center(
            img_1,
            self.acr_slice_position_task.ACR_obj.dx,
            self.acr_slice_position_task.ACR_obj.dy,
        )
        self.slice1_x_pts, self.slice1_y_pts = (
            self.acr_slice_position_task.find_wedges(img_1, cxy_1)
        )

        self.dcm_11 = self.acr_slice_position_task.ACR_obj.slice_stack[-1]
        img_11 = self.dcm_11.pixel_array
        cxy_11, _ = self.acr_slice_position_task.ACR_obj.find_phantom_center(
            img_11,
            self.acr_slice_position_task.ACR_obj.dx,
            self.acr_slice_position_task.ACR_obj.dy,
        )
        (
            self.slice11_x_pts,
            self.slice11_y_pts,
        ) = self.acr_slice_position_task.find_wedges(img_11, cxy_11)

    # IMAGE 1
    def test_find_wedge_slice1_x(self):
        assert self.slice1_x_pts == self.slice_1_x_pts

    def test_find_wedge_slice1_y(self):
        assert self.slice1_y_pts == self.slice_1_y_pts

    # IMAGE 11
    def test_find_wedge_slice11_x(self):
        assert self.slice11_x_pts == self.slice_11_x_pts

    def test_find_wedge_slice11_y(self):
        assert self.slice11_y_pts == self.slice_11_y_pts

    def test_slice_position(self):
        slice_position_val_1 = round(
            self.acr_slice_position_task.get_slice_position(self.dcm_1), 2
        )
        slice_position_val_11 = round(
            self.acr_slice_position_task.get_slice_position(self.dcm_11), 2
        )

        print(
            "\ntest_slice_position.py::TestSlicePosition::test_slice_position"
        )
        print("new_release_value:", slice_position_val_1)
        print("fixed_value:", self.dL[0])

        assert slice_position_val_1 == self.dL[0]
        assert slice_position_val_11 == self.dL[1]


class TestACRSlicePositionGE(TestACRSlicePositionSiemens):
    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "GE")
    slice_1_x_pts = [247, 257]
    slice_11_x_pts = [246, 256]
    slice_1_y_pts = [93, 171]
    slice_11_y_pts = [94, 172]
    dL = 0.51, 4.47


class TestACRSlicePositionPhilipsAchieva(TestACRSlicePositionSiemens):
    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "PhilipsAchieva")
    slice_1_x_pts = [125, 131]
    slice_11_x_pts = [123, 129]
    slice_1_y_pts = [43, 83]
    slice_11_y_pts = [41, 81]
    dL = 1.37, -3.12


class TestACRSlicePositionSiemensSolaFit(TestACRSlicePositionSiemens):
    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "SiemensSolaFit")
    slice_1_x_pts = [124, 130]
    slice_11_x_pts = [123, 129]
    slice_1_y_pts = [43, 83]
    slice_11_y_pts = [41, 81]
    dL = -0.2, -1.37


class TestACRSlicePositionSiemensLargeSliceLocationDelta(
    TestACRSlicePositionSiemens
):
    ACR_DATA = pathlib.Path(
        TEST_DATA_DIR / "acr" / "SiemensLargeSliceLocationDelta"
    )
    slice_1_x_pts = [123, 129]
    slice_11_x_pts = [123, 129]
    slice_1_y_pts = [47, 87]
    slice_11_y_pts = [46, 86]
    dL = -6.64, -6.84


class TestACRSlicePositionPhilips3TDStream(TestACRSlicePositionSiemens):
    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "Philips3TdStream")
    slice_1_x_pts = [124, 130]
    slice_11_x_pts = [124, 130]
    slice_1_y_pts = [44, 84]
    slice_11_y_pts = [41, 81]
    dL = 2.15, -2.15


class TestACRSlicePositionPhilips3TDStream2(TestACRSlicePositionSiemens):
    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "Philips3TdStream2")
    slice_1_x_pts = [124, 130]
    slice_11_x_pts = [124, 130]
    slice_1_y_pts = [23, 63]
    slice_11_y_pts = [42, 82]
    dL = 3.12, -0.78


# triggers mispositioning of line profiles under a prior algorithm revision.
class TestACRSlicePositionSiemensSolaFit2(TestACRSlicePositionSiemens):
    ACR_DATA = pathlib.Path(
        TEST_DATA_DIR / "acr" / "SiemensSolaFitSlicePosition"
    )
    slice_1_x_pts = [124, 130]
    slice_11_x_pts = [124, 130]
    slice_1_y_pts = [43, 83]
    slice_11_y_pts = [42, 82]
    dL = -1.17, -2.54
