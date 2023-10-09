import os
import unittest
import pathlib
import pydicom

from hazenlib.utils import get_dicom_files
from hazenlib.tasks.acr_slice_position import ACRSlicePosition
from hazenlib.ACRObject import ACRObject
from tests import TEST_DATA_DIR


class TestACRSlicePositionSiemens(unittest.TestCase):
    x_pts = [(123, 129), (123, 129)]
    y_pts = [(40, 82), (44, 82)]
    dL = -0.59, -1.56

    def setUp(self):
        ACR_DATA_SIEMENS = pathlib.Path(TEST_DATA_DIR / 'acr' / 'Siemens')
        siemens_files = get_dicom_files(ACR_DATA_SIEMENS)

        self.acr_slice_position_task = ACRSlicePosition(input_data=siemens_files)

        self.dcm_1 = self.acr_slice_position_task.ACR_obj.dcms[0]
        self.dcm_11 = self.acr_slice_position_task.ACR_obj.dcms[-1]

    def test_wedge_find(self):
        # IMAGE 1
        img = self.dcm_1.pixel_array
        res = self.dcm_1.PixelSpacing
        mask = self.acr_slice_position_task.ACR_obj.get_mask_image(img)
        assert (self.acr_slice_position_task.find_wedges(img, mask, res)[0] ==
                self.x_pts[0]).all() == True

        assert (self.acr_slice_position_task.find_wedges(img, mask, res)[1] ==
                self.y_pts[0]).all() == True

        # IMAGE 11
        img = self.dcm_11.pixel_array
        res = self.dcm_11.PixelSpacing
        mask = self.acr_slice_position_task.ACR_obj.get_mask_image(img)
        assert (self.acr_slice_position_task.find_wedges(img, mask, res)[0] ==
                self.x_pts[1]).all() == True

        assert (self.acr_slice_position_task.find_wedges(img, mask, res)[1] ==
                self.y_pts[1]).all() == True

    def test_slice_position(self):
        slice_position_val_1 = round(self.acr_slice_position_task.get_slice_position(self.dcm_1), 2)
        slice_position_val_11 = round(self.acr_slice_position_task.get_slice_position(self.dcm_11), 2)

        print("\ntest_slice_position.py::TestSlicePosition::test_slice_position")
        print("new_release_value:", slice_position_val_1)
        print("fixed_value:", self.dL[0])

        assert slice_position_val_1 == self.dL[0]
        assert slice_position_val_11 == self.dL[1]


class TestACRSlicePositionGE(TestACRSlicePositionSiemens):
    x_pts = [(246, 257), (246, 257)]
    y_pts = [(77, 164), (89, 162)]
    dL = 0.41, 0.3

    def setUp(self):
        ACR_DATA_GE = pathlib.Path(TEST_DATA_DIR / 'acr' / 'GE')
        ge_files = get_dicom_files(ACR_DATA_GE)

        self.acr_slice_position_task = ACRSlicePosition(input_data=ge_files)

        self.dcm_1 = self.acr_slice_position_task.ACR_obj.dcms[0]
        self.dcm_11 = self.acr_slice_position_task.ACR_obj.dcms[-1]
