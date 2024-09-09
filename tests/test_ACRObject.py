import os
import unittest
import pydicom
import numpy as np

from hazenlib.ACRObject import ACRObject
from tests import TEST_DATA_DIR


# Siemens (axial)
class TestACRTools(unittest.TestCase):
    centre = (130, 130)
    rotation = -1.0
    horizontal_distance = 191.40625
    horizontal_end = (128, 255)
    vertical_distance = 187.5
    vertical_end = (255, 126)

    def setUp(self):
        self.Siemens_data = [
            pydicom.read_file(os.path.join(TEST_DATA_DIR, "acr", "Siemens", f"{i}"))
            for i in os.listdir(os.path.join(TEST_DATA_DIR, "acr", "Siemens"))
        ]

        self.ACR_object = ACRObject(self.Siemens_data)
        self.img1 = self.ACR_object.slice_stack[0].pixel_array
        self.img7 = self.ACR_object.slice_stack[6].pixel_array

    def test_find_centre(self):
        phantom_centre, _ = self.ACR_object.find_phantom_center(
            self.img7, self.ACR_object.dx, self.ACR_object.dy
        )
        assert self.centre == phantom_centre

    def test_find_rotation(self):
        rotation_angle = self.ACR_object.determine_rotation(self.img1)
        assert self.rotation == np.round(rotation_angle, 1)

    def test_measure_orthogonal_lengths(self):
        mask = self.ACR_object.get_mask_image(self.img1)
        length_dict = self.ACR_object.measure_orthogonal_lengths(mask, 0)
        assert self.horizontal_distance == length_dict["Horizontal Distance"]
        assert self.horizontal_end == length_dict["Horizontal End"]
        assert self.vertical_distance == length_dict["Vertical Distance"]
        assert self.vertical_end == length_dict["Vertical End"]


"""
# Siemens transverse = axial
class TestACRToolsTRA(TestACRTools):
    centre = (136, 128)
    rotation = 0.0
    horizontal_distance = 189.0
    horizontal_end = (128, 255)
    vertical_distance = 191.0
    vertical_end = (255, 136)

    def setUp(self):
        self.TRA_data = [
            pydicom.read_file(
                os.path.join(TEST_DATA_DIR, "acr", "ACR_BODY_TRA_ONE", f"{i}")
            )
            for i in os.listdir(os.path.join(TEST_DATA_DIR, "acr", "ACR_BODY_TRA_ONE"))
        ]

        self.ACR_object = ACRObject(self.TRA_data)
        self.img1 = self.ACR_object.slice_stack[0].pixel_array
        self.img7 = self.ACR_object.slice_stack[6].pixel_array


# Siemens coronal
class TestACRToolsCOR(TestACRTools):
    centre = (128, 128)
    rotation = 0.0
    horizontal_distance = 190.0
    horizontal_end = (128, 255)
    vertical_distance = 191.0
    vertical_end = (255, 126)

    def setUp(self):
        self.COR_data = [
            pydicom.read_file(
                os.path.join(TEST_DATA_DIR, "acr", "ACR_BODY_COR_ONE", f"{i}")
            )
            for i in os.listdir(os.path.join(TEST_DATA_DIR, "acr", "ACR_BODY_COR_ONE"))
        ]

        self.ACR_object = ACRObject(self.COR_data)
        self.img1 = self.ACR_object.slice_stack[0].pixel_array
        self.img7 = self.ACR_object.slice_stack[6].pixel_array


# Siemens sagittal
class TestACRToolsSAG(TestACRTools):
    rotation = -90.0
    centre = (130, 148)
    horizontal_distance = 190.0
    horizontal_end = (148, 255)
    vertical_distance = 189.0
    vertical_end = (255, 130)

    def setUp(self):
        self.SAG_data = [
            pydicom.read_file(
                os.path.join(TEST_DATA_DIR, "acr", "ACR_BODY_SAG_ONE", f"{i}")
            )
            for i in os.listdir(os.path.join(TEST_DATA_DIR, "acr", "ACR_BODY_SAG_ONE"))
        ]

        self.ACR_object = ACRObject(self.SAG_data)
        self.img1 = self.ACR_object.slice_stack[0].pixel_array
        self.img7 = self.ACR_object.slice_stack[6].pixel_array
"""


# GE axial
class TestACRToolsGE(TestACRTools):
    rotation = 0.0
    centre = (254, 256)
    horizontal_distance = 190.425
    horizontal_end = (262, 511)
    vertical_distance = 188.9016
    vertical_end = (511, 260)
    test_point = (-60.98, -45.62)

    def setUp(self):
        self.GE_data = [
            pydicom.read_file(os.path.join(TEST_DATA_DIR, "acr", "GE", f"{i}"))
            for i in os.listdir(os.path.join(TEST_DATA_DIR, "acr", "GE"))
        ]

        self.ACR_object = ACRObject(self.GE_data)
        self.img1 = self.ACR_object.slice_stack[0].pixel_array
        self.img7 = self.ACR_object.slice_stack[6].pixel_array

    def test_rotate_point(self):
        rotated_point = np.array(self.ACR_object.rotate_point((0, 0), (30, 70), 150))
        rotated_point = np.round(rotated_point, 2)
        assert (rotated_point == self.test_point).all() == True
