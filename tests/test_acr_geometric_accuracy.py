import unittest
import pathlib
import numpy as np
import pydicom
import hazenlib.acr_geometric_accuracy as hazen_acr_geometric_accuracy
from tests import TEST_DATA_DIR


class TestACRGeometricAccuracySiemens(unittest.TestCase):
    ACR_GEOMETRIC_ACCURACY_DATA = pathlib.Path(TEST_DATA_DIR / 'acr')
    centre = [129, 128]
    L1 = 190.43, 186.52
    L5 = 190.43, 186.52, 191.41, 189.45
    test_matrix = np.array(((0.7071, -0.7071), (0.7071, 0.7071)))

    def setUp(self):
        self.dcm_file = pydicom.read_file(str(self.ACR_GEOMETRIC_ACCURACY_DATA / 'Siemens' / 'Test' / '0.dcm'))
        self.dcm_file2 = pydicom.read_file(str(self.ACR_GEOMETRIC_ACCURACY_DATA / 'Siemens' / 'Test' / '4.dcm'))

    def test_object_centre(self):
        data = self.dcm_file.pixel_array
        assert hazen_acr_geometric_accuracy.centroid_com(data) == self.centre

    def test_geo_accuracy_slice1(self):
        assert(np.round(hazen_acr_geometric_accuracy.geo_accuracy_slice1(self.dcm_file),2) == self.L1).all() == True

    def test_geo_accuracy_slice5(self):
        assert(np.round(hazen_acr_geometric_accuracy.geo_accuracy_slice5(self.dcm_file2),2) == self.L5).all() == True

    def test_rot_matrix(self):
        assert(np.round(hazen_acr_geometric_accuracy.rot_matrix(45),4) == self.test_matrix).all() == True



# class TestACRUniformityPhilips(unittest.TestCase):

class TestACRGeometricAccuracyGE(unittest.TestCase):
    ACR_GEOMETRIC_ACCURACY_DATA = pathlib.Path(TEST_DATA_DIR / 'acr')
    centre = [255, 253]
    L1 = 189.92, 187.89
    L5 = 189.92, 188.39, 189.92, 190.42
    test_matrix = np.array(((0, -1), (1, 0)))

    def setUp(self):
        self.dcm_file = pydicom.read_file(str(self.ACR_GEOMETRIC_ACCURACY_DATA / 'GE' / 'Test' / '10.dcm'))
        self.dcm_file2 = pydicom.read_file(str(self.ACR_GEOMETRIC_ACCURACY_DATA / 'GE' / 'Test' / '6.dcm'))

    def test_object_centre(self):
        data = self.dcm_file.pixel_array
        assert hazen_acr_geometric_accuracy.centroid_com(data) == self.centre

    def test_geo_accuracy_slice1(self):
        assert (np.round(hazen_acr_geometric_accuracy.geo_accuracy_slice1(self.dcm_file), 2) == self.L1).all() == True

    def test_geo_accuracy_slice5(self):
        assert (np.round(hazen_acr_geometric_accuracy.geo_accuracy_slice5(self.dcm_file2), 2) == self.L5).all() == True

    def test_rot_matrix(self):
        assert (np.round(hazen_acr_geometric_accuracy.rot_matrix(90), 4) == self.test_matrix).all() == True
