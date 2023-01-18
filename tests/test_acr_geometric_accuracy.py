import os
import unittest
import pathlib
import pydicom

from hazenlib.tasks.acr_geometric_accuracy import ACRGeometricAccuracy
from tests import TEST_DATA_DIR, TEST_REPORT_DIR


class TestACRGeometricAccuracySiemens(unittest.TestCase):
    ACR_GEOMETRIC_ACCURACY_DATA = pathlib.Path(TEST_DATA_DIR / 'acr')
    centre = [129, 128]
    L1 = 190.4296875, 186.5234375
    L5 = 190.4296875, 186.5234375, 189.45312500000003, 191.40624999999997
    test_point = (-60.98076211353315, -45.62177826491071)

    def setUp(self):
        self.acr_geometric_accuracy_task = ACRGeometricAccuracy(data_paths=[os.path.join(TEST_DATA_DIR, 'acr')],
                                                                report_dir=pathlib.PurePath.joinpath(TEST_REPORT_DIR))
        self.dcm = pydicom.read_file(os.path.join(TEST_DATA_DIR, 'acr', 'Siemens', '0.dcm'))
        self.dcm2 = pydicom.read_file(os.path.join(TEST_DATA_DIR, 'acr', 'Siemens', '4.dcm'))

    def test_object_centre(self):
        data = self.dcm.pixel_array
        assert self.acr_geometric_accuracy_task.centroid_com(data) == self.centre

    def test_geo_accuracy_slice1(self):
        assert (self.acr_geometric_accuracy_task.get_geometric_accuracy_slice1(self.dcm) == self.L1).all() == True

    def test_geo_accuracy_slice5(self):
        assert (self.acr_geometric_accuracy_task.get_geometric_accuracy_slice5(self.dcm) == self.L1).all() == True

    def test_rotate_point(self):
        assert (self.acr_geometric_accuracy_task.rotate_point((0, 0), (30, 70), 150) == self.test_point).all() == True


# class TestACRUniformityPhilips(unittest.TestCase):

class TestACRGeometricAccuracyGE(unittest.TestCase):
    ACR_GEOMETRIC_ACCURACY_DATA = pathlib.Path(TEST_DATA_DIR / 'acr')
    centre = [255, 253]
    L1 = 189.92, 187.89
    L5 = 189.92, 188.39, 189.92, 190.42
    test_matrix = np.array(((0, -1), (1, 0)))

    def setUp(self):
        self.dcm_file = pydicom.read_file(str(self.ACR_GEOMETRIC_ACCURACY_DATA / 'GE' / '10.dcm'))
        self.dcm_file2 = pydicom.read_file(str(self.ACR_GEOMETRIC_ACCURACY_DATA / 'GE' / '6.dcm'))

    # def test_object_centre(self):
    #     data = self.dcm_file.pixel_array
    #     assert hazen_acr_geometric_accuracy.centroid_com(data) == self.centre
    #
    # def test_geo_accuracy_slice1(self):
    #     assert (np.round(hazen_acr_geometric_accuracy.geo_accuracy_slice1(self.dcm_file), 2) == self.L1).all() == True
    #
    # def test_geo_accuracy_slice5(self):
    #     assert (np.round(hazen_acr_geometric_accuracy.geo_accuracy_slice5(self.dcm_file2), 2) == self.L5).all() == True
    #
    # def test_rot_matrix(self):
    #     assert (np.round(hazen_acr_geometric_accuracy.rot_matrix(90), 4) == self.test_matrix).all() == True
