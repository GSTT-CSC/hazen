import os
import unittest
import pathlib
import pydicom
import numpy as np

from hazenlib.tasks.acr_geometric_accuracy import ACRGeometricAccuracy
from tests import TEST_DATA_DIR, TEST_REPORT_DIR


class TestACRGeometricAccuracySiemens(unittest.TestCase):
    ACR_GEOMETRIC_ACCURACY_DATA = pathlib.Path(TEST_DATA_DIR / 'acr')
    centre = (128, 129)
    L1 = 190.43, 186.52
    L5 = 190.43, 186.52, 189.45, 191.41
    test_point = (-60.98, -45.62)

    def setUp(self):
        self.acr_geometric_accuracy_task = ACRGeometricAccuracy(data_paths=[os.path.join(TEST_DATA_DIR, 'acr')],
                                                                report_dir=pathlib.PurePath.joinpath(TEST_REPORT_DIR))
        self.dcm = pydicom.read_file(os.path.join(TEST_DATA_DIR, 'acr', 'Siemens', '0.dcm'))
        self.dcm2 = pydicom.read_file(os.path.join(TEST_DATA_DIR, 'acr', 'Siemens', '4.dcm'))

    def test_object_centre(self):
        data = self.dcm.pixel_array
        assert self.acr_geometric_accuracy_task.centroid_com(data)[1] == self.centre

    def test_geo_accuracy_slice1(self):
        slice1_vals = np.array(self.acr_geometric_accuracy_task.get_geometric_accuracy_slice1(self.dcm))
        slice1_vals = np.round(slice1_vals, 2)
        assert (slice1_vals == self.L1).all() == True

    def test_geo_accuracy_slice5(self):
        slice5_vals = np.array(self.acr_geometric_accuracy_task.get_geometric_accuracy_slice5(self.dcm2))
        slice5_vals = np.round(slice5_vals, 2)
        assert (slice5_vals == self.L5).all() == True

    def test_rotate_point(self):
        rotated_point = np.array(self.acr_geometric_accuracy_task.rotate_point((0, 0), (30, 70), 150))
        rotated_point = np.round(rotated_point, 2)
        print(rotated_point)
        assert (rotated_point == self.test_point).all() == True


# class TestACRUniformityPhilips(unittest.TestCase):

class TestACRGeometricAccuracyGE(unittest.TestCase):
    ACR_GEOMETRIC_ACCURACY_DATA = pathlib.Path(TEST_DATA_DIR / 'acr')
    L1 = 189.92, 187.89
    L5 = 189.92, 188.39, 190.43, 189.92
    distortion_metrics = [-0.59, 2.11, 0.49]

    def setUp(self):
        self.acr_geometric_accuracy_task = ACRGeometricAccuracy(data_paths=[os.path.join(TEST_DATA_DIR, 'acr')],
                                                                report_dir=pathlib.PurePath.joinpath(TEST_REPORT_DIR))
        self.dcm = pydicom.read_file(os.path.join(TEST_DATA_DIR, 'acr', 'GE', '10.dcm'))
        self.dcm2 = pydicom.read_file(os.path.join(TEST_DATA_DIR, 'acr', 'GE', '6.dcm'))

    def test_geo_accuracy_slice1(self):
        slice1_vals = np.array(self.acr_geometric_accuracy_task.get_geometric_accuracy_slice1(self.dcm))
        slice1_vals = np.round(slice1_vals, 2)
        assert (slice1_vals == self.L1).all() == True

    def test_geo_accuracy_slice5(self):
        slice5_vals = np.array(self.acr_geometric_accuracy_task.get_geometric_accuracy_slice5(self.dcm2))
        slice5_vals = np.round(slice5_vals, 2)
        assert (slice5_vals == self.L5).all() == True

    def test_distortion_metrics(self):
        metrics = np.array(self.acr_geometric_accuracy_task.distortion_metric(self.L1+self.L5))
        metrics = np.round(metrics, 2)
        assert (metrics == self.distortion_metrics).all() == True
