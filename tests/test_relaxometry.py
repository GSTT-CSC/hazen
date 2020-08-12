# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 10:52:40 2020

@author: Paul Wilson
"""
import unittest
import pydicom
import numpy as np
import os, os.path
from pydicom.errors import InvalidDicomError

import hazenlib.relaxometry as hazen_relaxometry
from tests import TEST_DATA_DIR


class TestRelaxometry(unittest.TestCase):
    # test parameters here
    
    def test_transform_coords(self):
        TEST_COORDS = np.array([[0, 0], [0, 1], [1, 2]])
        
        # no translation, no rotation, input = yx, output = yx
        warp_matrix = np.array([[1, 0, 0], [0, 1, 0]])
        op = hazen_relaxometry.transform_coords(TEST_COORDS, warp_matrix, input_yx=True, output_yx=True)
        assert np.testing.assert_allclose(op, TEST_COORDS) is None # 'Identity coordinate transformation YX -> YX failed'
        
        # no translation, no rotation, input = xy, output = yx
        op = hazen_relaxometry.transform_coords(TEST_COORDS, warp_matrix, input_yx=False, output_yx=True)
        desired_output = np.array([[0, 0], [1, 0], [2, 1]])
        assert np.testing.assert_allclose(op, desired_output) is None # 'Identity coordinate transformation XY -> YX failed'
        
         # no translation, no rotation, input = xy, output = xy
        op = hazen_relaxometry.transform_coords(TEST_COORDS, warp_matrix, input_yx=False, output_yx=False)
        assert np.testing.assert_allclose(op, TEST_COORDS) is None # 'Identity coordinate transformation XY -> XY failed'
        
        # translation x=1, y=3, no rotation, input = yx, output = yx
        desired_output = np.array([[3, 1], [3, 2], [4, 3]])
        warp_matrix = np.array([[1, 0, 1], [0, 1, 3]])
        op = hazen_relaxometry.transform_coords(TEST_COORDS, warp_matrix, input_yx=True, output_yx=True)
        assert np.testing.assert_allclose(op, desired_output) is None # 'Translation coordinate transformation YX -> YX failed'
       
        # translation x=1, y=3, no rotation, input = xy, output = yx
        desired_output = np.array([[3, 1], [4, 1], [5, 2]])
        warp_matrix = np.array([[1, 0, 1], [0, 1, 3]])
        op = hazen_relaxometry.transform_coords(TEST_COORDS, warp_matrix, input_yx=False, output_yx=True)
        assert np.testing.assert_allclose(op, desired_output) is None # 'Translation coordinate transformation XY -> YX failed'
       
        # translation x=1, y=3, no rotation, input = xy, output = xy
        desired_output = np.array([[1, 3], [1, 4], [2, 5]])
        warp_matrix = np.array([[1, 0, 1], [0, 1, 3]])
        op = hazen_relaxometry.transform_coords(TEST_COORDS, warp_matrix, input_yx=False, output_yx=False)
        assert np.testing.assert_allclose(op, desired_output) is None # 'Translation coordinate transformation XY -> XY failed'
       
        # rotation (-30) degrees, translation x=10, y=20, input = xy, output = xy
        desired_output = np.array([[10, 20], 
                                   [10.5, 20+np.sqrt(3)/2],
                                   [11+np.sqrt(3)/2, 19.5+np.sqrt(3)]])
        warp_matrix = np.array([[np.sqrt(3)/2, 0.5, 10], [-0.5, np.sqrt(3)/2, 20]])
        # use float64 rather than int32 for coordinates to better test rotation
        op = hazen_relaxometry.transform_coords(np.array(TEST_COORDS, 
                                                         dtype=np.float64),
                                                warp_matrix, input_yx=False,
                                                output_yx=False)
        assert np.testing.assert_allclose(op, desired_output) is None # 'Rotation / translation coordinate transformation XY -> XY failed'
    
    
    def test_template_fit(self):
        template_path = os.path.join(TEST_DATA_DIR, 'relaxometry', 'T1',
                                     'Template_plate5_T1_signed')
        template_px=pydicom.read_file(template_path).pixel_array
        
        # get list of pydicom objects
        target_path = os.path.join(TEST_DATA_DIR, 'relaxometry', 'T1', 
                                     'site1 20200218', 'plate 5', '20530224')
        target_dcm = pydicom.dcmread(target_path)
        t1_image_stack = hazen_relaxometry.T1ImageStack([target_dcm], template_px, plate_number=5)
        t1_image_stack.template_fit()
        
        # desired_warp_matrix = np.array([[ 1.0051669 , -0.05144447,  4.1132936 ],
        #                                 [ 0.05119833,  0.98579186, -1.7364146 ]])
        plate5_sphere_centres_yx = np.array([[56, 95],
                                             [62, 117],
                                             [81, 133],
                                             [104, 134],
                                             [124, 121],
                                             [133, 98],
                                             [127, 75],
                                             [109, 61],
                                             [84, 60],
                                             [64, 72],
                                             [80, 81],
                                             [78, 111],
                                             [109, 113],
                                             [110, 82]])
        
        desired_output_xy = np.array([[ 97,  58],
                                      [119,  65],
                                      [134,  85],
                                      [133, 108],
                                      [119, 127],
                                      [ 96, 134],
                                      [ 73, 127],
                                      [ 60, 109],
                                      [ 60,  84],
                                      [ 73,  65],
                                      [ 81,  81],
                                      [112,  81],
                                      [112, 112],
                                      [ 81, 111]])
        
        transformed_coordinates_xy = hazen_relaxometry.transform_coords(
            plate5_sphere_centres_yx, t1_image_stack.warp_matrix, 
            input_yx=True, output_yx=False)
                                                                     
        # test to within +/- 1 pixel
        assert np.testing.assert_allclose(
            transformed_coordinates_xy, desired_output_xy, atol=1) is None

if __name__ == '__main__':
    unittest.main(verbosity=2)