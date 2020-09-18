# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 10:52:40 2020

@author: Paul Wilson
"""
import unittest
import pydicom
import numpy as np
import os
import os.path
from pydicom.errors import InvalidDicomError

import hazenlib.relaxometry as hazen_relaxometry
from tests import TEST_DATA_DIR


class TestRelaxometry(unittest.TestCase):
    # test parameters here
    
    # Values for transform_coords tests
    TEST_COORDS = [[0, 0], [0, 1], [1, 2]]
    COORDS_FLIP = [[0, 0], [1, 0], [2, 1]]
    COORDS_TRANS = [[3, 1], [3, 2], [4, 3]]
    COORDS_TRANS_FLIP = [[3, 1], [4, 1], [5, 2]]
    COORDS_TRANS_COL_ROW = [[1, 3], [1, 4], [2, 5]]
    COORDS_TRANS_ROTATE = [[10, 20], [10.5, 20 + np.sqrt(3)/2],
                           [11 + np.sqrt(3)/2, 19.5 + np.sqrt(3)]]
    
    # T1_FILES are in random order to test sorting    
    T1_DIR = os.path.join(TEST_DATA_DIR, 'relaxometry', 'T1', 'site1 20200218',
                          'plate 5')
    T1_FILES = ['20530320', '20530224', '20530416', '20530272', '20530464',
                '20530368']
    T1_TI_SORTED = [50.0, 100.0, 200.0, 400.0, 600.0, 800.0]

    # T2_FILES are in random order to test sorting
    T2_DIR = os.path.join(TEST_DATA_DIR, 'relaxometry', 'T2', 'site1 20200218',
                          'plate 4')
    T2_FILES = ['20528529', '20528577', '20528625', '20528673',
                '20528721', '20528769', '20530528', '20530576',
                '20530624', '20530672', '20530720', '20530768',
                '20530816', '20530864', '20530912', '20530960',
                '20531008', '20531056', '20531104', '20531152',
                '20531200', '20531248', '20531296', '20531344',
                '20531392', '20531440', '20531488', '20531536',
                '20531584', '20531632', '20531680', '20531728']
    T2_TE_SORTED = [12.7, 25.4, 38.1, 50.8, 63.5, 76.2, 88.9, 101.6, 114.3,
                    127.0, 139.7, 152.4, 165.1, 177.8, 190.5, 203.2, 215.9,
                    228.6, 241.3, 254.0, 266.7, 279.4, 292.1, 304.8, 317.5,
                    330.2, 342.9, 355.6, 368.3, 381.0, 393.7, 406.4]

    # Template fitting
    TEMPLATE_PATH_T1_P5 = os.path.join(TEST_DATA_DIR, 'relaxometry', 'T1',
                                 'Template_plate5_T1_signed')
    TEMPLATE_TARGET_PATH_T1_P5 = os.path.join(
        TEST_DATA_DIR, 'relaxometry', 'T1','site1 20200218', 'plate 5',
        '20530224')
    TEMPLATE_TEST_COORDS_ROW_COL = [[56, 95], [62, 117], [81, 133], [104, 134],
                               [124, 121], [133, 98], [127, 75], [109, 61],
                               [84, 60], [64, 72], [80, 81], [78, 111],
                               [109, 113], [110, 82]]
    TEMPLATE_TARGET_COORDS_COL_ROW = [[97, 58], [119, 65], [134, 85],
                                      [133, 108], [119, 127], [96, 134],
                                      [73, 127], [60, 109], [60, 84], [73, 65],
                                      [81, 81], [112, 81], [112, 112],
                                      [81, 111]]
    
    # Mask generation
    MASK_POI_TEMPLATE = np.zeros((14,192,192))
    
    for i in range(14):
        MASK_POI_TEMPLATE[i,TEMPLATE_TEST_COORDS_ROW_COL[i][0], 
                      TEMPLATE_TEST_COORDS_ROW_COL[i][1]] = 1
    MASK_POI_TARGET = np.zeros((14,192,192))
    for i in range(14):
        MASK_POI_TARGET[i,TEMPLATE_TARGET_COORDS_COL_ROW[i][1], 
                      TEMPLATE_TARGET_COORDS_COL_ROW[i][0]] = 1

    ROI0_TEMPLATE_PIXELS = [-620, -706, -678, -630, -710, -672, -726, -684,
                            -714, -654, -692, -702, -644, -738, -668, -652,
                            -744, -702, -702, -658, -664, -672, -658, -668,
                            -738]

    # Only check first three ROIs
    ROI_TEMPLATE_MEANS_T0 = [-683.840, -819.28, -1019.84]
    
    # Values from IDL routine
    PLATE5_T1 = [1862.6, 1435.8, 999.9, 740.7, 498.2, 351.6, 255.2, 178.0,
                 131.7, 93.3, 66.6, 45.1, 32.5, 22.4]
    
    # T2 values from python--will only show consistncy, not accuracy
    PLATE4_T2 = [808.4154451392269, 584.487800647457, 426.0526345966451,
                 306.1406835995861, 213.75915975744184, 153.37330787558878,
                 107.21922831866343, 77.42944345546508, 54.63059517999989,
                 38.131281246132694, 26.60743022583671, 17.62814288965341,
                 11.59178582053302, 7.873018213281735]

    TEMPLATE_PATH_T2 = os.path.join(TEST_DATA_DIR, 'relaxometry', 'T2',
                                    'Template_plate4_T2')

    TEMPLATE_P4_TEST_COORDS_ROW_COL = [[56, 95], [62, 117], [81, 133],
                                       [104, 134], [124, 121], [133, 98],
                                       [127, 75], [109, 61], [84, 60],
                                       [64, 72], [80, 81], [78, 111], 
                                       [109, 113], [110, 82]]
    
    # Values from IDL routine. Site 2 T1 values are signed not magnitude
    SITE2_PLATE5_T1 = [1880.5, 1432.3, 1012.0, 742.5, 504.0, 354.4, 256.3,
                       178.9, 131.9, 93.8, 67.9, 45.8, 33.4, 23.6]

    SITE2_T1_DIR = os.path.join(TEST_DATA_DIR, 'relaxometry', 'T1',
                                'site2 20180925', 'plate 5')
    SITE2_T1_FILES = ['77189804', '77189870', '77189936', '77190002',
                      '77190068', '77190134']

   

    def test_transform_coords(self):
        # no translation, no rotation, input = yx, output = yx
        warp_matrix = np.array([[1, 0, 0], [0, 1, 0]])
        op = hazen_relaxometry.transform_coords(self.TEST_COORDS,
                                                warp_matrix,
                                                input_row_col=True,
                                                output_row_col=True)
        np.testing.assert_allclose(op, self.TEST_COORDS)
        # 'Identity coordinate transformation row_col (yx) -> row_col (yx)' 
        # ' failed'

        # no translation, no rotation, flip
        # input = col_row (xy), output = row_col (yx)
        op = hazen_relaxometry.transform_coords(self.TEST_COORDS, warp_matrix,
                                                input_row_col=False,
                                                output_row_col=True)
        
        np.testing.assert_allclose(op, self.COORDS_FLIP)

        # no translation, no rotation, input = col_row (xy), 
        # output = col_row (xy)
        op = hazen_relaxometry.transform_coords(self.TEST_COORDS, warp_matrix,
                                                input_row_col=False,
                                                output_row_col=False)
        np.testing.assert_allclose(op, self.TEST_COORDS)
        # 'Identity coordinate transformation XY -> XY failed'

        # translation x=1, y=3, no rotation, input = row_col (yx),
        # output = row_col (yx)
        warp_matrix = np.array([[1, 0, 1], [0, 1, 3]])
        op = hazen_relaxometry.transform_coords(self.TEST_COORDS, warp_matrix,
                                                input_row_col=True,
                                                output_row_col=True)
        np.testing.assert_allclose(op, self.COORDS_TRANS)
        # 'Translation coordinate transformation YX -> YX failed'

        # translation x=1, y=3, no rotation, input = col_row (xy),
        # output = row_col (yx)
        warp_matrix = np.array([[1, 0, 1], [0, 1, 3]])
        op = hazen_relaxometry.transform_coords(self.TEST_COORDS, warp_matrix,
                                                input_row_col=False,
                                                output_row_col=True)
        np.testing.assert_allclose(op, self.COORDS_TRANS_FLIP)
        # 'Translation coordinate transformation XY -> YX failed'

        # translation x=1, y=3, no rotation, input = col_row (xy),
        # output = col_row (xy)
        warp_matrix = np.array([[1, 0, 1], [0, 1, 3]])
        op = hazen_relaxometry.transform_coords(self.TEST_COORDS, warp_matrix,
                                                input_row_col=False,
                                                output_row_col=False)
        np.testing.assert_allclose(op, self.COORDS_TRANS_COL_ROW)
        # 'Translation coordinate transformation XY -> XY failed'

        # rotation (-30) degrees, translation col=10, row=20,
        # input = col_row (xy), output = col_row (xy)
        warp_matrix = np.array([[np.sqrt(3)/2, 0.5, 10],
                                [-0.5, np.sqrt(3)/2, 20]])
        # use float64 rather than int32 for coordinates to better test rotation
        op = hazen_relaxometry.transform_coords(np.array(self.TEST_COORDS,
                                                         dtype=np.float64),
                                                warp_matrix,
                                                input_row_col=False,
                                                output_row_col=False)
        np.testing.assert_allclose(op, self.COORDS_TRANS_ROTATE)


    def test_template_fit(self):
        template_dcm = pydicom.read_file(self.TEMPLATE_PATH_T1_P5)

        target_dcm = pydicom.dcmread(self.TEMPLATE_TARGET_PATH_T1_P5)
        t1_image_stack = hazen_relaxometry.T1ImageStack([target_dcm],
                                                        template_dcm,
                                                        plate_number=5)
        t1_image_stack.template_fit()

        transformed_coordinates_xy = hazen_relaxometry.transform_coords(
            self.TEMPLATE_TEST_COORDS_ROW_COL, t1_image_stack.warp_matrix,
            input_row_col=True, output_row_col=False)

        # test to within +/- 1 pixel (also checks YX-XY change)
        np.testing.assert_allclose(
            transformed_coordinates_xy, self.TEMPLATE_TARGET_COORDS_COL_ROW,
            atol=1)

    def test_image_stack_T1_sort(self):
        # read list of un-ordered T1 files, sort by TI, test sorted
        t1_dcms = [pydicom.dcmread(os.path.join(self.T1_DIR, fname))
                   for fname in self.T1_FILES]
        t1_image_stack = hazen_relaxometry.T1ImageStack(t1_dcms)
        sorted_output = [image.InversionTime.real for image in
                         t1_image_stack.images]
        assert sorted_output == self.T1_TI_SORTED

    def test_image_stack_T2_sort(self):
        # read list of un-ordered T2 files, sort by TE, test sorted
        t2_dcms = [pydicom.dcmread(os.path.join(self.T2_DIR, fname)) 
                   for fname in self.T2_FILES]
        t2_image_stack = hazen_relaxometry.T2ImageStack(t2_dcms)

        sorted_output = [image.EchoTime.real for image in
                         t2_image_stack.images]

        assert sorted_output == self.T2_TE_SORTED
        
        
    def test_generate_time_series_template_POIs(self):
        # Test on template first, no image fitting needed
        # Need image to get correct size
        template_dcm = pydicom.dcmread(self.TEMPLATE_PATH_T1_P5)
        template_image_stack = hazen_relaxometry.T1ImageStack([template_dcm])
        template_image_stack.generate_time_series(
            self.TEMPLATE_TEST_COORDS_ROW_COL,
            fit_coords=False)
        
        for i in range(np.size(self.MASK_POI_TEMPLATE, 0)):
            np.testing.assert_equal(
                template_image_stack.ROI_time_series[i].POI_mask,
                self.MASK_POI_TEMPLATE[i])

    def test_generate_time_series_target_POIs(self):
        # Test on target and check image fitting too.
        template_dcm = pydicom.read_file(self.TEMPLATE_PATH_T1_P5)

        target_dcm = pydicom.dcmread(self.TEMPLATE_TARGET_PATH_T1_P5)
        target_image_stack = hazen_relaxometry.T1ImageStack([target_dcm],
                                                            template_dcm)
        target_image_stack.template_fit()
        # transformed_coordinates_yx = hazen_relaxometry.transform_coords(
        #     self.TEMPLATE_TEST_COORDS_YX, target_image_stack.warp_matrix,
        #     input_yx=True, output_yx=True)
        target_image_stack.generate_time_series(
            self.TEMPLATE_TEST_COORDS_ROW_COL, fit_coords=True)
        for i in range(np.size(self.MASK_POI_TARGET, 0)):
            np.testing.assert_equal(
                target_image_stack.ROI_time_series[i].POI_mask,
                self.MASK_POI_TARGET[i])

    def test_extract_single_roi(self):
        # Test that ROI pixel value extraction works. Use template DICOM for
        # both template and image to avoid errors due to slight variation in
        # fitting.
        template_dcm = pydicom.read_file(self.TEMPLATE_PATH_T1_P5)

        template_image_stack = hazen_relaxometry.T1ImageStack([template_dcm],
                                                            template_dcm)
        # set warp_matrix to identity matrix
        # template_image_stack.warp_matrix = np.eye(2, 3, dtype=np.float32)
        template_image_stack.generate_time_series(
            self.TEMPLATE_TEST_COORDS_ROW_COL, fit_coords=False)

        np.testing.assert_equal(
            template_image_stack.ROI_time_series[0].pixel_values[0],
            self.ROI0_TEMPLATE_PIXELS)
        
    def test_template_roi_means(self):
        # Check mean of first 3 ROIs in template match with ImageJ calculations
        template_dcm = pydicom.read_file(self.TEMPLATE_PATH_T1_P5)

        template_image_stack = hazen_relaxometry.T1ImageStack([template_dcm],
                                                            template_dcm)
        # set warp_matrix to identity matrix
        # template_image_stack.warp_matrix = np.eye(2, 3, dtype=np.float32)
        template_image_stack.generate_time_series(
            self.TEMPLATE_TEST_COORDS_ROW_COL, fit_coords=False)
        
        for i in self.ROI_TEMPLATE_MEANS_T0:
            np.testing.assert_allclose(
                template_image_stack.ROI_time_series[0].pixel_values[0],
                self.ROI0_TEMPLATE_PIXELS)
        
    def test_t1_calc_magnitude_image(self):
        """Test T1 value for plate 5 spheres."""
        template_dcm = pydicom.read_file(self.TEMPLATE_PATH_T1_P5)
        t1_dcms = [pydicom.dcmread(os.path.join(self.T1_DIR, fname))
                   for fname in self.T1_FILES]
        t1_image_stack = hazen_relaxometry.T1ImageStack(t1_dcms, template_dcm)
        t1_image_stack.template_fit()
        t1_image_stack.generate_time_series(
            self.TEMPLATE_TEST_COORDS_ROW_COL, fit_coords=True)
        t1_image_stack.generate_fit_function()
        t1_published = \
            hazen_relaxometry.TEMPLATE_VALUES\
                ['plate5']['t1']['relax_times']
        t1_image_stack.initialise_fit_parameters(t1_estimates=t1_published)

        #t1_image_stack.initialise_fit_parameters()
        t1_image_stack.find_t1s()
    
        np.testing.assert_allclose(t1_image_stack.t1s, self.PLATE5_T1,
                                          rtol=0.02, atol=1)

    def test_t2_calc_magnitude_image(self):
        """Test T2 value for plate 4 spheres."""
        template_dcm = pydicom.read_file(self.TEMPLATE_PATH_T2)
        t2_dcms = [pydicom.dcmread(os.path.join(self.T2_DIR, fname))
                   for fname in self.T2_FILES]
        t2_image_stack = hazen_relaxometry.T2ImageStack(t2_dcms, template_dcm)
        t2_image_stack.template_fit()
        t2_image_stack.generate_time_series(
            self.TEMPLATE_P4_TEST_COORDS_ROW_COL, fit_coords=True)
        t2_published = \
            hazen_relaxometry.TEMPLATE_VALUES['plate4']['t2']['relax_times']
        t2_image_stack.initialise_fit_parameters(t2_estimates=t2_published)
        t2_image_stack.initialise_fit_parameters(t2_published)
        t2_image_stack.find_t2s()
    
        np.testing.assert_allclose(t2_image_stack.t2s, self.PLATE4_T2,
                                         rtol=0.01, atol=1)

    def test_t1_calc_signed_image(self):
        """Test T1 value for signed plate 5 spheres (site 2)."""
        template_dcm = pydicom.read_file(self.TEMPLATE_PATH_T1_P5)
        t1_dcms = [pydicom.dcmread(os.path.join(self.SITE2_T1_DIR, fname))
                   for fname in self.SITE2_T1_FILES]
        t1_image_stack = hazen_relaxometry.T1ImageStack(t1_dcms, template_dcm)
        t1_image_stack.template_fit()
        t1_image_stack.generate_time_series(
            self.TEMPLATE_TEST_COORDS_ROW_COL, fit_coords=True)
        t1_image_stack.generate_fit_function()
        t1_published = \
            hazen_relaxometry.TEMPLATE_VALUES['plate5']['t1']['relax_times']
        t1_image_stack.initialise_fit_parameters(t1_estimates=t1_published)
        t1_image_stack.find_t1s()
    
        np.testing.assert_allclose(t1_image_stack.t1s, self.SITE2_PLATE5_T1,
                                          rtol=0.02, atol=1)
        

    
        
if __name__ == '__main__':
    unittest.main(verbosity=2)
