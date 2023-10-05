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

from hazenlib.tasks.relaxometry import (
    transform_coords, T1ImageStack, T2ImageStack, Relaxometry)
from hazenlib.utils import get_dicom_files
from hazenlib.exceptions import ArgumentCombinationError
from tests import TEST_DATA_DIR, TEST_REPORT_DIR
from hazenlib.relaxometry_params import TEMPLATE_VALUES


class TestRelaxometry(unittest.TestCase):
    # test parameters here

    # Values for transform_coords tests
    TEST_COORDS = [[0, 0], [0, 1], [1, 2]]
    COORDS_FLIP = [[0, 0], [1, 0], [2, 1]]
    COORDS_TRANS = [[3, 1], [3, 2], [4, 3]]
    COORDS_TRANS_FLIP = [[3, 1], [4, 1], [5, 2]]
    COORDS_TRANS_COL_ROW = [[1, 3], [1, 4], [2, 5]]
    COORDS_TRANS_ROTATE = [[10, 20], [10.5, 20 + np.sqrt(3) / 2],
                           [11 + np.sqrt(3) / 2, 19.5 + np.sqrt(3)]]

    # T1_FILES are in random order to test sorting    
    T1_DIR = os.path.join(TEST_DATA_DIR, 'relaxometry', 'T1', 'site1_20200218',
                          'plate5')
    T1_FILES = ['20530320', '20530224', '20530416', '20530272', '20530464',
                '20530368']
    T1_TI_SORTED = [50.0, 100.0, 200.0, 400.0, 600.0, 800.0]

    # T2_FILES are in random order to test sorting
    T2_DIR = os.path.join(TEST_DATA_DIR, 'relaxometry', 'T2', 'site1_20200218',
                          'plate4')
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
        TEST_DATA_DIR, 'relaxometry', 'T1', 'site1_20200218', 'plate5',
        '20530224')
    TEMPLATE_TEST_COORDS_ROW_COL = [[56, 95], [62, 117], [81, 133], [104, 134],
                                    [124, 121], [133, 98], [127, 75],
                                    [109, 61], [84, 60], [64, 72], [80, 81],
                                    [78, 111], [109, 113], [110, 82], [97, 43]]
    TEMPLATE_TARGET_COORDS_COL_ROW = [[97, 58], [119, 65], [134, 85],
                                      [133, 108], [119, 127], [96, 134],
                                      [73, 127], [60, 109], [60, 84], [73, 65],
                                      [81, 81], [112, 81], [112, 112],
                                      [81, 111], [43, 97]]

    # Mask generation
    MASK_POI_TEMPLATE = np.zeros((14, 192, 192))

    for i in range(14):
        MASK_POI_TEMPLATE[i, TEMPLATE_TEST_COORDS_ROW_COL[i][0],
                          TEMPLATE_TEST_COORDS_ROW_COL[i][1]] = 1
    MASK_POI_TARGET = np.zeros((14, 192, 192))
    for i in range(14):
        MASK_POI_TARGET[i, TEMPLATE_TARGET_COORDS_COL_ROW[i][1],
                        TEMPLATE_TARGET_COORDS_COL_ROW[i][0]] = 1

    ROI0_TEMPLATE_PIXELS = [-620, -706, -678, -630, -710, -672, -726, -684,
                            -714, -654, -692, -702, -644, -738, -668, -652,
                            -744, -702, -702, -658, -664, -672, -658, -668,
                            -738]

    # Only check first three ROIs
    ROI_TEMPLATE_MEANS_T0 = [-683.840, -819.28, -1019.84]

    # Values from IDL routine
    PLATE5_T1 = [1862.6, 1435.8, 999.9, 740.7, 498.2, 351.6, 255.2, 178.0,
                 131.7, 93.3, 66.6, 45.1, 32.5, 22.4, 2632]

    # Values from testing (to check for variations)
    PLATE4_T2 = [ 816.33093, 590.26915, 430.69191, 310.801929, 216.998622,
                  155.68107, 109.346141, 78.967168, 55.986545, 39.080988,
                  27.59972,   18.389453, 12.336855, 8.035046, 2374.533555]

    TEMPLATE_PATH_T2 = os.path.join(TEST_DATA_DIR, 'relaxometry', 'T2',
                                    'Template_plate4_T2')

    TEMPLATE_P4_TEST_COORDS_ROW_COL = [[56, 95], [62, 117], [81, 133],
                                       [104, 134], [124, 121], [133, 98],
                                       [127, 75], [109, 61], [84, 60],
                                       [64, 72], [80, 81], [78, 111],
                                       [109, 113], [110, 82], [148, 118]]

    # Values from IDL routine. Site 2 T1 values are signed not magnitude
    SITE2_PLATE5_T1 = [1880.5, 1432.3, 1012.0, 742.5, 504.0, 354.4, 256.3,
                       178.9, 131.9, 93.8, 67.9, 45.8, 33.4, 23.6, 2700]

    SITE2_T1_DIR = os.path.join(TEST_DATA_DIR, 'relaxometry', 'T1',
                                'site2_20180925', 'plate5')
    SITE2_T1_FILES = ['77189804', '77189870', '77189936', '77190002',
                      '77190068', '77190134']

    # Site 4 values from Philips scanner
    SITE4_T1_P4_DIR = os.path.join(TEST_DATA_DIR, 'relaxometry', 'T1',
                                   'site4_philips', 'plate4')

    SITE4_T1_P4_FILES = ['IM_0604', 'IM_0614', 'IM_0624', 'IM_0634', 'IM_0644',
                         'IM_0654']

    SITE4_T1_P4 = [2207.79, 1977.82, 1824.62, 1508.57, 1215.05,
                   986.90, 767.73, 592.15, 443.97, 324.34, 235.42,
                   162.76, 121.01, 85.62, 3219]

    SITE4_T1_P5_DIR = os.path.join(TEST_DATA_DIR, 'relaxometry', 'T1',
                                   'site4_philips', 'plate5')

    SITE4_T1_P5_FILES = ['IM_0603', 'IM_0613', 'IM_0623', 'IM_0633', 'IM_0643',
                         'IM_0653']

    SITE4_T1_P5 = [1856.54, 1414.10, 973.43, 706.07, 500.16, 354.23, 251.70,
                   175.92, 129.89, 91.89, 65.44, 45.51, 32.39, 23.12, 2587]

    SITE4_T2_P4_DIR = os.path.join(TEST_DATA_DIR, 'relaxometry', 'T2',
                                   'site4_philips', 'plate4')

    SITE4_T2_P4_FILES = ['IM_0439', 'IM_0440', 'IM_0441', 'IM_0442', 'IM_0443',
                         'IM_0444', 'IM_0445', 'IM_0446', 'IM_0447', 'IM_0448',
                         'IM_0449', 'IM_0450', 'IM_0451', 'IM_0452', 'IM_0453',
                         'IM_0454', 'IM_0455', 'IM_0456', 'IM_0457', 'IM_0458',
                         'IM_0459', 'IM_0460', 'IM_0461', 'IM_0462', 'IM_0463',
                         'IM_0464', 'IM_0465', 'IM_0466', 'IM_0467', 'IM_0468',
                         'IM_0469', 'IM_0470']

    SITE4_T2_P4 = [830.93, 597.30, 437.69, 313.22, 220.64, 157.19, 110.05, 78.48,
                   55.53, 39.33, 27.20, 18.24, 13.17, 9.38, 2254]

    SITE4_T2_P5_DIR = os.path.join(TEST_DATA_DIR, 'relaxometry', 'T2',
                                   'site4_philips', 'plate5')

    SITE4_T2_P5_FILES = ['IM_0407', 'IM_0408', 'IM_0409', 'IM_0410', 'IM_0411',
                         'IM_0412', 'IM_0413', 'IM_0414', 'IM_0415', 'IM_0416',
                         'IM_0417', 'IM_0418', 'IM_0419', 'IM_0420', 'IM_0421',
                         'IM_0422', 'IM_0423', 'IM_0424', 'IM_0425', 'IM_0426',
                         'IM_0427', 'IM_0428', 'IM_0429', 'IM_0430', 'IM_0431',
                         'IM_0432', 'IM_0433', 'IM_0434', 'IM_0435', 'IM_0436',
                         'IM_0437', 'IM_0438']

    SITE4_T2_P5 = [1637.27, 1210.97, 844.49, 615.70, 445.75, 313.76, 223.06,
                   155.87, 114.94, 81.03, 57.27, 39.23, 28.33, 19.89, 2295]

    # Data to test 256*256 input image with 192*192 template
    PATH_256_MATRIX = os.path.join(TEST_DATA_DIR, 'relaxometry', 'T1',
                                   'site3_ge', 'plate4', 'Z675')

    TARGET_COORDS_256 = np.array([[75, 125], [85, 155], [111, 173],
                                  [142, 175], [168, 155], [177, 127],
                                  [168, 95], [142, 76], [110, 76],
                                  [85, 95], [106, 106], [105, 146],
                                  [146, 147], [147, 105], [198, 151]])

    # Site 3 from GE scanner
    SITE3_T1_P4_VALS = [1688.86, 1719.20, 1630.20, 1434.49, 1200.83, 991.49, 774.25,
                        595.16, 443.76, 325.61, 234.83, 164.16, 121.54, 73.16,
                        2962]
    SITE3_T1_P4_DIR = os.path.join(TEST_DATA_DIR, 'relaxometry', 'T1',
                                   'site3_ge', 'plate4')
    SITE3_T1_P4_FILES = ['Z675', 'Z679', 'Z682', 'Z837', 'Z839', 'Z842']

    SITE3_T1_P5_VALS = [1702.05, 1302.77, 945.81, 692.32, 499.89, 351.64, 250.61,
                        174.67, 127.77, 89.40, 63.14, 43.45, 30.12, 17.11, 1912]
    SITE3_T1_P5_DIR = os.path.join(TEST_DATA_DIR, 'relaxometry', 'T1',
                                   'site3_ge', 'plate5')
    SITE3_T1_P5_FILES = ['Z677', 'Z678', 'Z683', 'Z838', 'Z840', 'Z844']

    SITE3_T2_P4_VALS = [942.22, 661.83, 464.63, 329.72, 229.59, 163.36, 115.27,
                        82.77, 58.49, 41.64, 28.69, 19.47, 14.25, 10.30, 3094]
    SITE3_T2_P4_DIR = os.path.join(TEST_DATA_DIR, 'relaxometry', 'T2',
                                   'site3_ge', 'plate4')
    SITE3_T2_P4_FILES = ['Z815', 'Z816', 'Z820', 'Z822', 'Z826', 'Z827', 'Z831',
                         'Z832']

    SITE3_T2_P5_VALS = [1878.80, 1227.30, 790.39, 582.16, 419.24, 298.09, 216.68,
                        154.16, 114.25, 80.52, 57.17, 39.56, 28.10, 19.72, 3456]
    SITE3_T2_P5_DIR = os.path.join(TEST_DATA_DIR, 'relaxometry', 'T2',
                                   'site3_ge', 'plate5')
    SITE3_T2_P5_FILES = ['Z812', 'Z813', 'Z814', 'Z819', 'Z823', 'Z825', 'Z830',
                         'Z834']

    # Site 5 tests - Philips 3T
    SITE5_T1_P4_DIR = os.path.join(TEST_DATA_DIR, 'relaxometry', 'T1',
                                   'site5_philips_3T', 'plate4')

    SITE5_T1_P4_FILES = ['IM_0086', 'IM_0038', 'IM_0054', 'IM_0070', 'IM_0599',
                         'IM_0485']

    SITE5_T1_P4 = [2156.1, 2013.6, 1866.8, 1573.8, 1479.4, 1031.9, 799.8,
                   617.3, 469.0, 346.8, 254.0, 180.4, 139.9, 70.4, 3279.4]

    SITE5_T1_P5_DIR = os.path.join(TEST_DATA_DIR, 'relaxometry', 'T1',
                                   'site5_philips_3T', 'plate5')

    SITE5_T1_P5_FILES = ['IM_0085', 'IM_0037', 'IM_0053', 'IM_0069', 'IM_0598',
                         'IM_0484']

    SITE5_T1_P5 = [1783.9, 1330.7, 921.6, 679.3, 486.3, 341.2, 243.0, 169.3,
                   125.0, 88.5, 72.8, 44.6, 30.3, 18.7, 2458.1]

    SITE5_T2_P4_DIR = os.path.join(TEST_DATA_DIR, 'relaxometry', 'T2',
                                   'site5_philips_3T', 'plate4')

    SITE5_T2_P4_FILES = ['IM_0532', 'IM_0533', 'IM_0534', 'IM_0535', 'IM_0536',
                         'IM_0537', 'IM_0538', 'IM_0539', 'IM_0540', 'IM_0541',
                         'IM_0542', 'IM_0543', 'IM_0544', 'IM_0545', 'IM_0546',
                         'IM_0547', 'IM_0548', 'IM_0549', 'IM_0550', 'IM_0551',
                         'IM_0552', 'IM_0553', 'IM_0554', 'IM_0555', 'IM_0556',
                         'IM_0557', 'IM_0558', 'IM_0559', 'IM_0560', 'IM_0561',
                         'IM_0562', 'IM_0563']

    SITE5_T2_P4 = [625.5, 435.9, 309.3, 216.8, 149.7, 106.3, 74.3, 51.6, 35.7,
                   25.4, 16.5, 11.0, 7.1, 4.2, 2497.2]

    SITE5_T2_P5_DIR = os.path.join(TEST_DATA_DIR, 'relaxometry', 'T2',
                                   'site5_philips_3T', 'plate5')

    SITE5_T2_P5_FILES = ['IM_0500', 'IM_0501', 'IM_0502', 'IM_0503', 'IM_0504',
                         'IM_0505', 'IM_0506', 'IM_0507', 'IM_0508', 'IM_0509',
                         'IM_0510', 'IM_0511', 'IM_0512', 'IM_0513', 'IM_0514',
                         'IM_0515', 'IM_0516', 'IM_0517', 'IM_0518', 'IM_0519',
                         'IM_0520', 'IM_0521', 'IM_0522', 'IM_0523', 'IM_0524',
                         'IM_0525', 'IM_0526', 'IM_0527', 'IM_0528', 'IM_0529',
                         'IM_0530', 'IM_0531']

    SITE5_T2_P5 = [1591.4, 1119.2, 742.7, 524.5, 370.7, 256.0, 180.5, 125.9,
                   91.2, 64.4, 45.2, 30.8, 22.0, 15.4, 2706.2]

    def test_transform_coords(self):
        # no translation, no rotation, input = yx, output = yx
        warp_matrix = np.array([[1, 0, 0], [0, 1, 0]])
        op = transform_coords(self.TEST_COORDS,
                                                warp_matrix,
                                                input_row_col=True,
                                                output_row_col=True)
        np.testing.assert_allclose(op, self.TEST_COORDS)
        # 'Identity coordinate transformation row_col (yx) -> row_col (yx)' 
        # ' failed'

        # no translation, no rotation, flip
        # input = col_row (xy), output = row_col (yx)
        op = transform_coords(self.TEST_COORDS, warp_matrix,
                                                input_row_col=False,
                                                output_row_col=True)

        np.testing.assert_allclose(op, self.COORDS_FLIP)

        # no translation, no rotation, input = col_row (xy), 
        # output = col_row (xy)
        op = transform_coords(self.TEST_COORDS, warp_matrix,
                                                input_row_col=False,
                                                output_row_col=False)
        np.testing.assert_allclose(op, self.TEST_COORDS)
        # 'Identity coordinate transformation XY -> XY failed'

        # translation x=1, y=3, no rotation, input = row_col (yx),
        # output = row_col (yx)
        warp_matrix = np.array([[1, 0, 1], [0, 1, 3]])
        op = transform_coords(self.TEST_COORDS, warp_matrix,
                                                input_row_col=True,
                                                output_row_col=True)
        np.testing.assert_allclose(op, self.COORDS_TRANS)
        # 'Translation coordinate transformation YX -> YX failed'

        # translation x=1, y=3, no rotation, input = col_row (xy),
        # output = row_col (yx)
        warp_matrix = np.array([[1, 0, 1], [0, 1, 3]])
        op = transform_coords(self.TEST_COORDS, warp_matrix,
                                                input_row_col=False,
                                                output_row_col=True)
        np.testing.assert_allclose(op, self.COORDS_TRANS_FLIP)
        # 'Translation coordinate transformation XY -> YX failed'

        # translation x=1, y=3, no rotation, input = col_row (xy),
        # output = col_row (xy)
        warp_matrix = np.array([[1, 0, 1], [0, 1, 3]])
        op = transform_coords(self.TEST_COORDS, warp_matrix,
                                                input_row_col=False,
                                                output_row_col=False)
        np.testing.assert_allclose(op, self.COORDS_TRANS_COL_ROW)
        # 'Translation coordinate transformation XY -> XY failed'

        # rotation (-30) degrees, translation col=10, row=20,
        # input = col_row (xy), output = col_row (xy)
        warp_matrix = np.array([[np.sqrt(3) / 2, 0.5, 10],
                                [-0.5, np.sqrt(3) / 2, 20]])
        # use float64 rather than int32 for coordinates to better test rotation
        op = transform_coords(np.array(self.TEST_COORDS,
                                                         dtype=np.float64),
                                                warp_matrix,
                                                input_row_col=False,
                                                output_row_col=False)
        np.testing.assert_allclose(op, self.COORDS_TRANS_ROTATE)

    def test_template_fit(self):
        template_dcm = pydicom.read_file(self.TEMPLATE_PATH_T1_P5)

        target_dcm = pydicom.dcmread(self.TEMPLATE_TARGET_PATH_T1_P5)
        t1_image_stack = T1ImageStack([target_dcm])
        warp_matrix = t1_image_stack.template_fit(template_dcm)

        transformed_coordinates_xy = transform_coords(
            self.TEMPLATE_TEST_COORDS_ROW_COL, warp_matrix,
            input_row_col=True, output_row_col=False)

        # test to within +/- 1 pixel (also checks YX-XY change)
        np.testing.assert_allclose(
            transformed_coordinates_xy, self.TEMPLATE_TARGET_COORDS_COL_ROW,
            atol=1)

    def test_image_stack_T1_sort(self):
        # read list of un-ordered T1 files, sort by TI, test sorted
        t1_dcms = [pydicom.dcmread(os.path.join(self.T1_DIR, fname))
                   for fname in self.T1_FILES]
        t1_image_stack = T1ImageStack(t1_dcms)
        sorted_output = [image.InversionTime.real for image in
                         t1_image_stack.images]
        assert sorted_output == self.T1_TI_SORTED

    def test_image_stack_T2_sort(self):
        # read list of un-ordered T2 files, sort by TE, test sorted
        t2_dcms = [pydicom.dcmread(os.path.join(self.T2_DIR, fname))
                   for fname in self.T2_FILES]
        t2_image_stack = T2ImageStack(t2_dcms)

        sorted_output = [image.EchoTime.real for image in
                         t2_image_stack.images]

        assert sorted_output == self.T2_TE_SORTED

    def test_generate_time_series_template_POIs(self):
        # Test on template first, no image fitting needed
        # Need image to get correct size
        template_dcm = pydicom.dcmread(self.TEMPLATE_PATH_T1_P5)
        template_image_stack = T1ImageStack([template_dcm])
        warp_matrix = template_image_stack.template_fit(template_dcm)
        template_image_stack.generate_time_series(
            self.TEMPLATE_TEST_COORDS_ROW_COL,
            warp_matrix=warp_matrix, fit_coords=False)

        for i in range(np.size(self.MASK_POI_TEMPLATE, 0)):
            np.testing.assert_equal(
                template_image_stack.ROI_time_series[i].POI_mask,
                self.MASK_POI_TEMPLATE[i])

    def test_generate_time_series_target_POIs(self):
        # Test on target and check image fitting too.
        template_dcm = pydicom.read_file(self.TEMPLATE_PATH_T1_P5)

        target_dcm = pydicom.dcmread(self.TEMPLATE_TARGET_PATH_T1_P5)
        target_image_stack = T1ImageStack([target_dcm])
        warp_matrix = target_image_stack.template_fit(template_dcm)
        # transformed_coordinates_yx = transform_coords(
        #     self.TEMPLATE_TEST_COORDS_YX, target_image_stack.warp_matrix,
        #     input_yx=True, output_yx=True)
        target_image_stack.generate_time_series(
            self.TEMPLATE_TEST_COORDS_ROW_COL,
            warp_matrix=warp_matrix)
        for i in range(np.size(self.MASK_POI_TARGET, 0)):
            np.testing.assert_equal(
                target_image_stack.ROI_time_series[i].POI_mask,
                self.MASK_POI_TARGET[i])

    def test_extract_single_roi(self):
        # Test that ROI pixel value extraction works. Use template DICOM for
        # both template and image to avoid errors due to slight variation in
        # fitting.
        template_dcm = pydicom.read_file(self.TEMPLATE_PATH_T1_P5)

        template_image_stack = T1ImageStack([template_dcm])
        # set warp_matrix to identity matrix
        # template_image_stack.warp_matrix = np.eye(2, 3, dtype=np.float32)
        warp_matrix = template_image_stack.template_fit(template_dcm)
        template_image_stack.generate_time_series(
            self.TEMPLATE_TEST_COORDS_ROW_COL,
            warp_matrix=warp_matrix, fit_coords=False)

        np.testing.assert_equal(
            template_image_stack.ROI_time_series[0].pixel_values[0],
            self.ROI0_TEMPLATE_PIXELS)

    def test_template_roi_means(self):
        # Check mean of first 3 ROIs in template match with ImageJ calculations
        template_dcm = pydicom.read_file(self.TEMPLATE_PATH_T1_P5)

        template_image_stack = T1ImageStack([template_dcm])

        warp_matrix = template_image_stack.template_fit(template_dcm)
        template_image_stack.generate_time_series(
            self.TEMPLATE_TEST_COORDS_ROW_COL,
            warp_matrix=warp_matrix, fit_coords=False)

        # Check all pixels in ROI[0] match
        np.testing.assert_allclose(
            template_image_stack.ROI_time_series[0].pixel_values[0],
            self.ROI0_TEMPLATE_PIXELS)

        # Check mean ROI for first three ROIs are correct
        for i in range(len(self.ROI_TEMPLATE_MEANS_T0)):
            self.assertAlmostEqual(
                np.mean(template_image_stack.ROI_time_series[i].pixel_values),
                self.ROI_TEMPLATE_MEANS_T0[i])

    def test_t1_calc_magnitude_image(self):
        """Test T1 value for plate 5 spheres."""
        template_dcm = pydicom.read_file(self.TEMPLATE_PATH_T1_P5)
        t1_dcms = [pydicom.dcmread(os.path.join(self.T1_DIR, fname))
                   for fname in self.T1_FILES]
        t1_image_stack = T1ImageStack(t1_dcms)
        warp_matrix = t1_image_stack.template_fit(template_dcm)
        t1_image_stack.generate_time_series(
            self.TEMPLATE_TEST_COORDS_ROW_COL,
            warp_matrix=warp_matrix)
        t1_image_stack.generate_fit_function()
        t1_published = TEMPLATE_VALUES['plate5']['t1']['relax_times']['1.5T']
        s0_est = t1_image_stack.initialise_fit_parameters(t1_estimates=t1_published)

        t1_image_stack.find_relax_times(
            t1_estimates=t1_published, s0_est=s0_est)

        np.testing.assert_allclose(t1_image_stack.relax_times, self.PLATE5_T1,
                                   rtol=0.02, atol=1)

    def test_t2_calc_magnitude_image(self):
        """Test T2 value for plate 4 spheres."""
        template_dcm = pydicom.read_file(self.TEMPLATE_PATH_T2)
        t2_dcms = [pydicom.dcmread(os.path.join(self.T2_DIR, fname))
                   for fname in self.T2_FILES]
        t2_image_stack = T2ImageStack(t2_dcms)
        warp_matrix = t2_image_stack.template_fit(template_dcm)
        t2_image_stack.generate_time_series(
            self.TEMPLATE_P4_TEST_COORDS_ROW_COL,
            warp_matrix=warp_matrix)
        t2_published = TEMPLATE_VALUES['plate4']['t2']['relax_times']['1.5T']
        s0_est = t2_image_stack.initialise_fit_parameters(
            t2_estimates=t2_published)
        t2_image_stack.find_relax_times(
            t2_estimates=t2_published, s0_est=s0_est)

        np.testing.assert_allclose(t2_image_stack.relax_times, self.PLATE4_T2,
                                   rtol=0.01, atol=1)

    def test_t1_calc_signed_image(self):
        """Test T1 value for signed plate 5 spheres (site 2)."""
        template_dcm = pydicom.read_file(self.TEMPLATE_PATH_T1_P5)
        t1_dcms = [pydicom.dcmread(os.path.join(self.SITE2_T1_DIR, fname))
                   for fname in self.SITE2_T1_FILES]
        t1_image_stack = T1ImageStack(t1_dcms)
        warp_matrix = t1_image_stack.template_fit(template_dcm)
        t1_image_stack.generate_time_series(
            self.TEMPLATE_TEST_COORDS_ROW_COL,
            warp_matrix=warp_matrix)
        t1_image_stack.generate_fit_function()
        t1_published = TEMPLATE_VALUES['plate5']['t1']['relax_times']['1.5T']
        s0_est = t1_image_stack.initialise_fit_parameters(
            t1_estimates=t1_published)
        t1_image_stack.find_relax_times(
            t1_estimates=t1_published, s0_est=s0_est)

        np.testing.assert_allclose(t1_image_stack.relax_times, self.SITE2_PLATE5_T1,
                                   rtol=0.02, atol=1)

    def test_t1_siemens(self):
        """Test T1 values on Siemens images."""
        dcms = get_dicom_files(self.T1_DIR)
        # dcms = [pydicom.dcmread(os.path.join(self.T1_DIR, fname)) for fname in
        #         self.T1_FILES]
        task = Relaxometry(input_data=dcms)
        results = task.run(plate_number=5, calc="T1", verbose=True)
        np.testing.assert_allclose(
                    results['additional data']['calc_times'], self.PLATE5_T1,
                    rtol=0.02, atol=1)

    def test_t1_p4_philips(self):
        """Test T1 values on plate 4 on Philips."""
        dcms = get_dicom_files(self.SITE4_T1_P4_DIR)
        # dcms = [pydicom.dcmread(os.path.join(self.SITE4_T1_P4_DIR, fname))
        #         for fname in self.SITE4_T1_P4_FILES]
        task = Relaxometry(input_data=dcms)
        results = task.run(plate_number=4, calc="T1", verbose=True)
        np.testing.assert_allclose(
                    results['additional data']['calc_times'], self.SITE4_T1_P4,
                    rtol=0.02, atol=1)

    def test_t1_p5_philips(self):
        """Test T1 values on plate 5 on Philips."""
        dcms = get_dicom_files(self.SITE4_T1_P5_DIR)
        # dcms = [pydicom.dcmread(os.path.join(self.SITE4_T1_P5_DIR, fname))
        #         for fname in self.SITE4_T1_P5_FILES]
        task = Relaxometry(input_data=dcms)
        results = task.run(plate_number=5, calc="T1", verbose=True)
        np.testing.assert_allclose(
                    results['additional data']['calc_times'], self.SITE4_T1_P5,
                    rtol=0.02, atol=1)

    def test_t2_p4_philips(self):
        """Test T2 values on plate 4 on Philips."""
        dcms = get_dicom_files(self.SITE4_T2_P4_DIR)
        # dcms = [pydicom.dcmread(os.path.join(self.SITE4_T2_P4_DIR, fname))
        #         for fname in self.SITE4_T2_P4_FILES]
        task = Relaxometry(input_data=dcms)
        results = task.run(plate_number=4, calc="T2", verbose=True)
        np.testing.assert_allclose(
                    results['additional data']['calc_times'], self.SITE4_T2_P4,
                    rtol=0.02, atol=1)

    def test_t2_p5_philips(self):
        """Test T2 values on plate 4 on Philips."""
        dcms = get_dicom_files(self.SITE4_T2_P5_DIR)
        # dcms = [pydicom.dcmread(os.path.join(self.SITE4_T2_P5_DIR, fname))
        #         for fname in self.SITE4_T2_P5_FILES]
        task = Relaxometry(input_data=dcms)
        results = task.run(plate_number=5, calc="T2", verbose=True)
        np.testing.assert_allclose(
                    results['additional data']['calc_times'], self.SITE4_T2_P5,
                    rtol=0.02, atol=1)

    def test_scale_up_template(self):
        """Test fit for 256x256 GE image with 192x192 template"""
        template_dcm = pydicom.read_file(
                            TEMPLATE_VALUES['plate4']['t1']['filename'])

        target_dcm = pydicom.dcmread(self.PATH_256_MATRIX)
        t1_image_stack = T1ImageStack([target_dcm])
        warp_matrix = t1_image_stack.template_fit(template_dcm)

        transformed_coordinates_xy = transform_coords(
            TEMPLATE_VALUES['plate4']['sphere_centres_row_col'],
            warp_matrix, input_row_col=True, output_row_col=True)

        # test to within +/- 1 pixel (also checks YX-XY change)
        np.testing.assert_allclose(
            transformed_coordinates_xy, self.TARGET_COORDS_256,
            atol=1)

    def test_ge(self):
        """Test relaxometry.py values on GE."""
        for plate in (4, 5):
            for tparam in ['T1', 'T2']:
                dcms = get_dicom_files(
                    getattr(self, f'SITE3_{tparam}_P{plate}_DIR'))
                # dcms = [pydicom.dcmread(os.path.join(
                #     getattr(self, f'SITE3_{tparam}_P{plate}_DIR'), fname))
                #     for fname in getattr(self, f'SITE3_{tparam}_P{plate}_FILES')]
                task = Relaxometry(input_data=dcms)
                results = task.run(plate_number=plate,
                                calc = tparam, verbose=True)
                np.testing.assert_allclose(
                    results['additional data']['calc_times'],
                    getattr(self, f'SITE3_{tparam}_P{plate}_VALS'),
                    rtol=0.02, atol=1)

    # # This type of expection is raised automatically at the arg parse level
    # def test_plate_number_not_specified(self):
    #     """Test exception raised if plate_number not specified."""
    #     self.assertRaises(ArgumentCombinationError,
    #                       Relaxometry.run, [], calc="T1")

    def test_philips_3T(self):
        """Test calculation on 3T dataset."""

        # T1 plate 4
        dcms = get_dicom_files(self.SITE5_T1_P4_DIR)
        # dcms = [pydicom.dcmread(os.path.join(self.SITE5_T1_P4_DIR, fname))
        #         for fname in self.SITE5_T1_P4_FILES]
        task = Relaxometry(input_data=dcms)
        results = task.run(plate_number=4, calc="T1", verbose=True)
        np.testing.assert_allclose(
                    results['additional data']['calc_times'], self.SITE5_T1_P4,
                    rtol=0.02, atol=1)

        # T1 plate 5
        dcms = get_dicom_files(self.SITE5_T1_P5_DIR)
        # dcms = [pydicom.dcmread(os.path.join(self.SITE5_T1_P5_DIR, fname))
        #         for fname in self.SITE5_T1_P5_FILES]
        task = Relaxometry(input_data=dcms)
        results = task.run(plate_number=5, calc="T1", verbose=True)
        np.testing.assert_allclose(
                    results['additional data']['calc_times'], self.SITE5_T1_P5,
                    rtol=0.02, atol=1)

        # T2 plate 4
        dcms = get_dicom_files(self.SITE5_T2_P4_DIR)
        # dcms = [pydicom.dcmread(os.path.join(self.SITE5_T2_P4_DIR, fname))
        #         for fname in self.SITE5_T2_P4_FILES]
        task = Relaxometry(input_data=dcms)
        results = task.run(plate_number=4, calc="T2", verbose=True)
        np.testing.assert_allclose(
                    results['additional data']['calc_times'], self.SITE5_T2_P4,
                    rtol=0.02, atol=1)

        # T2 plate 5
        dcms = get_dicom_files(self.SITE5_T2_P5_DIR)
        # dcms = [pydicom.dcmread(os.path.join(self.SITE5_T2_P5_DIR, fname))
        #         for fname in self.SITE5_T2_P5_FILES]
        task = Relaxometry(input_data=dcms)
        results = task.run(plate_number=5, calc="T2", verbose=True)
        np.testing.assert_allclose(
                    results['additional data']['calc_times'], self.SITE5_T2_P5,
                    rtol=0.02, atol=1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
