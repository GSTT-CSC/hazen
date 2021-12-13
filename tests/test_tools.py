import unittest
import os

import numpy as np
import pydicom

import hazenlib.tools as hazen_tools
from tests import TEST_DATA_DIR


class TestTools(unittest.TestCase):
    SMALL_CIRCLE_PHANTOM_FILE = str(
        TEST_DATA_DIR / 'ghosting' / 'PE_COL_PHANTOM_BOTTOM_RIGHT' / 'PE_COL_PHANTOM_BOTTOM_RIGHT.IMA')
    small_circle_x, small_circle_y, small_circle_r = 1, 1, 22.579364776611328

    LARGE_CIRCLE_PHANTOM_FILE = str(
        TEST_DATA_DIR / 'uniformity' / 'axial_oil.IMA'
    )
    large_circle_x, large_circle_y, large_circle_r = 128, 123, 97.84805297851562

    SAG_RECTANGLE_PHANTOM_FILE = str(
        TEST_DATA_DIR / 'uniformity' / 'sag.dcm'
    )
    rectangle_size = (177.0, 204.0)
    rectangle_angle = 0
    rectangle_centre = (130.5, 135.0)

    COR_RECTANGLE_PHANTOM_FILE = str(
        TEST_DATA_DIR / 'uniformity' / 'cor.dcm'
    )
    cor_rectangle_size = (204.18031311035156, 193.64901733398438)
    cor_rectangle_angle = -89.1756591796875
    cor_rectangle_centre = (128.3773956298828, 136.2716064453125)

    COR2_RECTANGLE_PHANTOM_FILE = str(
        TEST_DATA_DIR / 'uniformity' / 'cor2.dcm'
    )
    cor2_rectangle_size = (194.4591522216797, 200.846435546875)
    cor2_rectangle_angle = -1.618094563484192
    cor2_rectangle_centre = (127.2344970703125, 129.80128479003906)


# @pytest.mark.skip
class TestShapeDetector(TestTools):

    def setUp(self) -> None:
        pass

    def test_large_circle(self):
        arr = pydicom.read_file(self.LARGE_CIRCLE_PHANTOM_FILE).pixel_array
        shape_detector = hazen_tools.ShapeDetector(arr=arr)
        x, y, r = shape_detector.get_shape('circle')
        assert int(x), int(y) == (self.large_circle_x, self.large_circle_y)
        assert round(r) == round(self.large_circle_r)

    def test_small_circle(self):
        arr = pydicom.read_file(self.SMALL_CIRCLE_PHANTOM_FILE).pixel_array
        shape_detector = hazen_tools.ShapeDetector(arr=arr)
        x, y, r = shape_detector.get_shape('circle')
        assert int(x), int(y) == (self.small_circle_x, self.small_circle_y)
        assert round(r) == round(self.small_circle_r)

    def test_sag_rectangle(self):
        arr = pydicom.read_file(self.SAG_RECTANGLE_PHANTOM_FILE).pixel_array
        shape_detector = hazen_tools.ShapeDetector(arr=arr)
        centre, size, angle = shape_detector.get_shape('rectangle')
        np.testing.assert_allclose(centre, self.rectangle_centre, rtol=1e-04)
        np.testing.assert_allclose(size, self.rectangle_size, rtol=1e-04)
        np.testing.assert_allclose(angle, self.rectangle_angle, rtol=1e-04)

    def test_cor_rectangle(self):
        arr = pydicom.read_file(self.COR_RECTANGLE_PHANTOM_FILE).pixel_array
        shape_detector = hazen_tools.ShapeDetector(arr=arr)
        centre, size, angle = shape_detector.get_shape('rectangle')
        np.testing.assert_allclose(centre, self.cor_rectangle_centre, rtol=1e-04)
        np.testing.assert_allclose(size, self.cor_rectangle_size, rtol=1e-04)
        np.testing.assert_allclose(angle, self.cor_rectangle_angle, rtol=1e-04)

    def test_cor2_rectangle(self):
        arr = pydicom.read_file(self.COR2_RECTANGLE_PHANTOM_FILE).pixel_array
        shape_detector = hazen_tools.ShapeDetector(arr=arr)
        centre, size, angle = shape_detector.get_shape('rectangle')
        np.testing.assert_allclose(centre, self.cor2_rectangle_centre, rtol=1e-04)
        np.testing.assert_allclose(size, self.cor2_rectangle_size, rtol=1e-04)
        np.testing.assert_allclose(angle, self.cor2_rectangle_angle, rtol=1e-04)


class Test_is_Dicom_file(unittest.TestCase):


    def test_is_dicom(self):
        folder = r"./tests/data/tools"
        files = [os.path.join(folder, x).replace("\\","/") for x in os.listdir(folder)]


        result = hazen_tools.is_dicom_file(files[1])
        self.assertTrue(result)


        result = hazen_tools.is_dicom_file(files[0])
        self.assertFalse(result)

    def test_is_dicom_yes(self):
        folder = r"./tests/data/tools"
        files = [os.path.join(folder, x).replace("\\","/") for x in os.listdir(folder)]
        result = hazen_tools.is_dicom_file(files[1])
        self.assertTrue(result)

    def test_is_dicom_no(self):
        folder = r"./tests/data/tools"
        files = [os.path.join(folder, x).replace("\\","/") for x in os.listdir(folder)]
        result = hazen_tools.is_dicom_file(files[0])
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()


