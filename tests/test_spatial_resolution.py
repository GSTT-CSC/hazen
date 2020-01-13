import unittest
import pathlib

import numpy as np
import pydicom
import pylab
import cv2 as cv

import hazenlib.spatial_resolution as hazen_spatial_resolution
from tests import TEST_DATA_DIR


class TestSpatialResolution(unittest.TestCase):
    RESOLUTION_DATA = pathlib.Path(TEST_DATA_DIR / 'resolution')
    dicom = pydicom.read_file(str(RESOLUTION_DATA / 'RESOLUTION' / "ax_512_resolution.dcm"))

    def test_rescale_to_byte(self):
        img = hazen_spatial_resolution.rescale_to_byte(self.dicom.pixel_array)
        assert img.max() <= 255

    def test_thresh_image(self):
        img = hazen_spatial_resolution.rescale_to_byte(self.dicom.pixel_array)
        thresh = hazen_spatial_resolution.thresh_image(img)
        assert np.count_nonzero(thresh) < np.count_nonzero(img)

    def test_find_square(self):
        img = hazen_spatial_resolution.rescale_to_byte(self.dicom.pixel_array)
        thresh = hazen_spatial_resolution.thresh_image(img)
        square = hazen_spatial_resolution.find_square(thresh)
        # squares = cv.drawContours(img.copy(), [square], -1, (0, 255, 0), 2)
        # pylab.imshow(squares, cmap=pylab.cm.gray)
        # pylab.show()

        assert np.testing.assert_allclose(square, [[286, 301], [205, 287], [219, 206], [300, 220]]) is None

    def test_get_bisecting_normals(self):
        img = hazen_spatial_resolution.rescale_to_byte(self.dicom.pixel_array)
        thresh = hazen_spatial_resolution.thresh_image(img)
        square = hazen_spatial_resolution.find_square(thresh)
        vector = {"x": square[3][0] - square[0][0], "y": square[3][1] - square[0][1]}
        centre = {"x": square[0][0] + int(vector["x"] / 2), "y": square[0][1] + int(vector["y"] / 2)}
        n1x, n1y, n2x, n2y = hazen_spatial_resolution.get_bisecting_normal(vector, centre)
        assert (n1x, n1y, n2x, n2y) == (313, 264, 273, 258)

        # # Print over image
        # squares = cv.drawContours(img.copy(), [square], -1, (0, 255, 0), 2)
        # line_img = squares.copy()
        # cv.line(line_img, (n1x, n1y), (n2x, n2y), (0, 0, 255), 5)
        # pylab.imshow(line_img, cmap=pylab.cm.gray)
        # pylab.show()

    def test_create_line_iterator(self):
        pass

    def test_get_right_edge_normal_profile(self):
        spacing = self.dicom.PixelSpacing[0]
        img = hazen_spatial_resolution.rescale_to_byte(self.dicom.pixel_array)
        square = hazen_spatial_resolution.find_square(img)

        profile = hazen_spatial_resolution.get_right_edge_normal_profile(img, square)

        # dx = [x * spacing * 10 for x in range(len(profile[::-1]))]
        # pylab.plot(dx, profile[::-1])

    def test_get_void_arr(self):
        pixels = self.dicom.pixel_array
        img = hazen_spatial_resolution.rescale_to_byte(pixels)
        circle = hazen_spatial_resolution.get_circles(img)
        void_arr = hazen_spatial_resolution.get_void_roi(pixels, circle)

        assert np.mean(void_arr) == 12.515

    def test_calculate_mtf(self):
        res = hazen_spatial_resolution.calculate_mtf(self.dicom)
        assert res == 0.5322287345688292
