import unittest
import pathlib

import numpy as np
import pydicom
import pytest
import matplotlib
# matplotlib.use("agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches


import hazenlib.ghosting as hazen_ghosting
from tests import TEST_DATA_DIR


class TestSliceWidth(unittest.TestCase):
    SIGNAL_BOUNDING_BOX = (242, 325, 251, 335)
    SIGNAL_CENTRE = [284, 293]
    SIGNAL_SLICE = np.array(range(284-5, 284+5), dtype=np.intp)[:, np.newaxis], np.array(range(293-5, 293+5), dtype=np.intp)
    BACKGROUND_ROIS = [(88, 293), (88, 220), (88, 147), (88, 74)]

    def setUp(self):
        self.dcm = pydicom.read_file(str(TEST_DATA_DIR / 'ghosting' / 'GHOSTING' / 'IM_0001.dcm'))

    def test_calculate_ghost_intensity(self):
        with pytest.raises(Exception):
            _ = hazen_ghosting.calculate_ghost_intensity([], [], [])

        with pytest.raises(Exception):
            hazen_ghosting.calculate_ghost_intensity(-1, 100, 5)

        with pytest.raises(Exception):
            hazen_ghosting.calculate_ghost_intensity(ghost=np.asarray([-10]),
                                                     phantom=np.asarray([-100]),
                                                     noise=np.asarray([-5]))

        assert 5.0 == hazen_ghosting.calculate_ghost_intensity(ghost=np.asarray([10]),
                                                               phantom=np.asarray([100]),
                                                               noise=np.asarray([5]))

        assert -5.0 == hazen_ghosting.calculate_ghost_intensity(ghost=np.asarray([5]),
                                                                phantom=np.asarray([100]),
                                                                noise=np.asarray([10]))

    def test_get_signal_bounding_box(self):
        (upper_row, lower_row, left_column, right_column) = hazen_ghosting.get_signal_bounding_box(self.dcm.pixel_array)

        # Create figure and axes
        fig, ax = plt.subplots(1)

        # Display the image
        ax.imshow(self.dcm.pixel_array)
        rect = patches.Rectangle((left_column, upper_row),
                                 right_column-left_column, lower_row-upper_row,
                                 linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        rect = patches.Rectangle((288, 279), 10, 10, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.show()

        assert (upper_row, lower_row, left_column, right_column) == self.SIGNAL_BOUNDING_BOX

    def test_get_signal_slice(self):
        assert list(hazen_ghosting.get_signal_slice(self.SIGNAL_BOUNDING_BOX)[0]) == list(self.SIGNAL_SLICE[0])
        assert list(hazen_ghosting.get_signal_slice(self.SIGNAL_BOUNDING_BOX)[1]) == list(self.SIGNAL_SLICE[1])

    def test_get_pe_direction(self):
        assert hazen_ghosting.get_pe_direction(self.dcm) == 'ROW'

    def test_get_background_rois(self):
        # # Create figure and axes
        # fig, ax = plt.subplots(1)
        #
        # # Display the image
        # ax.imshow(self.dcm.pixel_array)
        #
        # for roi in self.BACKGROUND_ROIS:
        #
        #     rect = patches.Rectangle((roi[1]-5, roi[0]-5), 10, 10, linewidth=1, edgecolor='r', facecolor='none')
        #     ax.add_patch(rect)
        #
        # plt.show()
        assert hazen_ghosting.get_background_rois(self.dcm, self.SIGNAL_CENTRE) == self.BACKGROUND_ROIS
