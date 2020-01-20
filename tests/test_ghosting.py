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


class TestGhosting(unittest.TestCase):
    SIGNAL_BOUNDING_BOX = (242, 325, 251, 335)
    SIGNAL_CENTRE = [284, 293]
    SIGNAL_SLICE = np.array(range(284 - 5, 284 + 5), dtype=np.intp)[:, np.newaxis], np.array(range(293 - 5, 293 + 5),
                                                                                             dtype=np.intp)
    BACKGROUND_ROIS = [(88, 293), (88, 220), (88, 147), (88, 74)]
    PADDING_FROM_BOX = 30
    SLICE_RADIUS = 5
    PE = 'ROW'
    ELIGIBLE_GHOST_AREA = range(SIGNAL_BOUNDING_BOX[0], SIGNAL_BOUNDING_BOX[1]), range(
        5, SIGNAL_BOUNDING_BOX[2] - PADDING_FROM_BOX)

    GHOST_SLICE = np.array(range(283 - SLICE_RADIUS, 283 + SLICE_RADIUS), dtype=np.intp)[:, np.newaxis], np.array(
        range(13 - SLICE_RADIUS, 13 + SLICE_RADIUS)
    )
    GHOSTING = 0.787020192832454

    def setUp(self):
        self.file = str(TEST_DATA_DIR / 'ghosting' / 'GHOSTING' / 'IM_0001.dcm')
        self.dcm = pydicom.read_file(self.file)

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

        # # Create figure and axes
        # fig, ax = plt.subplots(1)
        #
        # # Display the image
        # ax.imshow(self.dcm.pixel_array)
        # rect = patches.Rectangle((left_column, upper_row),
        #                          right_column-left_column, lower_row-upper_row,
        #                          linewidth=1, edgecolor='r', facecolor='none')
        # ax.add_patch(rect)
        # rect = patches.Rectangle((288, 279), 10, 10, linewidth=1, edgecolor='r', facecolor='none')
        # ax.add_patch(rect)
        # plt.show()

        assert (upper_row, lower_row, left_column, right_column) == self.SIGNAL_BOUNDING_BOX

    def test_get_signal_slice(self):
        assert list(hazen_ghosting.get_signal_slice(self.SIGNAL_BOUNDING_BOX)[0]) == list(self.SIGNAL_SLICE[0])
        assert list(hazen_ghosting.get_signal_slice(self.SIGNAL_BOUNDING_BOX)[1]) == list(self.SIGNAL_SLICE[1])

    def test_get_pe_direction(self):
        assert hazen_ghosting.get_pe_direction(self.dcm) == self.PE

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

    def test_get_eligible_area(self):
        # # Create figure and axes
        # fig, ax = plt.subplots(1)
        #
        # # Display the image
        # ax.imshow(dcm.pixel_array)
        #
        # rect = patches.Rectangle((min(eligible_columns), min(eligible_rows)),
        #                          max(eligible_columns)-min(eligible_columns),
        #                          max(eligible_rows)-min(eligible_rows),
        #                          linewidth=1, edgecolor='r', facecolor='none')
        # ax.add_patch(rect)
        assert hazen_ghosting.get_eligible_area(self.SIGNAL_BOUNDING_BOX, self.dcm) == self.ELIGIBLE_GHOST_AREA

    def test_get_ghost_slice(self):
        assert list(hazen_ghosting.get_ghost_slice(self.SIGNAL_BOUNDING_BOX, self.dcm)[0]) == list(self.GHOST_SLICE[0])
        assert list(hazen_ghosting.get_ghost_slice(self.SIGNAL_BOUNDING_BOX, self.dcm)[1]) == list(self.GHOST_SLICE[1])

    def test_get_ghosting(self):
        assert hazen_ghosting.get_ghosting(self.dcm)['ghosting_percentage'] == self.GHOSTING


class TestCOLPEGhosting(TestGhosting):
    SIGNAL_BOUNDING_BOX = (165, 210, 164, 209)
    SIGNAL_CENTRE = [187, 186]
    SIGNAL_SLICE = np.array(range(187 - 5, 187 + 5), dtype=np.intp)[:, np.newaxis], np.array(range(186 - 5, 186 + 5),
                                                                                             dtype=np.intp)
    BACKGROUND_ROIS = [(187, 64), (140, 64), (93, 64), (46, 64)]
    PADDING_FROM_BOX = 30
    SLICE_RADIUS = 5
    ELIGIBLE_GHOST_AREA = range(5, SIGNAL_BOUNDING_BOX[0]-PADDING_FROM_BOX), range(
        SIGNAL_BOUNDING_BOX[2], SIGNAL_BOUNDING_BOX[3])

    GHOST_SLICE = np.array(range(134 - SLICE_RADIUS, 134 + SLICE_RADIUS), dtype=np.intp)[:, np.newaxis], np.array(
        range(197 - SLICE_RADIUS, 197 + SLICE_RADIUS)
    )
    PE = "COL"
    GHOSTING = 0.29948100239672915

    def setUp(self):
        self.file = str(TEST_DATA_DIR / 'ghosting' / 'PE_COL_PHANTOM_BOTTOM_RIGHT' / 'PE_COL_PHANTOM_BOTTOM_RIGHT.IMA')
        self.dcm = pydicom.read_file(self.file)
