import unittest

import numpy as np
import pydicom
import pytest
import os
import pathlib
from tests import TEST_DATA_DIR, TEST_REPORT_DIR
from hazenlib.tasks.ghosting import Ghosting
from hazenlib.utils import get_dicom_files


class TestGhosting(unittest.TestCase):
    SIGNAL_BOUNDING_BOX = (252, 334, 243, 325)
    SIGNAL_CENTRE = [293, 284]
    BACKGROUND_ROIS = [(293, 88), (220, 88), (147, 88), (74, 88)]
    PADDING_FROM_BOX = 30
    SLICE_RADIUS = 5
    PE = 'ROW'
    ELIGIBLE_GHOST_AREA = range(5, SIGNAL_BOUNDING_BOX[0] - PADDING_FROM_BOX), range(SIGNAL_BOUNDING_BOX[2],
                                                                                     SIGNAL_BOUNDING_BOX[3])

    SIGNAL_SLICE = np.array(range(SIGNAL_CENTRE[0] - SLICE_RADIUS, SIGNAL_CENTRE[0] + SLICE_RADIUS), dtype=np.intp)[:,
                   np.newaxis], np.array(
        range(SIGNAL_CENTRE[1] - SLICE_RADIUS, SIGNAL_CENTRE[1] + SLICE_RADIUS), dtype=np.intp)

    GHOST_SLICE = np.array(
        range(min(ELIGIBLE_GHOST_AREA[1]), max(ELIGIBLE_GHOST_AREA[1])), dtype=np.intp)[:, np.newaxis], np.array(
        range(min(ELIGIBLE_GHOST_AREA[0]), max(ELIGIBLE_GHOST_AREA[0])))

    GHOSTING = 0.11803264099090763

    def setUp(self):
        self.dcm = pydicom.read_file(os.path.join(TEST_DATA_DIR, 'ghosting', 'GHOSTING', 'IM_0001.dcm'))
        self.ghosting = Ghosting(input_data=get_dicom_files(os.path.join(TEST_DATA_DIR, 'ghosting', 'GHOSTING')),
                                 report_dir=pathlib.PurePath.joinpath(TEST_REPORT_DIR))

    def test_calculate_ghost_intensity(self):
        with pytest.raises(Exception):
            _ = self.ghosting.calculate_ghost_intensity([], [], [])

        with pytest.raises(Exception):
            self.ghosting.calculate_ghost_intensity(-1, 100, 5)

        with pytest.raises(Exception):
            self.ghosting.calculate_ghost_intensity(ghost=np.asarray([-10]),
                                                    phantom=np.asarray([-100]),
                                                    noise=np.asarray([-5]))

        assert 5.0 == self.ghosting.calculate_ghost_intensity(ghost=np.asarray([10]),
                                                              phantom=np.asarray([100]),
                                                              noise=np.asarray([5]))

    def test_get_signal_bounding_box(self):
        (left_column, right_column, upper_row, lower_row,) = self.ghosting.get_signal_bounding_box(self.dcm.pixel_array)
        assert (left_column, right_column, upper_row, lower_row) == self.SIGNAL_BOUNDING_BOX

    def test_get_signal_slice(self):
        assert list(self.ghosting.get_signal_slice(self.SIGNAL_BOUNDING_BOX)[0]) == list(self.SIGNAL_SLICE[0])
        assert list(self.ghosting.get_signal_slice(self.SIGNAL_BOUNDING_BOX)[1]) == list(self.SIGNAL_SLICE[1])

    def test_get_pe_direction(self):
        assert self.ghosting.get_pe_direction(self.dcm) == self.PE

    def test_get_background_rois(self):
        assert self.ghosting.get_background_rois(self.dcm, self.SIGNAL_CENTRE) == self.BACKGROUND_ROIS

    def test_get_eligible_area(self):
        assert self.ghosting.get_eligible_area(self.SIGNAL_BOUNDING_BOX, self.dcm) == self.ELIGIBLE_GHOST_AREA

    def test_get_ghost_slice(self):
        assert list(self.ghosting.get_ghost_slice(self.SIGNAL_BOUNDING_BOX, self.dcm)[0]) == list(self.GHOST_SLICE[0])
        assert list(self.ghosting.get_ghost_slice(self.SIGNAL_BOUNDING_BOX, self.dcm)[1]) == list(self.GHOST_SLICE[1])

    def test_get_ghosting(self):
        ghosting_val = self.ghosting.get_ghosting(self.dcm)

        print("\ntest_get_ghosting.py::TestGetGhosting::test_get_ghosting")
        print("new_release_value:", ghosting_val)
        print("fixed_value:", self.GHOSTING)

        assert ghosting_val == self.GHOSTING


class TestCOLPEGhosting(TestGhosting):
    SIGNAL_BOUNDING_BOX = (164, 208, 166, 209)
    SIGNAL_CENTRE = [186, 187]
    BACKGROUND_ROIS = [(64, 187), (64, 140), (64, 93), (64, 46)]
    PADDING_FROM_BOX = 30
    SLICE_RADIUS = 5
    ELIGIBLE_GHOST_AREA = range(SIGNAL_BOUNDING_BOX[0], SIGNAL_BOUNDING_BOX[1]), range(
        SLICE_RADIUS, SIGNAL_BOUNDING_BOX[2] - PADDING_FROM_BOX)

    SIGNAL_SLICE = np.array(range(SIGNAL_CENTRE[0] - SLICE_RADIUS, SIGNAL_CENTRE[0] + SLICE_RADIUS), dtype=np.intp)[:,
                   np.newaxis], \
                   np.array(range(SIGNAL_CENTRE[1] - SLICE_RADIUS, SIGNAL_CENTRE[1] + SLICE_RADIUS), dtype=np.intp)

    GHOST_SLICE = np.array(
        range(min(ELIGIBLE_GHOST_AREA[1]), max(ELIGIBLE_GHOST_AREA[1])), dtype=np.intp)[:, np.newaxis], np.array(
        range(min(ELIGIBLE_GHOST_AREA[0]), max(ELIGIBLE_GHOST_AREA[0])))

    PE = "COL"
    GHOSTING = 0.015138960417776908

    def setUp(self):
        self.dcm = pydicom.read_file(
            os.path.join(TEST_DATA_DIR, 'ghosting', 'PE_COL_PHANTOM_BOTTOM_RIGHT', 'PE_COL_PHANTOM_BOTTOM_RIGHT.IMA'))
        self.ghosting = Ghosting(
            input_data=get_dicom_files(os.path.join(TEST_DATA_DIR, 'ghosting', 'PE_COL_PHANTOM_BOTTOM_RIGHT')),
            report_dir=pathlib.PurePath.joinpath(TEST_REPORT_DIR))


class TestAxialPhilipsGhosting(TestGhosting):
    SIGNAL_BOUNDING_BOX = (217, 299, 11, 93)
    SIGNAL_CENTRE = [(SIGNAL_BOUNDING_BOX[0] + SIGNAL_BOUNDING_BOX[1]) // 2,
                     (SIGNAL_BOUNDING_BOX[2] + SIGNAL_BOUNDING_BOX[3]) // 2]
    BACKGROUND_ROIS = [(258, 264), (194, 264), (130, 264), (66, 264)]
    PADDING_FROM_BOX = 30
    SLICE_RADIUS = 5
    ELIGIBLE_GHOST_AREA = range(SLICE_RADIUS, SIGNAL_BOUNDING_BOX[0] - PADDING_FROM_BOX), range(
        SIGNAL_BOUNDING_BOX[2], SIGNAL_BOUNDING_BOX[3])
    SIGNAL_SLICE = np.array(range(SIGNAL_CENTRE[0] - SLICE_RADIUS, SIGNAL_CENTRE[0] + SLICE_RADIUS), dtype=np.intp)[:,
                   np.newaxis], \
                   np.array(range(SIGNAL_CENTRE[1] - SLICE_RADIUS, SIGNAL_CENTRE[1] + SLICE_RADIUS), dtype=np.intp)
    GHOST_SLICE = np.array(
        range(min(ELIGIBLE_GHOST_AREA[1]), max(ELIGIBLE_GHOST_AREA[1])), dtype=np.intp)[:, np.newaxis], np.array(
        range(min(ELIGIBLE_GHOST_AREA[0]), max(ELIGIBLE_GHOST_AREA[0])))

    GHOSTING = 0.007246960909896829

    def setUp(self):
        self.dcm = pydicom.read_file(
            os.path.join(TEST_DATA_DIR, 'ghosting', 'GHOSTING', 'axial_philips_ghosting.dcm'))
        self.ghosting = Ghosting(input_data=get_dicom_files(os.path.join(TEST_DATA_DIR, 'ghosting', 'GHOSTING')),
                                 report_dir=pathlib.PurePath.joinpath(TEST_REPORT_DIR))
