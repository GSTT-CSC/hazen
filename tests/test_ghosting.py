import unittest
import pathlib

import numpy as np
import pydicom
import pytest

import hazenlib.ghosting as hazen_ghosting
from tests import TEST_DATA_DIR


class TestSliceWidth(unittest.TestCase):
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
        (upper_row, lower_row, left_column, right_columns) = hazen_ghosting.get_signal_bounding_box(self.dcm.pixel_array)
        assert (upper_row, lower_row, left_column, right_columns) == (242, 325, 251, 335)