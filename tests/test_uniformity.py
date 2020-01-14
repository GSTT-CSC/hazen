import unittest
import pathlib
import json

import pytest
import pydicom

import hazenlib.uniformity as hazen_uniformity
from tests import TEST_DATA_DIR


class TestSlicePosition(unittest.TestCase):

    UNIFORMITY_DATA = pathlib.Path(TEST_DATA_DIR / 'uniformity')

    def setUp(self):
        self.test_file = [str(self.UNIFORMITY_DATA / 'axial_oil.IMA')]

    def test_uniformity(self):
        results = hazen_uniformity.main(self.test_file)

        assert results['uniformity']['horizontal']['IPEM'] == 1.0
        assert results['uniformity']['vertical']['IPEM'] == 0.98125
