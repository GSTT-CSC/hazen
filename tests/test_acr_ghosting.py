import pathlib
import unittest

import numpy as np

from hazenlib.tasks.acr_ghosting import ACRGhosting
from hazenlib.utils import get_dicom_files

from tests import TEST_DATA_DIR, TEST_REPORT_DIR, transform_series

TOLERANCE = 0.002  # % signal ghosting


class TestACRGhostingSiemens(unittest.TestCase):
    """Percent signal ghosting on test data.

    Values were checked against the report images: all ROIs lie in the
    background, clear of the phantom and inside the reconstructed field of
    view.
    """

    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "Siemens")
    psg = 0.063
    missing = ()  # ghost ROIs with no room on this dataset

    def setUp(self):
        input_files = get_dicom_files(self.ACR_DATA)
        self.acr_ghosting_task = ACRGhosting(
            input_data=input_files,
            report_dir=pathlib.PurePath.joinpath(TEST_REPORT_DIR),
        )

    def test_ghosting(self):
        ghosting_val = round(
            self.acr_ghosting_task.get_signal_ghosting(
                self.acr_ghosting_task.ACR_obj.slice_stack[6]
            ),
            3,
        )

        print("\ntest_ghosting.py::TestGhosting::test_ghosting")
        print("new_release_value:", ghosting_val)
        print("fixed_value:", self.psg)

        assert abs(ghosting_val - self.psg) <= TOLERANCE

    def test_ghost_rois(self):
        """Ghost ROIs are 10 cm2 (or absent when there is no room)."""
        task = self.acr_ghosting_task
        task.get_signal_ghosting(task.ACR_obj.slice_stack[6])
        for side in ("top", "bottom", "left", "right"):
            if side in self.missing:
                assert side not in task.ghost_rois
                continue
            roi = task.ghost_rois[side]
            assert abs(roi.area - task.GHOST_ROI_AREA) < 1.0, side
            assert np.isfinite(roi.mean), side


class TestACRGhostingGE(TestACRGhostingSiemens):
    """GE data with a 512 matrix (0.51 mm pixels) and a zero-filled border."""

    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "GE")
    psg = 0.548


class TestACRGhostingGEMR450WT2(TestACRGhostingSiemens):
    """Clear ghosting along the left-right direction."""

    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "GE_MR450W_1.5T_T2")
    psg = 1.073


class TestACRGhostingPhilips3TdStream(TestACRGhostingSiemens):
    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "Philips3TdStream")
    psg = 0.133


class TestACRGhostingGESigna3T(TestACRGhostingSiemens):
    """The phantom almost touches the top of the field of view.

    There is no room for the top ROI, so the bottom ROI is used for both
    (the previous version returned NaN).
    """

    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "GE_Signa_3T_T1")
    psg = 0.080
    missing = ("top",)


class TestACRGhostingRun(unittest.TestCase):
    """The task reports PSG plus the five ROI means, with a report image."""

    def test_run(self):
        task = ACRGhosting(
            input_data=get_dicom_files(TEST_DATA_DIR / "acr" / "Siemens"),
            report=True,
            report_dir=pathlib.PurePath.joinpath(TEST_REPORT_DIR),
        )
        result = task.run()
        values = [m.value for m in result.measurements]
        assert len(values) == 6
        assert abs(values[0] - TestACRGhostingSiemens.psg) <= TOLERANCE
        assert len(task.report_files) == 1


class TestACRGhostingOrientation(unittest.TestCase):
    """A series exported rotated by a multiple of 90 degrees, or mirrored,
    must give the same result, with each ghost ROI still labelled by its
    position relative to the phantom in its standard view."""

    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "GE_MR450W_1.5T_T2")

    @staticmethod
    def measure(files, k, flip):
        task = ACRGhosting(input_data=files)
        transform_series(task.ACR_obj, k, flip)
        psg = task.get_signal_ghosting(task.ACR_obj.slice_stack[6])
        means = [
            task.ghost_rois[s].mean for s in ("top", "bottom", "left", "right")
        ]
        return psg, np.array(means)

    def test_rotated_and_mirrored(self):
        files = get_dicom_files(self.ACR_DATA)
        psg0, means0 = self.measure(files, 0, False)
        psg, means = self.measure(files, 1, True)
        assert abs(psg - psg0) < 1e-6
        np.testing.assert_allclose(means, means0)


class TestACRGhostingSynthetic(unittest.TestCase):
    """PSG on synthetic images of known ghosting.

    A water disc (1000) in a background of 30 above and below it and 10 to
    either side gives PSG = 100 * |(30 + 30) - (10 + 10)| / (2 * 1000) = 2%.
    """

    def setUp(self):
        self.task = ACRGhosting(
            input_data=get_dicom_files(TEST_DATA_DIR / "acr" / "Siemens"),
            orientation=(0, False),
        )

    def make_slice(self, spacing, centre_y=0.0):
        row_sp, col_sp = spacing
        ny, nx = int(250 / row_sp), int(250 / col_sp)
        y = (np.arange(ny) - ny / 2) * row_sp - centre_y
        x = (np.arange(nx) - nx / 2) * col_sp
        xx, yy = np.meshgrid(x, y)
        img = np.where(np.abs(yy) > np.abs(xx), 30.0, 10.0)
        img[np.hypot(xx, yy) < 95] = 1000.0
        return img

    def measure(self, img, spacing):
        self.task.ACR_obj.dx, self.task.ACR_obj.dy = spacing
        return self.task.measure_ghosting(img)

    def test_known_ghosting(self):
        assert (
            abs(self.measure(self.make_slice((1.0, 1.0)), (1.0, 1.0)) - 2.0)
            < 1e-6
        )

    def test_non_square_pixels(self):
        spacing = (1.0, 0.8)
        assert (
            abs(self.measure(self.make_slice(spacing), spacing) - 2.0) < 1e-6
        )
        for roi in self.task.ghost_rois.values():
            assert abs(roi.area - self.task.GHOST_ROI_AREA) < 1.0

    def test_centre_from_edges(self):
        """A chunk missing off the axes pulls the mask centroid away from
        the phantom centre, but not the centre found from its edges."""
        img = self.make_slice((1.0, 1.0))
        yy, xx = np.indices(img.shape) - 125
        img[(xx > 40) & (yy > 40) & (np.hypot(xx, yy) < 95)] = 10.0
        self.measure(img, (1.0, 1.0))
        assert np.allclose(self.task.centre, (125.0, 125.0), atol=0.1)

    def test_narrow_gap_flattens_roi(self):
        """An 18 mm gap leaves 12 mm between the 3 mm margins, so the short
        axis drops from 8.92 to 6 mm and the long axis grows to keep 10 cm2."""
        phantom = {"top": -95.0, "bottom": 95.0, "left": -95.0, "right": 95.0}
        fov = {"top": -113.0, "bottom": 125.0, "left": -125.0, "right": 125.0}
        roi = self.task.place_ghost_roi("top", phantom, fov)
        assert abs(roi.b - 6.0) < 1e-6
        assert abs(np.pi * roi.a * roi.b - 1000.0) < 1e-6
        assert abs(roi.y - (-104.0)) < 1e-6

    def test_non_square_pixels_rotated_series(self):
        """Exported rotated by 90 degrees, so the stored row and column
        spacings are swapped relative to the standard view."""
        self.task = ACRGhosting(
            input_data=get_dicom_files(TEST_DATA_DIR / "acr" / "Siemens"),
            orientation=(3, False),
        )
        img = self.make_slice((1.0, 0.8))
        assert abs(self.measure(img, (0.8, 1.0)) - 2.0) < 1e-6
        for roi in self.task.ghost_rois.values():
            assert roi.mean in (10.0, 30.0), roi

    def test_phantom_near_edge(self):
        """No room above the phantom: the bottom ROI stands in for the top."""
        img = self.make_slice((1.0, 1.0), centre_y=-22.0)
        psg = self.measure(img, (1.0, 1.0))
        assert "top" not in self.task.ghost_rois
        assert self.task.ghost_notes
        assert abs(psg - 2.0) < 1e-6
