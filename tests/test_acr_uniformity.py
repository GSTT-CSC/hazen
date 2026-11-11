# Python imports
import pathlib
import unittest

# Module imports
import numpy as np

# Local imports
from hazenlib.tasks.acr_uniformity import ACRUniformity
from hazenlib.utils import get_dicom_files
from tests import TEST_DATA_DIR, TEST_REPORT_DIR, transform_series

PIU_TOLERANCE = 0.01


class TestACRUniformitySiemens(unittest.TestCase):
    """Percent integral uniformity on test data.

    Values were checked against the report images. This dataset has strong
    shading (bright rim), hence the low PIU.
    """

    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "Siemens")
    piu = 67.83

    # set once per class in setUpClass
    acr_uniformity_task: ACRUniformity
    result: float

    @classmethod
    def setUpClass(cls):
        input_files = get_dicom_files(cls.ACR_DATA)

        cls.acr_uniformity_task = ACRUniformity(
            input_data=input_files,
            report_dir=pathlib.PurePath.joinpath(TEST_REPORT_DIR),
        )
        cls.result = cls.acr_uniformity_task.get_integral_uniformity(
            cls.acr_uniformity_task.ACR_obj.slice_stack[6]
        )

    def test_roi_values_within_roi(self) -> None:
        """The minimum and maximum 1 cm2 ROIs lie fully inside the large ROI."""
        task = self.acr_uniformity_task
        kernel = task.small_roi_kernel()
        h_rows, h_cols = kernel.shape[0] // 2, kernel.shape[1] // 2
        for x, y, _ in (task.min_roi, task.max_roi):
            footprint = task.large_roi[
                y - h_rows : y + h_rows + 1, x - h_cols : x + h_cols + 1
            ]
            assert footprint[kernel].all()

    def test_small_roi_area(self) -> None:
        """The small ROI is 1 cm2 to within a few pixels."""
        task = self.acr_uniformity_task
        row_sp, col_sp = task.ACR_obj.spacing
        area = task.small_roi_kernel().sum() * row_sp * col_sp
        assert abs(area - 100) < 8

    def test_uniformity(self):
        rounded_results = round(self.result, 2)

        print("\ntest_uniformity.py::TestUniformity::test_uniformity")
        print("new_release_values:", rounded_results)
        print("fixed_values:", self.piu)

        self.assertAlmostEqual(self.result, self.piu, delta=PIU_TOLERANCE)


class TestACRUniformityGESignaArtistT1(TestACRUniformitySiemens):
    ACR_DATA = pathlib.Path(TEST_DATA_DIR) / "acr" / "GE_Signa_Artist_1.5T_T1"
    piu = 92.80


class TestACRUniformitySiemensSolaFit(TestACRUniformitySiemens):
    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "SiemensSolaFit")
    piu = 96.09


class TestACRUniformityPhilipsAchieva(TestACRUniformitySiemens):
    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "PhilipsAchieva")
    piu = 81.27


class TestACRUniformityGE(TestACRUniformitySiemens):
    """GE data with a 512 matrix (0.51 mm pixels)."""

    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "GE")
    piu = 84.60


class TestACRUniformityGESigna3TT2(TestACRUniformitySiemens):
    """The phantom sits high in the field of view and the lowest-signal
    1 cm2 ROI used to overlap the blurred edge of the notch (79.73%)."""

    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "GE_Signa_3T_T2")
    piu = 81.02

    def test_notch_excluded(self):
        """No large ROI pixel lies within 2 mm of the dark notch."""
        task = self.acr_uniformity_task
        _, rescaled, _ = task.ACR_obj.get_presentation_pixels(
            task.ACR_obj.slice_stack[6]
        )
        img = np.asarray(rescaled, dtype=float)
        roi = task.large_roi
        dark = img < task.DARK_FRACTION * np.median(img[roi])
        assert not (roi & dark).any()
        # part of the 195 cm2 circle was cut out, all of it at the top
        row_sp, col_sp = task.ACR_obj.spacing
        assert roi.sum() * row_sp * col_sp < task.LARGE_ROI_AREA - 20
        rows, cols = np.nonzero(roi)
        centre_row = rows.mean()
        cut_rows = rows.min()
        assert centre_row - cut_rows < task.r_large / row_sp


class TestACRUniformityRun(unittest.TestCase):
    def test_run(self):
        task = ACRUniformity(
            input_data=get_dicom_files(TEST_DATA_DIR / "acr" / "Siemens"),
        )
        values = [m.value for m in task.run().measurements]
        assert len(values) == 3  # PIU, minimum and maximum ROI means
        self.assertAlmostEqual(
            values[0], TestACRUniformitySiemens.piu, delta=PIU_TOLERANCE
        )


class TestACRUniformityOrientation(unittest.TestCase):
    """A series exported rotated by a multiple of 90 degrees, or mirrored,
    must give the same PIU and the same ROI positions in the standard view."""

    ACR_DATA = pathlib.Path(TEST_DATA_DIR / "acr" / "PhilipsAchieva")

    def test_rotated_and_mirrored(self):
        files = get_dicom_files(self.ACR_DATA)
        values = []
        for k, flip in ((0, False), (1, True)):
            task = ACRUniformity(input_data=files)
            transform_series(task.ACR_obj, k, flip)
            piu = task.get_integral_uniformity(task.ACR_obj.slice_stack[6])
            values.append((piu, task.min_roi, task.max_roi))
        assert values[1] == values[0]


class TestACRUniformitySynthetic(unittest.TestCase):
    """A disc with a linear left-right signal ramp.

    The mean of a circular ROI on a linear ramp is the value at its centre,
    so the extreme 1 cm2 ROIs are centred (R_large - r_small) either side of
    the large ROI centre, and PIU = 100 * (1 - slope * d / level) with
    d = 78.79 - 5.64 = 73.15 mm. The slope is chosen to give 90%.
    """

    LEVEL = 1000.0
    D = np.sqrt(19500 / np.pi) - np.sqrt(100 / np.pi)
    SLOPE = 0.1 * LEVEL / D  # per mm

    def setUp(self):
        self.task = ACRUniformity(
            input_data=get_dicom_files(TEST_DATA_DIR / "acr" / "Siemens"),
            orientation=(0, False),
        )

    def make_slice(self, spacing):
        row_sp, col_sp = spacing
        ny, nx = int(250 / row_sp), int(250 / col_sp)
        y = (np.arange(ny) - (ny - 1) / 2) * row_sp
        x = (np.arange(nx) - (nx - 1) / 2) * col_sp
        xx, yy = np.meshgrid(x, y)
        img = np.where(
            np.hypot(xx, yy) < 95, self.LEVEL + self.SLOPE * xx, 0.0
        )
        return img

    def check(self, spacing, tolerance, stored=None):
        """``spacing`` is that of the standard view; ``stored`` that in the
        DICOM, if the series was exported rotated."""
        self.task.ACR_obj.dx, self.task.ACR_obj.dy = stored or spacing
        img = self.make_slice(spacing)
        centre = ((img.shape[1] - 1) / 2, (img.shape[0] - 1) / 2)
        roi = self.task.large_roi_mask(img, centre)
        x_min, y_min, v_min, x_max, y_max, v_max = (
            self.task.get_mean_roi_values(img, roi)
        )
        piu = self.task.calculate_uniformity(v_min, v_max)
        assert abs(piu - 90.0) < tolerance, piu
        assert x_min < centre[0] < x_max  # dark on the left, bright right

    def test_square_pixels(self):
        self.check((1.0, 1.0), 0.15)

    def test_half_millimetre_pixels(self):
        self.check((0.5, 0.5), 0.08)

    def test_non_square_pixels(self):
        self.check((1.0, 0.8), 0.15)

    def test_non_square_pixels_rotated_series(self):
        """Exported rotated by 90 degrees, so the stored row and column
        spacings are swapped relative to the standard view."""
        self.task = ACRUniformity(
            input_data=get_dicom_files(TEST_DATA_DIR / "acr" / "Siemens"),
            orientation=(3, False),
        )
        self.check((1.0, 0.8), 0.15, stored=(0.8, 1.0))
