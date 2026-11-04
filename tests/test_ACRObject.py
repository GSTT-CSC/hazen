import os
import unittest
from unittest.mock import patch

import numpy as np

from hazenlib.ACRObject import ACRObject
from hazenlib.logger import logger
from hazenlib.tasks.acr_geometric_accuracy import ACRGeometricAccuracy
from hazenlib.tasks.acr_ghosting import ACRGhosting
from hazenlib.tasks.acr_low_contrast_object_detectability import (
    ACRLowContrastObjectDetectability,
)
from hazenlib.tasks.acr_slice_position import ACRSlicePosition
from hazenlib.tasks.acr_slice_thickness import ACRSliceThickness
from hazenlib.tasks.acr_snr import ACRSNR
from hazenlib.tasks.acr_uniformity import ACRUniformity
from hazenlib.utils import dcmread, get_dicom_files
from tests import TEST_DATA_DIR, TEST_REPORT_DIR


# Siemens (axial)
class TestACRTools(unittest.TestCase):
    centre = (129, 129)
    rotation = -1.0
    # Mask lengths, edge to edge (N pixels = N * spacing)
    horizontal_distance = 192.3828125
    horizontal_end = (129, 255)
    vertical_distance = 188.4765625
    vertical_end = (255, 128)

    def setUp(self):
        self.Siemens_data = [
            dcmread(os.path.join(TEST_DATA_DIR, "acr", "Siemens", f"{i}"))
            for i in os.listdir(os.path.join(TEST_DATA_DIR, "acr", "Siemens"))
        ]

        self.ACR_object = ACRObject(self.Siemens_data)
        self.img1 = self.ACR_object.slice_stack[0].pixel_array
        self.img7 = self.ACR_object.slice_stack[6].pixel_array

    def test_find_centre(self):
        phantom_centre, _ = self.ACR_object.find_phantom_center(
            self.img7, self.ACR_object.dx, self.ACR_object.dy
        )
        assert self.centre == phantom_centre

    def test_find_rotation(self):
        rotation_angle = self.ACR_object.determine_rotation(self.img1)
        assert self.rotation == np.round(rotation_angle, 1)

    def test_measure_orthogonal_lengths(self):
        cxy, _ = self.ACR_object.find_phantom_center(
            self.img1, self.ACR_object.dx, self.ACR_object.dy
        )
        mask = self.ACR_object.get_mask_image(self.img1, cxy)
        length_dict = self.ACR_object.measure_orthogonal_lengths(mask, cxy)
        # We report the distances with 2 significant figures of precision.
        assert np.round(self.horizontal_distance, 2) == np.round(
            length_dict["Horizontal Distance"], 2
        )
        assert self.horizontal_end == length_dict["Horizontal End"]
        assert np.round(self.vertical_distance, 2) == np.round(
            length_dict["Vertical Distance"], 2
        )
        assert self.vertical_end == length_dict["Vertical End"]


# GE axial
class TestACRToolsGE(TestACRTools):
    rotation = 0.0
    centre = (254, 255)
    horizontal_distance = 191.4406
    horizontal_end = (256, 511)
    vertical_distance = 189.4094
    vertical_end = (511, 255)
    test_point = (-60.98, -45.62)

    def setUp(self):
        self.GE_data = [
            dcmread(os.path.join(TEST_DATA_DIR, "acr", "GE", f"{i}"))
            for i in os.listdir(os.path.join(TEST_DATA_DIR, "acr", "GE"))
        ]

        self.ACR_object = ACRObject(self.GE_data)
        self.img1 = self.ACR_object.slice_stack[0].pixel_array
        self.img7 = self.ACR_object.slice_stack[6].pixel_array

    def test_find_centre(self):
        phantom_centre, _ = self.ACR_object.find_phantom_center(
            self.img7, self.ACR_object.dx, self.ACR_object.dy
        )
        assert self.centre == phantom_centre

    def test_rotate_point(self):
        rotated_point = np.array(
            self.ACR_object.rotate_point((0, 0), (30, 70), 150)
        )
        rotated_point = np.round(rotated_point, 2)
        assert (rotated_point == self.test_point).all()


# Philips Achieva axial
class TestACRToolsPhilips(TestACRTools):
    rotation = 0.0
    centre = (129, 128)
    horizontal_distance = 191.40625
    horizontal_end = (128, 255)
    vertical_distance = 190.4296875
    vertical_end = (255, 129)

    def setUp(self):
        self.Philips_data = [
            dcmread(
                os.path.join(TEST_DATA_DIR, "acr", "PhilipsAchieva", f"{i}")
            )
            for i in os.listdir(
                os.path.join(TEST_DATA_DIR, "acr", "PhilipsAchieva")
            )
        ]

        self.ACR_object = ACRObject(self.Philips_data)
        self.img1 = self.ACR_object.slice_stack[0].pixel_array
        self.img7 = self.ACR_object.slice_stack[6].pixel_array

    def test_find_centre(self):
        phantom_centre, _ = self.ACR_object.find_phantom_center(
            self.img7, self.ACR_object.dx, self.ACR_object.dy
        )
        assert self.centre == phantom_centre


# Siemens Magnetom Sola Fit axial
class TestACRToolsSiemensSolaFit(TestACRTools):
    rotation = 0.0
    centre = (128, 127)
    horizontal_distance = 191.406152
    horizontal_end = (127, 255)
    vertical_distance = 191.41
    vertical_end = (255, 128)

    def setUp(self):
        self.Philips_data = [
            dcmread(
                os.path.join(TEST_DATA_DIR, "acr", "SiemensSolaFit", f"{i}")
            )
            for i in os.listdir(
                os.path.join(TEST_DATA_DIR, "acr", "SiemensSolaFit")
            )
        ]

        self.ACR_object = ACRObject(self.Philips_data)
        self.img1 = self.ACR_object.slice_stack[0].pixel_array
        self.img7 = self.ACR_object.slice_stack[6].pixel_array

    def test_find_centre(self):
        phantom_centre, _ = self.ACR_object.find_phantom_center(
            self.img7, self.ACR_object.dx, self.ACR_object.dy
        )
        assert self.centre == phantom_centre


class TestMaskImageStrongIntensityVariation(unittest.TestCase):
    """get_mask_image on images that take its large-intensity-variation branch.

    That branch used to return the Sauvola threshold *map* as the mask, so the
    mask covered the whole image. The synthetic phantom is a 190 x 100 mm
    rectangle that is bright on the left and falls to a few percent of the
    maximum on the right (strong coil shading), in magnitude-image noise.
    """

    def setUp(self):
        data = [
            dcmread(os.path.join(TEST_DATA_DIR, "acr", "Siemens", f))
            for f in os.listdir(os.path.join(TEST_DATA_DIR, "acr", "Siemens"))
        ]
        self.acr = ACRObject(data)
        self.acr.dx = self.acr.dy = 1.0

    @staticmethod
    def make_phantom(floor, noise_sd=4.0, zero_background=False, seed=0):
        n = 256
        x = np.arange(n) - n / 2
        xx, yy = np.meshgrid(x, x)
        inside = (np.abs(xx) < 95) & (np.abs(yy) < 50)
        shade = floor + (1 - floor) / (1 + np.exp((xx - 10) / 12))
        rng = np.random.default_rng(seed)
        noise = np.hypot(
            rng.normal(0, noise_sd, xx.shape),
            rng.normal(0, noise_sd, xx.shape),
        )
        if zero_background:  # most background pixels exactly zero
            noise[rng.random(xx.shape) < 0.7] = 0
        img = np.round(np.where(inside, 1000 * shade, 0) + noise)
        return img, inside

    def check(self, floor, **kwargs):
        img, truth = self.make_phantom(floor, **kwargs)
        with self.assertLogs(logger, level="WARNING") as logs:
            mask = self.acr.get_mask_image(img, (128, 128))
        assert any("Large intensity variations" in m for m in logs.output)
        assert (mask == truth).mean() > 0.999, (mask == truth).mean()

    def test_strong_shading(self):
        """The dim end of the phantom, at 2% of the maximum, is kept."""
        self.check(0.02)

    def test_no_shading(self):
        """The branch also triggers when the central disc includes
        background; a uniform phantom must still be masked correctly."""
        self.check(1.0)

    def test_zero_filled_background(self):
        """Backgrounds that are mostly exact zeros (as on some scanners)."""
        self.check(0.03, zero_background=True)


class TestMeasureOrthogonalLengthsSynthetic(unittest.TestCase):
    """measure_orthogonal_lengths on masks of known size.

    A rectangle of 150 x 80 pixels must measure 150 and 80 pixels edge to
    edge (it used to measure 149 and 79), with horizontal lengths in column
    spacing and vertical lengths in row spacing, on non-square images.
    """

    def setUp(self):
        data = [
            dcmread(os.path.join(TEST_DATA_DIR, "acr", "Siemens", f))
            for f in os.listdir(os.path.join(TEST_DATA_DIR, "acr", "Siemens"))
        ]
        self.acr = ACRObject(data)

    def test_non_square(self):
        self.acr.dx, self.acr.dy = 0.5, 0.8  # PixelSpacing (row, column)
        mask = np.zeros((300, 200), dtype=bool)  # rows, columns
        mask[100:180, 20:170] = True  # 80 rows x 150 columns
        lengths = self.acr.measure_orthogonal_lengths(mask, (95, 140))
        assert lengths["Horizontal Distance"] == 150 * 0.8
        assert lengths["Vertical Distance"] == 80 * 0.5
        assert lengths["Horizontal End"] == (140, 199)
        assert lengths["Vertical End"] == (299, 95)

    def test_offset_lines(self):
        self.acr.dx = self.acr.dy = 1.0
        mask = np.zeros((256, 256), dtype=bool)
        mask[50:200, 40:220] = True
        mask[50:120, 40:100] = False  # notch in the top-left corner
        lengths = self.acr.measure_orthogonal_lengths(
            mask, (128, 128), h_offset=(0, -40), v_offset=(-50, 0)
        )
        # horizontal line on row 88 and vertical line on column 78 both
        # run through the notch
        assert lengths["Horizontal Distance"] == 120
        assert lengths["Vertical Distance"] == 80


class TestCircularMask(unittest.TestCase):
    def test_shape_and_centre(self):
        mask = ACRObject.circular_mask((30, 10), 5, (40, 60))
        assert mask.shape == (40, 60)
        rows, cols = np.nonzero(mask)
        assert rows.mean() == 10 and cols.mean() == 30


class TestMissingSlices(unittest.TestCase):
    """Images missing from a series must not shift the slices after them.

    Slice n is always at slice_stack[n - 1] (and the last slice at [-1]);
    asking for a missing slice raises an error naming it.
    """

    @classmethod
    def setUpClass(cls):
        folder = os.path.join(TEST_DATA_DIR, "acr", "Siemens")
        cls.files = [os.path.join(folder, f) for f in os.listdir(folder)]
        complete = ACRObject([dcmread(f) for f in cls.files])
        # SOP instance UID of each ACR slice, in phantom order
        cls.uids = [d.SOPInstanceUID for d in complete.slice_stack]

    def files_without(self, *slice_numbers):
        drop = {self.uids[n - 1] for n in slice_numbers}
        return [
            f
            for f in self.files
            if dcmread(f, stop_before_pixels=True).SOPInstanceUID not in drop
        ]

    def series_without(self, *slice_numbers):
        files = self.files_without(*slice_numbers)
        return ACRObject([dcmread(f) for f in files])

    def test_complete_series_unchanged(self):
        obj = ACRObject([dcmread(f) for f in self.files])
        assert [d.SOPInstanceUID for d in obj.slice_stack] == self.uids
        assert len(obj.slice_stack) == 11

    def test_missing_slices(self):
        """The slices present stay at their index, missing ones raise and
        are named in the log, and iterating gives only the slices present."""
        for missing in ((2,), (10,), (11,), (4, 5, 9, 10, 11)):
            with self.subTest(missing=missing):
                with self.assertLogs(logger, level="WARNING") as logs:
                    obj = self.series_without(*missing)
                listed = ", ".join(str(n) for n in missing)
                several = len(missing) > 1
                message = (
                    f"slices {listed} are missing"
                    if several
                    else f"slice {listed} is missing"
                )
                assert any(message in m for m in logs.output)
                for n in range(1, 12):
                    if n in missing:
                        with self.assertRaisesRegex(
                            ValueError, f"slice {n} is missing"
                        ):
                            obj.slice_stack[n - 1]
                    else:
                        uid = obj.slice_stack[n - 1].SOPInstanceUID
                        assert uid == self.uids[n - 1]
                assert len(list(obj.slice_stack)) == 11 - len(missing)
                if 11 in missing:
                    with self.assertRaisesRegex(ValueError, "slice 11 is"):
                        obj.slice_stack[-1]
                else:
                    uid = obj.slice_stack[-1].SOPInstanceUID
                    assert uid == self.uids[-1]

    def test_positions_that_cannot_be_placed(self):
        """Without usable slice positions the order found is kept."""

        def same_position(dcms):
            for d in dcms:
                d.ImagePositionPatient = dcms[0].ImagePositionPatient

        def shift(dcm, mm):
            ipp = [float(v) for v in dcm.ImagePositionPatient]
            iop = np.array(dcm.ImageOrientationPatient, dtype=float)
            normal = np.cross(iop[:3], iop[3:])
            dcm.ImagePositionPatient = list(np.add(ipp, mm * normal))

        def off_pitch(dcms):
            shift(dcms[-1], 3.0)  # 0.3 of the 10 mm pitch

        def beyond_slice_11(dcms):
            for d in dcms[2:]:
                shift(d, 30.0)

        def unreadable(dcms):
            del dcms[3].ImagePositionPatient

        cases = [
            (same_position, "irregular positions"),
            (off_pitch, "irregular positions"),
            (beyond_slice_11, "irregular positions"),
            (unreadable, "could not be read"),
        ]
        for corrupt, message in cases:
            with self.subTest(corrupt.__name__):
                obj = self.series_without(2)
                dcms = list(obj.slice_stack)
                corrupt(dcms)
                with self.assertLogs(logger, level="WARNING") as logs:
                    stack = obj.place_slices(dcms)
                assert list(stack) == dcms
                assert len(stack) == 10
                assert any(message in m for m in logs.output)

    def test_slicing_across_a_gap(self):
        obj = self.series_without(9)
        uids = [d.SOPInstanceUID for d in obj.slice_stack[:8]]
        assert uids == self.uids[:8]
        for index in (slice(8, None), slice(10, 6, -1)):
            with (
                self.subTest(index=index),
                self.assertRaisesRegex(ValueError, "slice 9 is missing"),
            ):
                obj.slice_stack[index]

    def test_lcod_without_slice_9(self):
        task = ACRLowContrastObjectDetectability(
            input_data=self.files_without(9),
            report_dir=TEST_REPORT_DIR,
        )
        with self.assertRaisesRegex(ValueError, "slice 9 is missing"):
            task.run()

    def test_missing_slice_1(self):
        """Without slice 1 the numbering cannot be anchored: keep the order
        found (as before) and warn."""
        with self.assertLogs(logger, level="WARNING") as logs:
            obj = self.series_without(1)
        assert any("slice 1 could not be identified" in m for m in logs.output)
        assert len(obj.slice_stack) == 10


class TestSeriesOrientation(unittest.TestCase):
    """The orientation is detected once per series, or given."""

    FILES = get_dicom_files(TEST_DATA_DIR / "acr" / "Siemens")
    DETECT = "hazenlib.ACRObject.determine_phantom_orientation"

    def test_detected_once_per_series(self):
        task = ACRUniformity(input_data=self.FILES)
        with patch(self.DETECT, return_value=(0, False)) as detect:
            for _ in range(2):
                task.get_integral_uniformity(task.ACR_obj.slice_stack[6])
        detect.assert_called_once()

    def test_given_orientation_is_used(self):
        tasks = (
            ACRGeometricAccuracy,
            ACRGhosting,
            ACRSlicePosition,
            ACRSliceThickness,
            ACRSNR,
            ACRUniformity,
        )
        for task_class in tasks:
            with (
                self.subTest(task_class.__name__),
                patch(self.DETECT) as detect,
            ):
                task = task_class(input_data=self.FILES, orientation=(1, True))
                assert task.ACR_obj.orientation == (1, True)
                # odd k swaps the row and column spacing
                task.ACR_obj.dx, task.ACR_obj.dy = 0.5, 1.0
                assert task.ACR_obj.spacing == (1.0, 0.5)
                detect.assert_not_called()

    def test_snr_subtraction_given_both_orientations(self):
        """Each series gets its own given orientation."""
        task = ACRSNR(
            input_data=self.FILES,
            subtract=TEST_DATA_DIR / "acr" / "Siemens2",
            orientation=(0, False),
            orientation2=(2, False),
        )
        with (
            patch(self.DETECT) as detect,
            patch.object(
                ACRSNR, "snr_by_subtraction", return_value=(1.0, 1.0)
            ) as subtract,
        ):
            task.run()
        detect.assert_not_called()
        assert task.ACR_obj.orientation == (0, False)
        assert subtract.call_args.kwargs["acr_obj2"].orientation == (2, False)
