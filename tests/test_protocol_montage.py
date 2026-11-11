"""Tests for the montage of report overlays made by session protocols."""

from __future__ import annotations

import contextlib
import io
import logging
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from matplotlib.figure import Figure
from PIL import Image as PILImage

import hazenlib
from hazenlib.orchestration import init_task
from hazenlib.protocols.acr_gstt import ACRGSTTProtocol
from hazenlib.protocols.outputs import Tile, make_montage
from hazenlib.types import Measurement, Result
from hazenlib.utils import get_dicom_files, wait_on_parallel_results
from tests import TEST_DATA_DIR, mock_detect_orientation

SIEMENS_T1 = TEST_DATA_DIR / "acr" / "Siemens_Sola_1.5T_T1"

RED, BLUE = (255, 0, 0), (0, 0, 255)
WHITE = (255, 255, 255)


class TestMakeMontage(unittest.TestCase):
    """Tiles are laid out on a grid of rows and figure columns."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.paths = {}
        for name, colour in (("red", RED), ("blue", BLUE)):
            self.paths[name] = str(self.tmp / f"{name}.png")
            PILImage.new("RGB", (64, 64), colour).save(self.paths[name])

    def montage(self, tiles: list[Tile]) -> np.ndarray:
        out = self.tmp / "sub" / "montage.png"
        make_montage(tiles, out)
        return np.asarray(PILImage.open(out).convert("RGB"))

    @staticmethod
    def cell(img: np.ndarray, row: int, col: int, shape: tuple[int, int]):
        """Split a montage into the image area and caption of one cell."""
        height, width = img.shape[0] // shape[0], img.shape[1] // shape[1]
        block = img[
            row * height : (row + 1) * height,
            col * width : (col + 1) * width,
        ]
        # Cells are a caption strip above a square image area
        return block[height - width :], block[: height - width]

    def test_grid(self) -> None:
        img = self.montage(
            [
                Tile("HEAD_TRA", "Uniformity", "cap", self.paths["red"]),
                Tile("HEAD_TRA", "Ghosting", "cap", self.paths["blue"]),
                # Body has no ghosting, and its uniformity failed
                Tile("BODY_TRA", "Uniformity", "cap", None),
            ],
        )

        def centre(row: int, col: int) -> tuple[int, ...]:
            area, _ = self.cell(img, row, col, (2, 2))
            return tuple(area[area.shape[0] // 2, area.shape[1] // 2])

        self.assertEqual(centre(0, 0), RED)
        self.assertEqual(centre(0, 1), BLUE)
        # A failed or skipped step is a grey tile; no step is a gap
        failed = centre(1, 0)
        self.assertEqual(len(set(failed)), 1)
        self.assertNotEqual(failed, WHITE)
        self.assertEqual(centre(1, 1), WHITE)

    def test_captions(self) -> None:
        img = self.montage(
            [
                Tile(
                    "R", "A", "HEAD_TRA: Ghosting = 0.12%", self.paths["red"]
                ),
                Tile("R", "B", "", self.paths["red"]),
            ],
        )
        _, with_caption = self.cell(img, 0, 0, (1, 2))
        _, without = self.cell(img, 0, 1, (1, 2))
        self.assertTrue((with_caption != 255).any())
        self.assertTrue((without == 255).all())


class TestTaskOverlays(unittest.TestCase):
    """Each protocol task saves its annotated phantom panel on its own."""

    def test_no_title_or_legend(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp)
        task = init_task(
            "acr_uniformity",
            get_dicom_files(str(SIEMENS_T1))[:1],
            report_dir=tmp,
        )
        # All black apart from a white title and a white legend box
        fig = Figure(facecolor="black")
        ax = fig.add_subplot()
        ax.imshow(np.zeros((8, 8)), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        ax.plot([1, 6], [1, 6], c="black", label="Horizontal: 190 mm")
        ax.legend(facecolor="white", framealpha=1)
        ax.set_title("Geometric Accuracy", color="white")

        task.save_overlay(fig, ax, tmp / "figure.png")

        overlay = PILImage.open(task.overlays[None]).convert("L")
        self.assertEqual(Path(task.overlays[None]).name, "figure_overlay.png")
        self.assertLess(np.asarray(overlay).max(), 50)

    def test_overlays(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp)
        files = get_dicom_files(str(SIEMENS_T1))
        # The slice each overlay shows, where a task has several figures
        for task_name, slices in {
            "acr_slice_thickness": {None},
            "acr_slice_position": {1, 11},
            "acr_geometric_accuracy": {1, 5},
            "acr_uniformity": {None},
            "acr_ghosting": {None},
            "acr_snr": {None},
        }.items():
            with self.subTest(task=task_name):
                result = init_task(
                    task_name, files, report=True, report_dir=tmp
                ).run()
                self.assertEqual(set(result.overlays), slices)
                # Overlays are not report images: they are not opened
                # after a single task or put in the workbook
                self.assertFalse(
                    set(result.overlays.values()) & set(result.report_images),
                )
                if len(slices) > 1:
                    # Each overlay is of its own slice's figure
                    self.assertEqual(
                        [result.overlays[n] for n in sorted(slices)],
                        [
                            image.removesuffix(".png") + "_overlay.png"
                            for image in result.report_images
                        ],
                    )
                for overlay in result.overlays.values():
                    width, height = PILImage.open(overlay).size
                    # The square phantom panel, not the whole wide or
                    # tall figure
                    self.assertAlmostEqual(width / height, 1, delta=0.15)


def distance(slice_number: int, name: str, value: float) -> Measurement:
    return Measurement(
        "GeometricAccuracy",
        value,
        subtype=f"Slice {slice_number} {name} distance",
        unit="mm",
    )


MEASUREMENTS = {
    "acr_slice_thickness": [
        Measurement("SliceWidth", 4.98, subtype="slice width", unit="mm"),
        Measurement(
            "SliceWidth",
            50.0,
            subtype="Ramps",
            unit="mm",
            visibility="intermediate",
        ),
    ],
    "acr_slice_position": [
        Measurement(
            "SlicePosition",
            value,
            subtype="length difference",
            description=f"Slice {n} series_1_{n}",
        )
        for n, value in ((1, -0.001), (11, -1.1))
    ],
    "acr_geometric_accuracy": [
        # Nothing for slice 1: its measurement failed
        distance(5, "Horizontal", 190.2),
        distance(5, "Vertical", 189.9),
        distance(5, "Diagonal SW", 190.3),
        distance(5, "Diagonal SE", 190.0),
    ],
    "acr_uniformity": [
        Measurement(
            "Uniformity", 97.0, subtype="Integral uniformity", unit="%"
        ),
    ],
    # No final measurement, so the step fails but keeps its overlay
    "acr_ghosting": [
        Measurement(
            "Ghosting",
            1000.0,
            subtype="ROI mean",
            visibility="intermediate",
        ),
    ],
    "acr_snr": [
        Measurement("SNR", 1500.0, type="normalised", subtype="smoothing"),
        Measurement("SNR", 2000.0, type="normalised", subtype="subtraction"),
    ],
}
OVERLAY_SLICES = {
    "acr_slice_position": (1, 11),
    "acr_geometric_accuracy": (1, 5),
}


def head_session(test: unittest.TestCase) -> Path:
    """Session folder with the five head roles, removed after the test."""
    mock_detect_orientation(test)
    root = Path(tempfile.mkdtemp()).resolve()
    test.addCleanup(shutil.rmtree, root)
    for n in range(1, 6):
        (root / f"Head_{n}").symlink_to(SIEMENS_T1)
    return root


def fake_init_task(task_name: str, files: list, **kwargs) -> Mock:
    """Stand-in for init_task: canned measurements and overlay paths."""
    result = Result(task=task_name, _load_metadata=False)
    for m in MEASUREMENTS[task_name]:
        result.add_measurement(m)
    for n in OVERLAY_SLICES.get(task_name, (None,)):
        overlay = Path(kwargs["report_dir"]) / f"{task_name}_{n}.png"
        overlay.parent.mkdir(parents=True, exist_ok=True)
        PILImage.new("RGB", (8, 8), RED).save(overlay)
        result.overlays[n] = str(overlay)
    task = Mock()
    task.run.return_value = result
    return task


@patch("hazenlib.protocols.engine.init_task", side_effect=fake_init_task)
class TestProtocolTiles(unittest.TestCase):
    """A protocol run gives one captioned tile per overlay."""

    def test_tiles(self, _init: Mock) -> None:
        tmp = head_session(self)
        protocol = ACRGSTTProtocol(tmp, report_dir=tmp / "out")
        tiles = protocol.tiles(protocol.run(debug=True))

        head = {t.column: t for t in tiles if t.row == "HEAD_TRA"}
        self.assertEqual(
            [t.caption for t in tiles if t.row == "HEAD_TRA"],
            [
                "HEAD_TRA: Slice Thickness = 4.98mm",
                "HEAD_TRA: Slice Position (slice 1) = 0.00",
                "HEAD_TRA: Slice Position (slice 11) = -1.10",
                "HEAD_TRA: Geometric Accuracy (slice 1) = failed",
                "HEAD_TRA: Geometric Accuracy (slice 5)"
                " = H 190.20, V 189.90, SW 190.30, SE 190.00 mm",
                "HEAD_TRA: Uniformity = 97.00%",
                "HEAD_TRA: Ghosting = failed",
                # Subtraction is preferred when both methods ran
                "HEAD_TRA: SNR = 2000.00",
            ],
        )
        # Each tile shows the overlay of its own slice and role
        self.assertEqual(
            head["Slice Position (slice 11)"].path,
            f"{tmp}/out/Head_4/acr_slice_position_11.png",
        )
        self.assertEqual(
            head["Ghosting"].path,
            f"{tmp}/out/Head_1/acr_ghosting_None.png",
        )

        # Rows are the results table's columns, in the same order, and
        # skipped steps keep their tile; the body coil has no ghosting
        self.assertEqual(
            list(dict.fromkeys(t.row for t in tiles)),
            ["HEAD_TRA", "BODY_TRA", "BODY_SAG", "BODY_COR"],
        )
        body = [t for t in tiles if t.row == "BODY_SAG"]
        self.assertEqual(len(body), len(head) - 1)
        self.assertEqual(
            (body[0].caption, body[0].path),
            ("BODY_SAG: Slice Thickness = skipped", None),
        )


@patch("hazenlib.protocols.engine.init_task", side_effect=fake_init_task)
@patch(
    "hazenlib.protocols.engine.wait_on_parallel_results",
    # Mocked tasks do not survive into worker processes, so run in-process
    side_effect=lambda f, args, **_: wait_on_parallel_results(
        f, args, debug=True
    ),
)
class TestCliMontage(unittest.TestCase):
    """``hazen acr_gstt`` writes one montage and opens only that."""

    def test_montage_written_and_shown(self, *_mocks: Mock) -> None:
        root = head_session(self)
        montage = root / "hazen_report" / "montage.png"
        warning = "--show has no effect with --no-report"

        # flags: (montage written, montage opened, warned)
        for flags, (written, opened, warned) in {
            (): (True, True, False),
            ("--no-show",): (True, False, False),
            ("--no-report",): (False, False, False),
            ("--no-report", "--show"): (False, False, True),
        }.items():
            with self.subTest(flags=flags):
                montage.unlink(missing_ok=True)
                argv = ["hazen", "acr_gstt", str(root), *flags]
                with (
                    patch.object(sys, "argv", argv),
                    patch("hazenlib.open_report_images") as opener,
                    contextlib.redirect_stdout(io.StringIO()) as out,
                    self.assertLogs("hazenlib", "WARNING") as logs,
                    # The mocked ghosting step fails, so hazen exits with 1
                    self.assertRaises(SystemExit),
                ):
                    # assertLogs needs at least one record
                    logging.getLogger("hazenlib").warning("ran")
                    hazenlib.main()
                self.assertEqual(montage.exists(), written)
                self.assertEqual(
                    f"Montage: {montage}" in out.getvalue(),
                    written,
                )
                self.assertEqual(
                    opener.call_args_list,
                    [((([montage],)), {})] if opened else [],
                )
                self.assertEqual(warning in "".join(logs.output), warned)


if __name__ == "__main__":
    unittest.main()
