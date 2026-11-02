"""Writers for protocol results (xlsx workbook, CSV/TSV and JSON)."""

from __future__ import annotations

# Type Checking
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from hazenlib.protocols.engine import Table

# Python imports
import csv
import io
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Module imports
from openpyxl import Workbook
from openpyxl.drawing.image import Image
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from PIL import Image as PILImage

Record = dict[str, Any]

# Excel's "light red fill", for skipped or failed steps
MISSING_FILL = PatternFill("solid", fgColor="FFC7CE")

# Report images are scaled to this width (px) and laid out side by side
_IMAGE_WIDTH = 360
_IMAGE_GAP = 20
_ROW_HEIGHT = 20  # px, Excel's default row height
# Images are stored at twice their displayed width to stay sharp on zoom
_IMAGE_STORED_WIDTH = 2 * _IMAGE_WIDTH


@dataclass(frozen=True)
class ImageGroup:
    """Images shown together under one heading on an image sheet.

    Attributes:
        title: Heading written above the images, e.g. ``COR_SlicePosition``.
        paths: Image files, laid out left to right and stored as JPEG.

    """

    title: str
    paths: tuple[str, ...]


def write_workbook(
    table: Table,
    images: Mapping[str, Sequence[ImageGroup]],
    path: str | Path,
) -> None:
    """Write the results table and one image sheet per test to *path*.

    Units are kept in each cell's number format so values stay numeric,
    cells of skipped or failed steps are filled red, and each test's
    name is merged down its rows.
    """
    wb = Workbook()
    ws = wb.worksheets[0]
    ws.title = "Results"
    _write_results(ws, table)
    tests = dict.fromkeys(row.test for row in table.rows)
    for test in [*tests, *(t for t in images if t not in tests)]:
        if images.get(test):
            _add_image_sheet(wb.create_sheet(test[:31]), images[test])
    wb.save(path)


def _write_results(ws: Any, table: Table) -> None:
    ws.append(["test", "measurement", *table.columns])
    for cell in ws[1]:
        cell.font = Font(bold=True)
    ws.freeze_panes = "C2"

    # First and last row of each test, to merge its name down them
    spans: dict[str, list[int]] = {}
    for row_number, row in enumerate(table.rows, start=2):
        if row.test not in spans:
            ws.cell(row_number, 1, row.test)
        spans.setdefault(row.test, [row_number, row_number])[1] = row_number
        ws.cell(row_number, 2, row.measurement)
        for col, column in enumerate(table.columns, start=3):
            value = row.cells.get(column)
            if value is None:
                continue
            cell = ws.cell(row_number, col)
            if value.missing:
                cell.fill = MISSING_FILL
                continue
            cell.value = value.value
            if value.unit:
                cell.number_format = f'General" {value.unit}"'

    for first, last in spans.values():
        if last > first:
            ws.merge_cells(
                start_row=first,
                end_row=last,
                start_column=1,
                end_column=1,
            )
        ws.cell(first, 1).alignment = Alignment(vertical="top")
    ws.column_dimensions["A"].width = 20
    ws.column_dimensions["B"].width = 32
    for col in range(3, len(table.columns) + 3):
        ws.column_dimensions[get_column_letter(col)].width = 14


def _add_image_sheet(ws: Any, groups: Sequence[ImageGroup]) -> None:
    """Write each group as a heading row followed by its images."""
    columns = max(len(g.paths) for g in groups)
    for col in range(1, columns + 1):
        # Excel column width in characters is roughly (px - 5) / 7
        ws.column_dimensions[get_column_letter(col)].width = (
            _IMAGE_WIDTH + _IMAGE_GAP - 5
        ) / 7

    row = 1
    for group in groups:
        ws.cell(row, 1, group.title).font = Font(bold=True)
        row += 1
        tallest = 0
        for col, image_path in enumerate(group.paths, start=1):
            image = _downscaled(image_path)
            image.height = round(image.height * _IMAGE_WIDTH / image.width)
            image.width = _IMAGE_WIDTH
            tallest = max(tallest, image.height)
            ws.add_image(image, f"{get_column_letter(col)}{row}")
        row += math.ceil(tallest / _ROW_HEIGHT) + 1


def _downscaled(path: str) -> Image:
    """Load an image as a reduced JPEG so the workbook stays small."""
    with PILImage.open(path) as src:
        src.thumbnail((_IMAGE_STORED_WIDTH, src.height))
        flat = PILImage.new("RGB", src.size, "white")
        flat.paste(src, mask=src.getchannel("A") if "A" in src.mode else None)
    buffer = io.BytesIO()
    flat.save(buffer, format="jpeg", quality=85)
    # openpyxl reads the bytes back from the image's file at save time
    return Image(PILImage.open(buffer))


def write_records(
    records: Sequence[Record],
    fmt: str,
    path: str | Path = "-",
) -> None:
    """Write records as json, csv or tsv to *path* (``"-"`` for stdout)."""
    if fmt not in {"json", "csv", "tsv"}:
        msg = f"Unrecognised format {fmt!r}"
        raise ValueError(msg)

    if path == "-":
        _write_records(records, fmt, sys.stdout)
        return
    with Path(path).open("w", newline="") as fh:
        _write_records(records, fmt, fh)


def _write_records(records: Sequence[Record], fmt: str, fh: Any) -> None:
    if fmt == "json":
        json.dump(
            [{k: _cell_value(v) for k, v in r.items()} for r in records],
            fh,
            indent=2,
        )
        fh.write("\n")
        return
    if not records:
        return
    writer = csv.DictWriter(
        fh,
        fieldnames=list(records[0].keys()),
        delimiter="," if fmt == "csv" else "\t",
    )
    writer.writeheader()
    for record in records:
        writer.writerow({k: _cell_value(v) for k, v in record.items()})


def _cell_value(value: Any) -> Any:
    """Convert a value to something Excel/CSV/JSON can store."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    try:
        return value.item()  # numpy scalar
    except (AttributeError, ValueError):
        return str(value)
