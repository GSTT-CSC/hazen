"""Writers for protocol results (CSV/TSV and JSON)."""

from __future__ import annotations

# Type Checking
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

# Python imports
import csv
import json
import sys
from pathlib import Path
from typing import Any

Record = dict[str, Any]


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
    """Convert a value to something CSV/JSON can store."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    try:
        return value.item()  # numpy scalar
    except (AttributeError, ValueError):
        return str(value)
