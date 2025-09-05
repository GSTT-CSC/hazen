"""Output formatters from dict-like objects."""

# ruff: noqa: ANN401

from __future__ import annotations

# Python imports
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence, TextIO, TypeAlias

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Public type alias for the format argument
# ----------------------------------------------------------------------

Format: TypeAlias = Literal["json", "csv", "tsv"]

# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def write_result(data: dict, fmt: Format, path: str | Path = "-") -> None:
    r"""Serialise data to a format and write the output.

    Serialize *data* into the requested *fmt* and write it either to
    ``stdout`` (default) or to a file.

    Parameters
    ----------
    data:
        Mapping that will be serialised.
    fmt:
        Output format. ``json`` is the default; other choices are
        ``csv``, ``tsv``.
    path:
        Path to the destination file.  ``"-"`` (the default) means
        write to ``sys.stdout``.  Any other value is interpreted as a
        filesystem path; the file is opened in text mode with
        ``newline=\"\"`` for CSV/TSV compatibility.

    Raises
    ------
    ValueError
        If *fmt* is not one of the supported literals.

    """
    if path == "-":
         _format_results(data, fmt, sys.stdout)
         return
    with Path(path).open("w", newline="") as fp:
        _format_results(data, fmt, fp)

# ----------------------------------------------------------------------
# Private API
# ----------------------------------------------------------------------

def _to_python_scalar(value: Any) -> Any:
    try:
        return value.item()
    except AttributeError:
        return value


def _format_results(
        data: Mapping[str, Any],
        fmt: Format,
        out_fh: TextIO,
) -> None:
    """Dispatcher that writes *data* to *out_fh* in the requested *fmt*.

    The function assumes *out_fh* is already opened and will **not** close it.
    """
    match fmt:
        case "json":
            print(data.to_json(), file=out_fh)
        case "csv":
            _write_csv(data, out_fh, delimiter=",")
        case "tsv":
            _write_csv(data, out_fh, delimiter="\t")
        case _:
            msg = f"Unrecognised format {fmt!r}"
            logger.critical(msg)
            raise ValueError(msg)


def _write_csv(
    data: Mapping[str, Any], out_fh: TextIO, *, delimiter: str,
) -> None:
    """Write a flat mapping as a two-row CSV/TSV file."""
    rows = _build_rows(data)
    if not rows:
        logger.warning("No rows generated from data.")
        logger.debug(str(data))
        return

    writer = csv.DictWriter(
        out_fh,
        fieldnames=rows[0].keys(),
        delimiter=delimiter,
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(row)


def _build_rows(data: Mapping[str, Any]) -> list[dict]:
    """Convert nested data to flat rows.

    Convert the nested result dictionary
    into a list of flat rows suitable for CSV.
    """
    d = data.to_dict()
    measurements = d.pop("measurements", {})
    metadata = d.pop("metadata", {})
    return [{**m, **d, **metadata} for m in measurements]
