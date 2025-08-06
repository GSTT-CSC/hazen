"""Output formatters from dict-like objects."""

# ruff: noqa: ANN401

from __future__ import annotations

# Python imports
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence, TextIO, TypeAlias, TypedDict

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Public type alias for the format argument
# ----------------------------------------------------------------------

Format: TypeAlias = Literal["json", "csv", "tsv"]

# ----------------------------------------------------------------------
# Row definition for the CSV/TSV output
# ----------------------------------------------------------------------

class CsvRow(TypedDict, total=False):
    """CSV Row datatype."""

    task: str
    file: str
    measurement_type: str       # e.g. "snr by smoothing"
    subfile: str | None         # for the inner dict keys when present
    measured: float | None
    normalised: float | None

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
    path = None if path == "-" else path
    try:
        with Path(path).open("w", newline="") as fp:
            _format_results(data, fmt, fp)
    except TypeError:
        logger.debug("Couldn't write to %s - falling back to STDOUT.", path)
        _format_results(data, fmt, sys.stdout)


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
            json.dump(data, out_fh, indent=2)
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


def _build_rows(data: Mapping[str, Any]) -> list[CsvRow]:
    """Convert nested data to flat rows.

    Convert the nested measurement dictionary
    into a list of flat rows suitable for CSV.

    The algorithm walks the structure
    and emits one row for each *leaf* measurement
    (i.e. each measured/normalised pair).
    The hierarchy is captured in the columns
    ``measurement_type`` and ``subfile``.
    """
    rows: list[CsvRow] = []

    task = data.get("task", "")
    files: Sequence[str] = data.get("file", [])
    measurement = data.get("measurement", {})

    # The measurement dict can contain multiple top-level keys
    # e.g. ("snr by smoothing", "snr by subtraction", …)
    for meas_type, inner in measurement.items():
        # ``inner`` may be a dict of per‑file entries (as in the example)
        # or a flat dict with the scalar values directly.
        if isinstance(inner, Mapping):
            # Detect the two‑level case: inner keys are file names
            for sub_key, leaf in inner.items():
                # ``leaf`` may itself be a mapping with "measured"/"normalised"
                if isinstance(leaf, Mapping):
                    measured = _to_python_scalar(leaf.get("measured"))
                    normalised = _to_python_scalar(leaf.get("normalised"))
                else:
                    # Unexpected shape - fall back to a single value
                    measured = _to_python_scalar(leaf)
                    normalised = None

                # If we have an explicit list of files, match them;
                # otherwise just use the sub_key as the file identifier.
                file_name = sub_key if sub_key in files else sub_key

                rows.append(
                    CsvRow(
                        task=task,
                        file=file_name,
                        measurement_type=meas_type,
                        subfile=None,
                        measured=measured,
                        normalised=normalised,
                    ),
                )
        else:
            # ``inner`` is a scalar
            # explicitly raise an error as this shouldn't happen
            msg = "Measurement should be a dict"
            logger.critical("No measurement type found in %s. %s.", data, msg)
            raise TypeError(msg)
    return rows
