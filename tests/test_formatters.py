"""Tests for output formatters."""

# ruff: noqa: D401 PT009 PT027

from __future__ import annotations

# Python imports
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

# Module imports
import numpy as np

# Local imports
from hazenlib.formatters import (
    _build_rows,
    _format_results,
    write_result,
)
from hazenlib.types import Measurement, Metadata, Result


class TestWriteResult(unittest.TestCase):
    """Unit tests for the write_result entry point."""

    def _create_result(self, visibility: str = "final") -> Result:
        """Helper to create a Result with measurements of specific visibility."""
        result = Result(task="SNR", desc="SNR test", files=["test.dcm"])
        result.metadata = Metadata(institution_name="Test Hospital")
        result.add_measurement(
            Measurement(
                name="SNR",
                value=42.5,
                type="measured",
                unit="dB",
                visibility=visibility,
            ),
        )
        return result

    def test_writes_json_to_stdout_by_default(self) -> None:
        """Verify JSON output is written to stdout when path is '-'."""
        result = self._create_result()

        with patch("sys.stdout", new=io.StringIO()) as mock_stdout:
            write_result(result, "json", path="-", level="all")
            output = mock_stdout.getvalue()

        # Verify structure
        data = json.loads(output)
        self.assertEqual(data["task"], "SNR")
        self.assertEqual(len(data["measurements"]), 1)
        self.assertEqual(data["measurements"][0]["value"], 42.5)

    def test_writes_csv_to_file_with_header(self) -> None:
        """Verify CSV is written to file with comma delimiter."""
        result = self._create_result()

        with tempfile.NamedTemporaryFile(
            mode="w",
            delete=False,
            suffix=".csv",
        ) as tmp:
            tmp_path = tmp.name

        try:
            write_result(result, "csv", path=tmp_path, level="all")

            with Path(tmp_path).open("r") as f:
                content = f.read()
                lines = content.strip().split("\n")

                # Header + 1 data row
                self.assertEqual(len(lines), 2)
                self.assertIn("name", lines[0])  # header
                self.assertIn("42.5", lines[1])  # data
                self.assertIn(",", content)  # comma delimited
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_level_filtering_excludes_intermediate(self) -> None:
        """Verify level='final' filters out intermediate measurements."""
        result = Result(task="SNR", desc="test", files=["1.dcm"])
        result.add_measurement(
            Measurement(
                name="SNR",
                subtype="final",
                value=1.0,
                visibility="final",
            ),
        )
        result.add_measurement(
            Measurement(
                name="SNR",
                subtype="intermediate",
                value=2.0,
                visibility="intermediate",
            ),
        )

        with patch("sys.stdout", new=io.StringIO()) as mock_stdout:
            write_result(result, "json", path="-", level="final")
            output = mock_stdout.getvalue()

        data = json.loads(output)
        self.assertEqual(len(data["measurements"]), 1)
        self.assertEqual(data["measurements"][0]["name"], "SNR")

    def test_invalid_level_defaults_to_all_with_warning(self) -> None:
        """Verify unknown level logs warning and defaults to 'all'."""
        result = self._create_result()

        with patch("hazenlib.formatters.logger") as mock_logger:
            with patch(
                "sys.stdout",
                new=io.StringIO(),
            ) as mock_stdout:  # noqa: F841
                write_result(result, "json", path="-", level="unknown_level")
                # Should not raise, should log warning

            mock_logger.warning.assert_called_once()
            self.assertIn(
                "Unknown measurement filter level",
                str(mock_logger.warning.call_args),
            )

    def test_invalid_format_raises_valueerror(self) -> None:
        """Verify ValueError is raised for unsupported format strings."""
        result = self._create_result()

        with self.assertRaises(ValueError) as context:
            write_result(result, "xml", path="-")  # type: ignore[arg-type]

        self.assertIn("Unrecognised format", str(context.exception))


class TestFormatResultsDispatcher(unittest.TestCase):
    """Tests for _format_results dispatching and formatting logic."""

    def test_json_format_calls_to_json(self) -> None:
        """Verify json format uses Result.to_json() method."""
        mock_result = Mock(spec=Result)
        mock_result.to_json.return_value = '{"task": "test"}'

        output = io.StringIO()
        _format_results(mock_result, "json", output)

        self.assertEqual(output.getvalue().strip(), '{"task": "test"}')
        mock_result.to_json.assert_called_once()

    def test_csv_uses_comma_delimiter(self) -> None:
        """Verify csv format uses comma as delimiter."""
        result = Result(task="Test", desc="test", files="file.dcm")
        result.add_measurement(
            Measurement(name="SNR", value=1.0, visibility="final"),
        )

        output = io.StringIO()
        _format_results(result, "csv", output, write_header=False)

        content = output.getvalue()
        lines = content.strip().split("\n")
        # Check first line contains commas, not tabs
        self.assertIn(",", lines[0])
        self.assertNotIn("\t", content)

    def test_tsv_uses_tab_delimiter(self) -> None:
        """Verify tsv format uses tab as delimiter."""
        result = Result(task="SNR", desc="test", files="file.dcm")
        result.add_measurement(
            Measurement(name="SNR", value=1.0, visibility="final"),
        )

        output = io.StringIO()
        _format_results(result, "tsv", output, write_header=False)

        content = output.getvalue()
        self.assertIn("\t", content)

    def test_csv_header_written_conditionally(self) -> None:
        """Verify write_header parameter controls header output."""
        result = Result(task="SNR", desc="test", files="file.dcm")
        result.add_measurement(
            Measurement(name="SNR", value=42.0, visibility="final"),
        )

        # With header (default)
        with_header = io.StringIO()
        _format_results(result, "csv", with_header, write_header=True)
        lines_with = with_header.getvalue().strip().split("\n")
        self.assertIn("name", lines_with[0])

        # Without header
        without_header = io.StringIO()
        _format_results(result, "csv", without_header, write_header=False)
        lines_without = without_header.getvalue().strip().split("\n")
        # Should still have content but no header row with field names
        self.assertEqual(len(lines_without) + 1, len(lines_with))


class TestBuildRows(unittest.TestCase):
    """Tests for _build_rows data transformation."""

    def test_flattens_nested_structure(self) -> None:
        """Verify nested Result/Metadata/Measurements are flattened into rows."""
        result = Result(
            task="ACRSNR",
            desc="SNR analysis",
            files=["1.dcm", "2.dcm"],
        )
        result.metadata = Metadata(
            institution_name="Test Hospital",
            manufacturer="Siemens",
            plate=1,
        )
        result.add_measurement(
            Measurement(
                name="SNR",
                value=125.5,
                type="measured",
                unit="ratio",
                visibility="final",
            ),
        )

        rows = _build_rows(result)

        self.assertEqual(len(rows), 1)
        row = rows[0]
        # Measurement fields
        self.assertEqual(row["name"], "SNR")
        self.assertEqual(row["value"], 125.5)
        self.assertEqual(row["unit"], "ratio")
        # Result fields
        self.assertEqual(row["task"], "ACRSNR")
        # Metadata fields
        self.assertEqual(row["institution_name"], "Test Hospital")
        self.assertEqual(row["manufacturer"], "Siemens")

    def test_handles_multiple_measurements(self) -> None:
        """Verify multiple measurements create multiple rows."""
        result = Result(
            task="SNR",
            desc="Geometric accuracy",
            files=["1.dcm"],
        )
        result.add_measurement(
            Measurement(
                name="SNR",
                description="x_dim",
                value=100.0,
                visibility="final",
            ),
        )
        result.add_measurement(
            Measurement(
                name="SNR",
                description="y_dim",
                value=100.1,
                visibility="final",
            ),
        )

        rows = _build_rows(result)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["description"], "x_dim")
        self.assertEqual(rows[1]["description"], "y_dim")
        # Both rows should inherit same task/metadata
        self.assertEqual(rows[0]["task"], "SNR")
        self.assertEqual(rows[1]["task"], "SNR")

    def test_returns_empty_list_for_no_measurements(self) -> None:
        """Verify empty measurements returns empty list (no rows)."""
        result = Result(task="Empty", desc="No data", files=[])
        rows = _build_rows(result)

        self.assertEqual(rows, [])

    def test_converts_numpy_scalars(self) -> None:
        """Verify numpy float64 etc. are converted to Python scalars."""
        mock_value = np.float64(42.0)

        result = Result(task="Test", desc="test", files="f.dcm")
        result.add_measurement(
            Measurement(
                name="SNR",
                value=mock_value,  # type: ignore[arg-type]
                visibility="final",
            ),
        )

        rows = _build_rows(result)
        self.assertEqual(rows[0]["value"], 42.0)
        self.assertIsInstance(rows[0]["value"], float)


if __name__ == "__main__":
    unittest.main()
