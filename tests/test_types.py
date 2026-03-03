"""Tests for hazenlib types and data structures."""

# ruff: noqa: PT009 PT027

from __future__ import annotations

# Type Checking
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

# Python imports
import json
import unittest
from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock, patch

# Module imports
import numpy as np

# Local imports
from hazenlib.types import Measurement, Metadata, Result, TaskMetadata


class TestJsonSerializableMixin(unittest.TestCase):
    """Tests for JSON serialization base class."""

    def test_to_dict_converts_nested_objects(self) -> None:
        """Verify to_dict recursively converts dataclass instances."""
        result = Result(task="TestTask", desc="test", files=["file.dcm"])
        result.add_measurement(
            Measurement(name="SNR", value=42.0, visibility="final"),
        )

        d = result.to_dict()

        self.assertIsInstance(d, dict)
        self.assertEqual(d["task"], "TestTask")
        self.assertIsInstance(d["measurements"], list)
        self.assertEqual(d["measurements"][0]["name"], "SNR")

    def test_to_dict_handles_numpy_arrays(self) -> None:
        """Verify numpy arrays are converted to lists in to_dict."""
        arr = np.array([1.0, 2.0, 3.0])
        result = Result(task="Test", desc="test", files=["f.dcm"])
        result.add_measurement(
            Measurement(name="SNR", value=arr, visibility="final"),
        )

        d = result.to_dict()

        self.assertIsInstance(d["measurements"][0]["value"], np.ndarray)
        self.assertTrue(
            np.all(d["measurements"][0]["value"] == [1.0, 2.0, 3.0]),
        )

    def test_to_json_returns_valid_json(self) -> None:
        """Verify to_json produces parseable JSON string."""
        result = Result(task="JSONTest", desc="json test", files=[])
        result.add_measurement(
            Measurement(name="SNR", value=1.0, visibility="final"),
        )

        json_str = result.to_json()

        # Should be valid JSON
        parsed = json.loads(json_str)
        self.assertEqual(parsed["task"], "JSONTest")
        self.assertEqual(parsed["measurements"][0]["value"], 1.0)

    def test_to_dict_handles_empty_collections(self) -> None:
        """Verify empty measurements and metadata are handled gracefully."""
        result = Result(task="Empty", desc="empty", files=[])
        # No measurements added

        d = result.to_dict()

        self.assertEqual(d["measurements"], [])
        self.assertEqual(d["report_images"], [])


class TestMeasurement(unittest.TestCase):
    """Tests for the Measurement dataclass."""

    def test_measurement_creation(self) -> None:
        """Verify Measurement can be created with all fields."""
        m = Measurement(
            name="SNR",
            value=42.5,
            type="measured",
            subtype="ratio",
            description="Signal to noise ratio",
            unit="dB",
            visibility="final",
        )

        self.assertEqual(m.name, "SNR")
        self.assertEqual(m.value, 42.5)
        self.assertEqual(m.type, "measured")
        self.assertEqual(m.visibility, "final")

    def test_measurement_defaults(self) -> None:
        """Verify Measurement has sensible defaults for optional fields."""
        m = Measurement(name="SNR", value=1.0)

        self.assertEqual(m.type, "measured")
        self.assertEqual(m.subtype, "")
        self.assertEqual(m.description, "")
        self.assertEqual(m.unit, "")
        self.assertEqual(m.visibility, "final")  # Assuming default is final

    def test_measurement_is_frozen(self) -> None:
        """Verify Measurement is immutable (frozen dataclass)."""
        m = Measurement(name="SNR", value=1.0)

        with self.assertRaises(FrozenInstanceError):
            m.name = "NewName"  # type: ignore[misc]

        with self.assertRaises(FrozenInstanceError):
            m.value = 2.0  # type: ignore[misc]

    def test_measurement_accepts_numpy_values(self) -> None:
        """Verify Measurement can store numpy scalar types."""
        np_value = np.float64(42.5)
        m = Measurement(
            name="SNR",
            value=np_value,
            visibility="final",
        )

        self.assertIsInstance(m.value, np.float64)
        self.assertEqual(float(m.value), 42.5)


class TestResult(unittest.TestCase):
    """Tests for the Result dataclass and its methods."""

    def test_result_initialization(self) -> None:
        """Verify Result initializes with correct default values."""
        result = Result(
            task="TestTask",
            desc="test desc",
            files=["a.dcm", "b.dcm"],
        )

        self.assertEqual(result.task, "TestTask")
        self.assertEqual(result.desc, "test desc")
        self.assertEqual(result.files, ["a.dcm", "b.dcm"])
        self.assertEqual(result.measurements, ())
        self.assertEqual(set(result.report_images), set())
        self.assertIsInstance(result.metadata, Metadata)

    def test_add_measurement(self) -> None:
        """Verify add_measurement appends to internal list."""
        result = Result(task="Test", desc="test", files=["f.dcm"])
        m1 = Measurement(name="SNR", value=1.0, visibility="final")
        m2 = Measurement(name="Ghosting", value=2.0, visibility="intermediate")

        result.add_measurement(m1)
        result.add_measurement(m2)

        self.assertEqual(len(result.measurements), 2)
        self.assertEqual(result.measurements[0].name, "SNR")
        self.assertEqual(result.measurements[1].name, "Ghosting")

    def test_add_report_image(self) -> None:
        """Verify add_report_image adds paths to internal list."""
        result = Result(task="Test", desc="test", files=["input.dcm"])

        result.add_report_image("output1.png")
        result.add_report_image("output2.png")

        self.assertEqual(
            set(result.report_images),
            {"output1.png", "output2.png"},
        )

    def test_get_measurement_filtering(self) -> None:
        """Verify get_measurement filters correctly by attributes."""
        result = Result(task="Test", desc="test", files=["f.dcm"])
        result.add_measurement(
            Measurement(name="SNR", value=10.0, visibility="final"),
        )
        result.add_measurement(
            Measurement(
                name="SNR",
                type="raw",
                value=20.0,
                visibility="intermediate",
            ),
        )
        result.add_measurement(
            Measurement(name="Ghosting", value=5.0, visibility="final"),
        )

        # Filter by name
        snr_measurements = result.get_measurement(name="SNR")
        self.assertEqual(len(snr_measurements), 2)

        # Filter by combined criteria
        specific = result.get_measurement(
            name="SNR",
            measurement_type="raw",
        )
        self.assertEqual(len(specific), 1)
        self.assertEqual(specific[0].value, 20.0)

    def test_get_measurement_empty_result(self) -> None:
        """Verify get_measurement returns empty list when no matches."""
        result = Result(task="SNR", desc="test", files=[])

        matches = result.get_measurement(name="nonexistent")
        self.assertEqual(matches, [])

    def test_filtered_creates_copy(self) -> None:
        """Verify filtered() returns new Result without modifying original."""
        result = Result(task="Test", desc="test", files=["f.dcm"])
        result.add_measurement(
            Measurement(name="SNR", value=1.0, visibility="final"),
        )
        result.add_measurement(
            Measurement(
                name="Ghosting",
                value=2.0,
                visibility="intermediate",
            ),
        )

        filtered = result.filtered(level="final")

        # Original should be unchanged
        self.assertEqual(len(result.measurements), 2)
        # Filtered should have only one
        self.assertEqual(len(filtered.measurements), 1)
        self.assertEqual(filtered.measurements[0].name, "SNR")
        # Metadata copied
        self.assertEqual(filtered.task, "Test")

    def test_filtered_all_returns_self(self) -> None:
        """Verify filtered(level='all') returns self for efficiency."""
        result = Result(task="Test", desc="test", files=["f.dcm"])
        result.add_measurement(
            Measurement(name="SNR", value=1.0, visibility="final"),
        )

        filtered = result.filtered(level="all")

        # Should be the same object (identity check)
        self.assertIs(filtered, result)

    def test_filtered_invalid_level_raises_error(self) -> None:
        """Verify filtered raises ValueError for invalid level."""
        result = Result(task="Test", desc="test", files=["f.dcm"])

        with self.assertRaises(ValueError) as context:
            result.filtered(level="invalid_level")

        self.assertIn("Invalid measurement visibility", str(context.exception))


class TestMetadata(unittest.TestCase):
    """Tests for the Metadata dataclass."""

    def test_metadata_defaults(self) -> None:
        """Verify Metadata initializes with None/empty defaults."""
        m = Metadata()

        self.assertIsNone(m.institution_name)
        self.assertIsNone(m.manufacturer)
        self.assertIsNone(m.model)
        self.assertIsNone(m.date)
        self.assertIsNone(m.series_id)
        self.assertIsNone(m.study_id)
        self.assertIsNotNone(m.version)

    def test_metadata_with_values(self) -> None:
        """Verify Metadata accepts construction with values."""
        m = Metadata(
            institution_name="Test Hospital",
            manufacturer="GE",
            model="Artist",
        )

        self.assertEqual(m.institution_name, "Test Hospital")
        self.assertEqual(m.manufacturer, "GE")
        self.assertEqual(m.model, "Artist")


class TestTaskMetadata(unittest.TestCase):
    """Tests for TaskMetadata configuration class."""

    def test_task_metadata_defaults(self) -> None:
        """Verify TaskMetadata has correct defaults."""
        tm = TaskMetadata(module_name="Test", class_name="TestClass")

        self.assertEqual(tm.single_image, False)
        self.assertIsNone(tm.phantom)
        self.assertIsNone(tm.requires_args)


class TestMetadataDicomExtraction(unittest.TestCase):
    """Tests for Metadata __post_init__ DICOM extraction."""

    def setUp(self) -> None:
        """Set up realistic GE scanner DICOM attributes."""
        self.ge_dicom_attrs = {
            "InstitutionName": "Anon",  # From your DICOM dump
            "Manufacturer": "GE MEDICAL SYSTEMS",
            "ManufacturerModelName": "Signa HDxt",
            "StudyDate": "20200903",
            "SeriesInstanceUID": (
                "1.2.840.113619.2.322.2807.4256007.22970.1599113147.28"
            ),
            "StudyInstanceUID": (
                "1.2.840.113619.6.322.29258601288877239892320825850091988423"
            ),
        }

    def _create_mock_dicom(self, **overrides: dict[str, Any]) -> None:
        """Create a mock pydicom Dataset with GE-like attributes."""
        mock = MagicMock()
        attrs = {**self.ge_dicom_attrs, **overrides}
        for key, value in attrs.items():
            setattr(mock, key, value)
        # Ensure hasattr works correctly
        mock.__contains__ = lambda _, key: key in attrs
        return mock

    @patch("hazenlib.types.pydicom.dcmread")
    def test_extracts_ge_scanner_metadata(self, mock_dcmread: Callable) -> None:
        """Verify all metadata fields extracted from GE DICOM."""
        mock_dcmread.return_value = self._create_mock_dicom()

        m = Metadata(files=["tests/data/acr/GE/0.dcm"])

        self.assertEqual(m.institution_name, "Anon")
        self.assertEqual(m.manufacturer, "GE MEDICAL SYSTEMS")
        self.assertEqual(m.model, "Signa HDxt")
        self.assertEqual(m.date, "20200903")
        self.assertEqual(
            m.series_id,
            "1.2.840.113619.2.322.2807.4256007.22970.1599113147.28",
        )
        self.assertEqual(
            m.study_id,
            "1.2.840.113619.6.322.29258601288877239892320825850091988423",
        )

    @patch("hazenlib.types.pydicom.dcmread")
    def test_skips_extraction_if_fields_populated(
        self, mock_dcmread: Callable,
    ) -> None:
        """Verify existing values are preserved, not overwritten."""
        mock_dcmread.return_value = self._create_mock_dicom(
            InstitutionName="Different Hospital",
        )

        m = Metadata(
            files=["test.dcm"],
            institution_name="Keep This Value",
            manufacturer="Siemens",  # Also set differently
        )

        # Should keep provided values, not extract from DICOM
        self.assertEqual(m.institution_name, "Keep This Value")
        self.assertEqual(m.manufacturer, "Siemens")
        # But unset fields should still extract
        self.assertEqual(m.model, "Signa HDxt")

    @patch("hazenlib.types.pydicom.dcmread")
    def test_handles_missing_optional_attributes(
        self, mock_dcmread: Callable,
    ) -> None:
        """Verify extraction works when optional DICOM tags are missing."""
        # Create DICOM without InstitutionName (anonymized/removed)
        mock = self._create_mock_dicom()
        delattr(mock, "InstitutionName")

        mock_dcmread.return_value = mock

        m = Metadata(files=["anon.dcm"])

        # Should still extract other fields
        self.assertEqual(m.manufacturer, "GE MEDICAL SYSTEMS")
        self.assertIsNone(m.institution_name)  # Not present in DICOM

    @patch("hazenlib.types.pydicom.dcmread")
    def test_warns_on_multiple_different_values(
        self, mock_dcmread: Callable,
    ) -> None:
        """Verify warning logged when files have different institutions."""
        mock_dcmread.side_effect = [
            self._create_mock_dicom(InstitutionName="Hospital A"),
            self._create_mock_dicom(InstitutionName="Hospital B"),
            self._create_mock_dicom(InstitutionName="Hospital C"),
        ]

        with self.assertLogs("hazenlib.types", level="WARNING") as log_context:
            m = Metadata(files=["1.dcm", "2.dcm", "3.dcm"])

        # Should warn about multiple values
        self.assertTrue(
            any(
                "institution_name" in msg and "Multiple" in msg
                for msg in log_context.output
            ),
        )
        # Should still pick one value (arbitrary from set)
        self.assertIn(
            m.institution_name, ["Hospital A", "Hospital B", "Hospital C"],
        )

    @patch("hazenlib.types.pydicom.dcmread")
    def test_graceful_on_dicom_read_error(self, mock_dcmread: Callable) -> None:
        """Verify Metadata creation succeeds even if files can"t be read."""
        mock_dcmread.side_effect = Exception("Permission denied")

        # Should not raise, just log warning
        with self.assertLogs("hazenlib.types", level="WARNING"):
            m = Metadata(files=["corrupt.dcm"])

        # Fields remain None (or version defaults)
        self.assertIsNone(m.institution_name)
        self.assertIsNone(m.manufacturer)

    @patch("hazenlib.types.pydicom.dcmread")
    def test_no_extraction_without_files(self, mock_dcmread: Callable) -> None:
        """Verify dcmread is never called when files is None."""
        m = Metadata()  # No files

        mock_dcmread.assert_not_called()
        self.assertIsNone(m.institution_name)


if __name__ == "__main__":
    unittest.main()
