"""Tests for execution instrumentation."""

# ruff: noqa: PT009 PT027

import unittest
from unittest.mock import Mock, patch

from hazenlib.execution import timed_execution
from hazenlib.tasks.acr_sagittal_geometric_accuracy import (
    ACRSagittalGeometricAccuracy,
)
from hazenlib.types import Result
from hazenlib.utils import get_dicom_files

from tests import TEST_DATA_DIR


class TestTimedExecution(unittest.TestCase):
    """Unit tests for the timed_execution wrapper."""

    def test_adds_timing_measurement(self) -> None:
        """Verify timing metadata is added to successful task results."""
        # Arrange
        mock_result = Result(task="TestTask", desc="test")
        mock_task = Mock(return_value=mock_result)

        # Act: Mock time advancing by 1.5 seconds
        with patch(
            "hazenlib.execution.time.perf_counter", side_effect=[0.0, 1.5]
        ):
            result = timed_execution(mock_task, "arg1", kwarg1="value1")

        # Assert
        self.assertEqual(result, mock_result)
        mock_task.assert_called_once_with("arg1", kwarg1="value1")

        # Verify measurement structure per constants.py spec
        timing = next(
            m
            for m in result.measurements
            if m.name == "ExecutionMetadata"
            and m.description == "analysis_duration"
        )

        self.assertEqual(timing.value, 1.5)
        self.assertEqual(timing.unit, "s")
        self.assertEqual(timing.type, "measured")

    def test_preserves_task_arguments(self) -> None:
        """Verify ParamSpec properly forwards args/kwargs."""
        mock_result = Result(task="TestTask", desc="test")
        mock_task = Mock(return_value=mock_result)

        with patch("hazenlib.execution.time.perf_counter", return_value=0.0):
            timed_execution(mock_task, 1, 2, foo="bar", baz=123)

        mock_task.assert_called_once_with(1, 2, foo="bar", baz=123)

    def test_exception_propagation(self) -> None:
        """Verify exceptions propagate and no timing added on failure."""
        mock_task = Mock(side_effect=RuntimeError("Analysis failed"))

        with (
            self.assertRaises(RuntimeError),
            patch("hazenlib.execution.time.perf_counter", return_value=0.0),
        ):
            timed_execution(mock_task)

        mock_task.assert_called_once()


class TestExecutionIntegration(unittest.TestCase):
    """Verify timed_execution works with real task infrastructure."""

    def test_real_task_timing_structure(self) -> None:
        """Smoke test: time a real task execution."""
        test_data = TEST_DATA_DIR / "acr" / "SiemensSolaFitLocalizer"
        if not test_data.exists():
            self.skipTest("Test data unavailable")

        task = ACRSagittalGeometricAccuracy(
            input_data=get_dicom_files(test_data),
        )
        result = timed_execution(task.run)

        # Verify timing present and reasonable (0 < t < 30s)
        exec_meta = [
            m for m in result.measurements if m.name == "ExecutionMetadata"
        ]
        self.assertEqual(len(exec_meta), 1)
        self.assertGreater(exec_meta[0].value, 0.0)
        self.assertLess(exec_meta[0].value, 30.0)


if __name__ == "__main__":
    unittest.main()
