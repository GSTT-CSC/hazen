"""Tests for the orchestration module."""

# ruff: noqa: PT009 PT027

# Python imports
import unittest
from collections.abc import Callable
from unittest.mock import Mock, patch

from hazenlib.exceptions import (
    UnknownAcquisitionTypeError,
    UnknownTaskNameError,
)
from hazenlib.HazenTask import HazenTask
from hazenlib.orchestration import (
    AcquisitionType,
    ACRLargePhantomProtocol,
    Protocol,
    ProtocolResult,
    ProtocolStep,
    init_task,
)
from hazenlib.types import Result

from tests import TEST_DATA_DIR, TEST_REPORT_DIR


class TestInitTask(unittest.TestCase):
    """Unit tests for task initialization."""

    def test_successful_initialization(self) -> None:
        """Verify successful task creation from registry."""
        # Arrange
        mock_task_class = Mock(spec=HazenTask)
        mock_module = Mock()
        mock_module.SNR = mock_task_class

        with patch(
            "hazenlib.orchestration.importlib.import_module",
            return_value=mock_module,
        ):
            # Act
            args = (
                "snr",
                ["file1.dcm", "file2.dcm"],
            )
            kwargs = {
                "report": True,
                "report_dir": TEST_REPORT_DIR,
            }
            result = init_task(
                *args,
                **kwargs,
            )

            # Assert
            self.assertEqual(result, mock_task_class.return_value)
            mock_task_class.assert_called_once_with(
                input_data=args[1],
                **kwargs,
            )

    def test_unknown_task_raises_value_error(self) -> None:
        """Verify ValueError raised for unknown task names."""
        with self.assertRaises(ValueError) as context:
            init_task(
                "unknown_task",
                [],
                report=False,
                report_dir=TEST_REPORT_DIR,
            )

        self.assertIn("Unknown task", str(context.exception))

    def test_missing_class_raises_import_error(self) -> None:
        """Verify ImportError when module lacks expected class."""
        # Mock module without the expected class attribute
        mock_module = Mock(spec=["__name__", "NotSNR"])

        with (
            patch(
                "hazenlib.orchestration.importlib.import_module",
                return_value=mock_module,
            ),
            self.assertRaises(ImportError) as context,
        ):
            init_task("snr", [], report=False, report_dir=TEST_REPORT_DIR)

        self.assertIn("has no class", str(context.exception))


class TestAcquisitionType(unittest.TestCase):
    """Unit tests for AcquisitionType enum."""

    def test_from_string_t1(self) -> None:
        """Verify T1 string parsing."""
        result = AcquisitionType.from_string("t1")
        self.assertEqual(result, AcquisitionType.ACR_T1)
        # Case insensitive
        result = AcquisitionType.from_string("T1")
        self.assertEqual(result, AcquisitionType.ACR_T1)

    def test_from_string_t2(self) -> None:
        """Verify T2 string parsing."""
        result = AcquisitionType.from_string("t2")
        self.assertEqual(result, AcquisitionType.ACR_T2)

    def test_from_string_sagittal_variants(self) -> None:
        """Verify sagittal localizer parsing with spelling variants."""
        # Full string
        result = AcquisitionType.from_string("sagittal localiser")
        self.assertEqual(result, AcquisitionType.ACR_SL)

        # American spelling
        result = AcquisitionType.from_string("sagittal localizer")
        self.assertEqual(result, AcquisitionType.ACR_SL)

        # Abbreviations
        result = AcquisitionType.from_string("sagittal")
        self.assertEqual(result, AcquisitionType.ACR_SL)

        result = AcquisitionType.from_string("localizer")
        self.assertEqual(result, AcquisitionType.ACR_SL)

    def test_from_string_invalid_raises_error(self) -> None:
        """Verify UnknownAcquisitionTypeError for invalid strings."""
        with self.assertRaises(UnknownAcquisitionTypeError):
            AcquisitionType.from_string("unknown_type")

        with self.assertRaises(UnknownAcquisitionTypeError):
            AcquisitionType.from_string("t3")


class TestProtocolStep(unittest.TestCase):
    """Unit tests for ProtocolStep dataclass."""

    def test_valid_initialization(self) -> None:
        """Verify ProtocolStep can be created with valid task name."""
        step = ProtocolStep("snr", AcquisitionType.ACR_T1)
        self.assertEqual(step.task_name, "snr")
        self.assertEqual(step.acquisition_type, AcquisitionType.ACR_T1)

    def test_invalid_task_name_raises_error(self) -> None:
        """Verify UnknownTaskNameError raised for unregistered tasks."""
        with self.assertRaises(UnknownTaskNameError):
            ProtocolStep("nonexistent_task", AcquisitionType.ACR_T1)


class TestProtocol(unittest.TestCase):
    """Unit tests for Protocol dataclass."""

    def test_basic_initialization(self) -> None:
        """Verify Protocol can be initialized with name and steps."""
        steps = (
            ProtocolStep("snr", AcquisitionType.ACR_T1),
            ProtocolStep("ghosting", AcquisitionType.ACR_T1),
        )
        protocol = Protocol(name="Test Protocol", steps=steps)
        self.assertEqual(protocol.name, "Test Protocol")
        self.assertEqual(len(protocol.steps), 2)

    def test_empty_steps_default(self) -> None:
        """Verify Protocol defaults to empty steps tuple."""
        protocol = Protocol(name="Empty Protocol")
        self.assertEqual(protocol.steps, ())

    def test_from_config_not_implemented(self) -> None:
        """Verify from_config raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            Protocol.from_config(TEST_DATA_DIR / "config.json")


class TestProtocolResult(unittest.TestCase):
    """Unit tests for ProtocolResult class."""

    def test_initialization(self) -> None:
        """Verify ProtocolResult initialization."""
        result = ProtocolResult(task="TestProtocol", desc="test description")
        self.assertEqual(result.task, "TestProtocol")
        self.assertEqual(result.desc, "test description")
        self.assertEqual(len(result.results), 1)  # Initial result

    def test_add_result(self) -> None:
        """Verify results can be added to collection."""
        protocol_result = ProtocolResult(task="Protocol", desc="test")
        mock_result = Result(task="SubTask", desc="subtask result")

        protocol_result.add_result(mock_result)

        self.assertEqual(len(protocol_result.results), 2)
        self.assertEqual(protocol_result.results[-1], mock_result)

    def test_results_immutable(self) -> None:
        """Verify results property returns immutable tuple."""
        protocol_result = ProtocolResult(task="Protocol", desc="test")
        protocol_result.add_result(Result(task="Task1", desc="desc1"))

        results = protocol_result.results
        self.assertIsInstance(results, tuple)

        # Verify immutability
        with self.assertRaises(TypeError):
            results[0] = Result(task="Task2", desc="desc2")

    def test_add_multiple_results(self) -> None:
        """Verify multiple results can be added and retrieved."""
        protocol_result = ProtocolResult(task="Protocol", desc="test")
        num_results = 3
        results = [
            Result(task=f"Task{i}", desc=f"result{i}")
            for i in range(num_results)
        ]

        for r in results:
            protocol_result.add_result(r)

        # Protocol results always start off with an initial result
        # for the result of the protocol.
        self.assertEqual(len(protocol_result.results), num_results + 1)
        for i, r in enumerate(protocol_result.results[1:]):
            self.assertEqual(r.task, f"Task{i}")


class TestACRLargePhantomProtocol(unittest.TestCase):
    """Integration tests for ACRLargePhantomProtocol.

    Attributes:
        PROTOCOL_STEPS: Number of steps for the protocol.
        MIN_DIRS : Minimum number of directories for the protocol.

    """

    PROTOCOL_STEPS: int = 15
    MIN_DIRS: int = 3

    def setUp(self) -> None:
        """Set up the test cases."""
        self.dirs = [
            TEST_DATA_DIR / "acr" / seq
            for seq in (
                "Siemens_Aera_1.5T_T1",
                "Siemens_Aera_1.5T_T2",
                "SiemensSolaFitLocalizer",
            )
        ]

    @patch("hazenlib.orchestration.ACRObject")
    def test_initialization_correct_dir_count(
        self,
        mock_acr_obj: Callable,
    ) -> None:
        """Verify protocol initializes with correct number of directories."""
        # Arrange
        mock_acr_instances = []
        for acq_type in ["T1", "T2", "sagittal localizer"]:
            mock_inst = Mock()
            mock_inst.acquisition_type.return_value = acq_type
            mock_acr_instances.append(mock_inst)

        mock_acr_obj.side_effect = mock_acr_instances

        # Act
        protocol = ACRLargePhantomProtocol(dirs=self.dirs)

        # Assert
        self.assertEqual(protocol.name, "ACR Large Phantom")
        self.assertEqual(len(protocol.steps), self.PROTOCOL_STEPS)
        self.assertEqual(len(protocol.file_groups), self.MIN_DIRS)

    def test_initialization_wrong_dir_count_raises_error(self) -> None:
        """Verify ValueError raised when directory count mismatch."""
        # Arrange - only provide 2 dirs when 3 unique types required
        # that is, missing localizer.
        dirs = self.dirs[:-1]

        with self.assertRaises(ValueError) as context:
            ACRLargePhantomProtocol(dirs=dirs)

        self.assertIn(
            "Incorrect number of directories",
            str(context.exception),
        )

    def test_nonunique_sequences_raise_error(self) -> None:
        """Verify ValueError raised when duplicated sequences are passed."""
        dirs = [*self.dirs[:2], self.dirs[0]]

        with self.assertRaises(ValueError) as context:
            ACRLargePhantomProtocol(dirs=dirs)

        self.assertIn(
            "Missing sequences",
            str(context.exception),
        )

    @patch("hazenlib.orchestration.ACRObject")
    @patch("hazenlib.orchestration.init_task")
    def test_run_executes_all_steps(
        self,
        mock_init_task: Callable,
        mock_acr_obj: Callable,
    ) -> None:
        """Verify run() executes all protocol steps and returns results."""
        # Arrange
        mock_acr_instances = []
        for acq_type in ["T1", "T2", "sagittal localizer"]:
            mock_inst = Mock()
            mock_inst.acquisition_type.return_value = acq_type
            mock_acr_instances.append(mock_inst)

        mock_acr_obj.side_effect = mock_acr_instances

        mock_task = Mock()
        mock_task.run.return_value = Result(task="MockTask", desc="done")
        mock_init_task.return_value = mock_task

        protocol = ACRLargePhantomProtocol(dirs=self.dirs)

        # Act
        # Use debug=True to disable using worker processes
        # which aren't registered in call_count
        result = protocol.run(debug=True)

        # Assert
        self.assertIsInstance(result, ProtocolResult)
        # Should initialize task for each step
        self.assertEqual(mock_init_task.call_count, self.PROTOCOL_STEPS)
        self.assertEqual(len(result.results), self.PROTOCOL_STEPS + 1)

    def test_steps_contain_expected_tasks(self) -> None:
        """Verify default steps contain expected task names."""
        with (
            patch(
                "hazenlib.orchestration.get_dicom_files",
                return_value=["mock_dicom.dcm"],
            ),
            patch(
                "hazenlib.orchestration.ACRObject",
            ) as mock_acr,
            patch(
                "hazenlib.orchestration.dcmread",
            ) as mock_dcmread,
        ):
            mock_inst = Mock()
            mock_inst.acquisition_type.side_effect = [
                "T1",
                "T2",
                "sagittal localizer",
            ]
            mock_acr.return_value = mock_inst

            mock_dcmread.return_value = Mock()

            protocol = ACRLargePhantomProtocol(dirs=self.dirs)

            task_names = [s.task_name for s in protocol.steps]
            expected_tasks = [
                "acr_geometric_accuracy",
                "acr_sagittal_geometric_accuracy",
                "acr_spatial_resolution",
                "acr_slice_thickness",
                "acr_slice_position",
                "acr_uniformity",
                "acr_ghosting",
                "acr_low_contrast_object_detectability",
                "acr_snr",
            ]

            for task in expected_tasks:
                self.assertIn(task, task_names)


if __name__ == "__main__":
    unittest.main()
