"""Tests for the orchestration module."""

# ruff: noqa: PT009 PT027 SLF001 S108

# Python imports
import unittest
from collections.abc import Callable
from pathlib import Path
from unittest.mock import Mock, call, mock_open, patch

from hazenlib.exceptions import (
    UnknownAcquisitionTypeError,
    UnknownTaskNameError,
)
from hazenlib.HazenTask import HazenTask
from hazenlib.orchestration import (
    PROTOCOL_REGISTRY,
    TASK_REGISTRY,
    AcquisitionType,
    ACRLargePhantomProtocol,
    BatchConfig,
    JobTaskConfig,
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


class TestProtocolResultToDocx(unittest.TestCase):
    """Unit tests for ProtocolResult.to_docx method."""

    @patch("hazenlib.orchestration.Document")
    @patch("hazenlib.orchestration.Inches")
    def test_empty_results_raises_error(
        self,
        mock_inches: Mock,  # noqa: ARG002
        mock_document: Mock,  # noqa: ARG002
    ) -> None:
        """Verify ValueError raised when results list is empty."""
        protocol_result = ProtocolResult(task="Protocol", desc="test")
        # Access private attribute to simulate uninitialized state
        protocol_result._results = []  # type: ignore[misc]

        with patch.object(ProtocolResult, "results", None):
            with self.assertRaises(ValueError) as context:
                protocol_result.to_docx()
            self.assertIn("Results cannot be empty", str(context.exception))

    @patch("hazenlib.orchestration.Document")
    @patch("hazenlib.orchestration.Inches")
    def test_creates_document_without_template(
        self,
        mock_inches: Mock,  # noqa: ARG002
        mock_document_class: Mock,
    ) -> None:
        """Verify Document created and protocol header added when no template."""
        mock_doc = Mock()
        mock_document_class.return_value = mock_doc
        protocol_result = ProtocolResult(
            task="MyProtocol",
            desc="test description",
        )

        result = protocol_result.to_docx()

        self.assertEqual(result, mock_doc)
        mock_document_class.assert_called_once_with()
        mock_doc.add_heading.assert_called_once_with(
            "QA Report: MyProtocol",
            0,
        )

    @patch("hazenlib.orchestration.Document")
    @patch("hazenlib.orchestration.Inches")
    @patch("pathlib.Path.exists")
    def test_uses_template_when_provided(
        self,
        mock_exists: Mock,
        mock_inches: Mock,  # noqa: ARG002
        mock_document_class: Mock,
    ) -> None:
        """Verify template document is loaded when path provided."""
        mock_exists.return_value = True

        mock_doc = Mock()
        mock_document_class.return_value = mock_doc
        protocol_result = ProtocolResult(task="Protocol", desc="test")
        template_path = Path("/path/to/template.docx")

        result = protocol_result.to_docx(template_path=template_path)

        self.assertEqual(result, mock_doc)
        mock_document_class.assert_called_once_with(template_path)
        # Should not add protocol header when using template
        protocol_header_calls = [
            c
            for c in mock_doc.add_heading.call_args_list
            if c == call("QA Report: Protocol", 0)
        ]
        self.assertEqual(len(protocol_header_calls), 0)

    @patch("hazenlib.orchestration.Document")
    @patch("hazenlib.orchestration.Inches")
    def test_skips_protocol_metadata_result(
        self,
        mock_inches: Mock,  # noqa: ARG002
        mock_document_class: Mock,
    ) -> None:
        """Verify result matching protocol task name is skipped in output."""
        mock_doc = Mock()
        mock_document_class.return_value = mock_doc
        protocol_result = ProtocolResult(task="Protocol", desc="test")

        # Result with same task name as protocol (should be skipped)
        same_name_result = Mock(spec=Result)
        same_name_result.task = "Protocol"

        # Result with different task name (should be included)
        subtask_result = Mock(spec=Result)
        subtask_result.task = "SubTask"
        subtask_result.filtered.return_value = subtask_result
        subtask_result.measurements = []
        subtask_result.report_images = []

        protocol_result.add_result(same_name_result)
        protocol_result.add_result(subtask_result)

        mock_table = Mock()
        mock_doc.add_table.return_value = mock_table
        mock_row_mock = Mock()
        mock_row_mock.cells = [Mock() for _ in range(6)]
        mock_table.rows = [mock_row_mock]
        mock_new_row = Mock()
        mock_new_row.cells = [Mock() for _ in range(6)]
        mock_table.add_row.return_value = mock_new_row

        protocol_result.to_docx()

        # Check headings: protocol header (0) + one task (1), but not metadata
        heading_calls = mock_doc.add_heading.call_args_list
        self.assertEqual(len(heading_calls), 2)
        self.assertEqual(heading_calls[0], call("QA Report: Protocol", 0))
        self.assertEqual(heading_calls[1], call("SubTask", level=1))

    @patch("hazenlib.orchestration.Document")
    @patch("hazenlib.orchestration.Inches")
    def test_applies_level_filter_to_results(
        self,
        mock_inches: Mock,  # noqa: ARG002
        mock_document_class: Mock,
    ) -> None:
        """Verify level parameter passed to result"s filtered method."""
        mock_doc = Mock()
        mock_document_class.return_value = mock_doc
        protocol_result = ProtocolResult(task="Protocol", desc="test")

        mock_result = Mock(spec=Result)
        mock_result.task = "Task"
        mock_result.filtered.return_value = mock_result
        mock_result.measurements = []
        mock_result.report_images = []

        protocol_result.add_result(mock_result)

        mock_table = Mock()
        mock_doc.add_table.return_value = mock_table
        mock_row_mock = Mock()
        mock_row_mock.cells = [Mock() for _ in range(6)]
        mock_table.rows = [mock_row_mock]

        mock_new_row = Mock()
        mock_new_row.cells = [Mock() for _ in range(6)]
        mock_table.add_row.return_value = mock_new_row

        protocol_result.to_docx(level="final")

        mock_result.filtered.assert_called_once_with("final")

    @patch("hazenlib.orchestration.Document")
    @patch("hazenlib.orchestration.Inches")
    def test_creates_table_with_correct_structure(
        self,
        mock_inches: Mock,  # noqa: ARG002
        mock_document_class: Mock,
    ) -> None:
        """Verify measurements table has 6 columns and correct headers."""
        mock_doc = Mock()
        mock_table = Mock()
        # Mock header cells (6 columns)
        mock_hdr_cells = [Mock() for _ in range(6)]
        mock_table.rows = [Mock(cells=mock_hdr_cells)]
        # Mock row cells for measurement data
        mock_row_cells = [Mock() for _ in range(6)]
        mock_table.add_row.return_value = Mock(cells=mock_row_cells)
        mock_doc.add_table.return_value = mock_table
        mock_document_class.return_value = mock_doc

        protocol_result = ProtocolResult(task="Protocol", desc="test")

        # Create mock measurement
        mock_measurement = Mock()
        mock_measurement.name = "snr"
        mock_measurement.type = "measured"
        mock_measurement.subtype = ""
        mock_measurement.description = "Signal to noise"
        mock_measurement.value = 42.5
        mock_measurement.unit = "ratio"

        mock_result = Mock(spec=Result)
        mock_result.task = "Task"
        mock_result.filtered.return_value = mock_result
        mock_result.measurements = [mock_measurement]
        mock_result.report_images = []

        protocol_result.add_result(mock_result)
        protocol_result.to_docx()

        # Verify table created with correct dimensions and style
        mock_doc.add_table.assert_called_once_with(rows=1, cols=6)
        self.assertEqual(mock_table.style, "Light Grid Accent 1")

        # Verify headers populated
        expected_headers = [
            "Name",
            "Type",
            "Subtype",
            "Description",
            "Value",
            "Unit",
        ]
        for i, header in enumerate(expected_headers):
            self.assertEqual(mock_hdr_cells[i].text, header)

        # Verify measurement data converted to strings
        mock_table.add_row.assert_called_once()
        self.assertEqual(mock_row_cells[0].text, "snr")
        self.assertEqual(mock_row_cells[1].text, "measured")
        self.assertEqual(mock_row_cells[4].text, "42.5")

    @patch("hazenlib.orchestration.Document")
    @patch("hazenlib.orchestration.Inches")
    @patch("pathlib.Path.exists")
    def test_adds_report_images_with_five_inch_width(
        self,
        mock_exists: Mock,
        mock_inches: Mock,
        mock_document_class: Mock,
    ) -> None:
        """Verify report images embedded with 5.0 inch width."""
        mock_exists.return_value = True

        mock_doc = Mock()
        mock_document_class.return_value = mock_doc
        mock_inches.return_value = "5_inches_width"

        protocol_result = ProtocolResult(task="Protocol", desc="test")

        mock_result = Mock(spec=Result)
        mock_result.task = "Task"
        mock_result.filtered.return_value = mock_result
        mock_result.measurements = []
        mock_result.report_images = [
            "/path/to/image1.png",
            "/path/to/image2.png",
        ]

        protocol_result.add_result(mock_result)

        mock_table = Mock()
        mock_doc.add_table.return_value = mock_table
        mock_row_mock = Mock()
        mock_row_mock.cells = [Mock() for _ in range(6)]
        mock_table.rows = [mock_row_mock]
        mock_new_row = Mock()
        mock_new_row.cells = [Mock() for _ in range(6)]
        mock_table.add_row.return_value = mock_new_row

        protocol_result.to_docx(level="all")

        self.assertEqual(mock_doc.add_picture.call_count, 2)
        mock_doc.add_picture.assert_any_call(
            "/path/to/image1.png",
            width="5_inches_width",
        )
        mock_doc.add_picture.assert_any_call(
            "/path/to/image2.png",
            width="5_inches_width",
        )
        mock_inches.assert_called_with(5.0)


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


class TestJobTaskConfig(unittest.TestCase):
    """Unit tests for JobTaskConfig dataclass."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.test_folder = Path("/tmp/test_folder")
        self.existing_folder = Path("/existing/folder")
        self.nonexistent_folder = Path("/nonexistent/folder")

    @patch.dict(TASK_REGISTRY, {"test_task": Mock()}, clear=False)
    @patch("pathlib.Path.exists")
    def test_valid_task_in_task_registry(self, mock_exists: Mock) -> None:
        """Verify task in TASK_REGISTRY sets is_protocol=False."""
        mock_exists.return_value = True

        config = JobTaskConfig(
            task="test_task",
            folders=[self.existing_folder],
        )

        self.assertEqual(config.task, "test_task")
        self.assertFalse(config.is_protocol)
        self.assertEqual(config.overrides, {})

    @patch.dict(PROTOCOL_REGISTRY, {"test_protocol": Mock()}, clear=False)
    @patch("pathlib.Path.exists")
    def test_valid_task_in_protocol_registry(self, mock_exists: Mock) -> None:
        """Verify task in PROTOCOL_REGISTRY sets is_protocol=True."""
        mock_exists.return_value = True

        config = JobTaskConfig(
            task="test_protocol",
            folders=[
                self.existing_folder,
                self.existing_folder,
                self.existing_folder,
            ],
        )

        self.assertTrue(config.is_protocol)

    def test_unknown_task_raises_error(self) -> None:
        """Verify UnknownTaskNameError for unregistered tasks."""
        with self.assertRaises(UnknownTaskNameError) as context:
            JobTaskConfig(
                task="unknown_task",
                folders=[self.existing_folder],
            )

        self.assertIn("Unknown task", str(context.exception))
        self.assertIn("unknown_task", str(context.exception))

    @patch.dict(
        PROTOCOL_REGISTRY,
        {"acr_all": Mock()},
        clear=False,
    )
    @patch("pathlib.Path.exists")
    def test_acr_all_protocol_validates_three_folders(
        self,
        mock_exists: Mock,
    ) -> None:
        """Verify acr_all protocol accepts exactly 3 folders."""
        mock_exists.return_value = True

        config = JobTaskConfig(
            task="acr_all",
            folders=[
                self.existing_folder,
                self.existing_folder,
                self.existing_folder,
            ],
        )

        self.assertTrue(config.is_protocol)

    @patch.dict(
        PROTOCOL_REGISTRY,
        {"acr_all": Mock()},
        clear=False,
    )
    def test_acr_all_with_wrong_folder_count_raises_error(self) -> None:
        """Verify ValueError when acr_all doesn't have exactly 3 folders."""
        with self.assertRaises(ValueError) as context:
            JobTaskConfig(
                task="acr_all",
                folders=[self.existing_folder, self.existing_folder],
            )

        self.assertIn("3 directories", str(context.exception))
        self.assertIn("got 2", str(context.exception))

    def test_nonexistent_folder_raises_file_not_found(self) -> None:
        """Verify FileNotFoundError when folder doesn't exist."""
        with (
            patch.dict(TASK_REGISTRY, {"test_task": Mock()}, clear=False),
            self.assertRaises(FileNotFoundError) as context,
        ):
            JobTaskConfig(
                task="test_task",
                folders=[self.nonexistent_folder],
            )

        self.assertIn("Folder not found", str(context.exception))

    @patch.dict(TASK_REGISTRY, {"test_task": Mock()}, clear=False)
    @patch("pathlib.Path.exists")
    def test_overrides_stored(self, mock_exists: Mock) -> None:
        """Verify overrides dict is stored correctly."""
        mock_exists.return_value = True
        overrides = {"param1": "value1", "param2": 42}

        config = JobTaskConfig(
            task="test_task",
            folders=[self.existing_folder],
            overrides=overrides,
        )

        self.assertEqual(config.overrides, overrides)


class TestBatchConfig(unittest.TestCase):
    """Unit tests for BatchConfig dataclass."""

    def setUp(self) -> None:
        """Set up test configuration data."""
        self.config_data = {
            "version": "1.0",
            "description": "Test batch config",
            "output": "/tmp/output",
            "jobs": [
                {
                    "task": "snr",
                    "folders": ["/tmp/data"],
                    "overrides": {"param": "value"},
                },
            ],
        }
        self.config_path = Path("/tmp/test_config.yaml")

    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.open", mock_open(read_data="{}"))
    @patch("yaml.safe_load")
    def test_from_config_loads_yaml(
        self,
        mock_yaml: Mock,
        mock_exists: Mock,
    ) -> None:
        """Verify YAML file is loaded and parsed."""
        mock_exists.return_value = True
        mock_yaml.return_value = self.config_data

        with patch.dict(TASK_REGISTRY, {"snr": Mock()}, clear=False):
            config = BatchConfig.from_config(self.config_path)

        self.assertEqual(config.version, "1.0")
        self.assertEqual(config.description, "Test batch config")
        self.assertEqual(len(config.jobs), 1)

    @patch("pathlib.Path.open", mock_open(read_data="{}"))
    @patch("yaml.safe_load")
    def test_from_config_resolves_relative_paths(
        self,
        mock_yaml: Mock,
    ) -> None:
        """Verify relative paths are resolved relative to config file."""
        mock_yaml.return_value = {
            **self.config_data,
            "output": "relative_output",
            "jobs": [{"task": "snr", "folders": ["relative_folder"]}],
        }

        with (
            patch("pathlib.Path.exists") as mock_exists,
            patch.dict(TASK_REGISTRY, {"snr": Mock()}, clear=False),
        ):
            mock_exists.return_value = True
            config = BatchConfig.from_config("/home/user/config.yaml")

        # Check that relative paths were resolved
        self.assertTrue(str(config.output).startswith("/home/user"))
        job_folders = config.jobs[0].folders
        self.assertTrue(job_folders[0].as_posix().startswith("/home/user"))

    @patch("pathlib.Path.open", mock_open(read_data="{}"))
    @patch("yaml.safe_load")
    def test_from_config_preserves_absolute_paths(
        self,
        mock_yaml: Mock,
    ) -> None:
        """Verify absolute paths are not modified."""
        mock_yaml.return_value = {
            **self.config_data,
            "output": "/absolute/output/path",
        }

        with (
            patch("pathlib.Path.exists") as mock_exists,
            patch.dict(TASK_REGISTRY, {"snr": Mock()}, clear=False),
        ):
            mock_exists.return_value = True
            config = BatchConfig.from_config("config.yaml")

        self.assertEqual(str(config.output), "/absolute/output/path")

    def test_validate_hazen_version_valid_constraint(self) -> None:
        """Verify no error when version constraint satisfied."""
        # Test with broad constraint that should always pass
        with patch("hazenlib.orchestration.__version__", "2.0.0"):
            # Should not raise
            BatchConfig._validate_hazen_version(">=1.0.0")

    def test_validate_hazen_version_mismatch_raises_runtime_error(
        self,
    ) -> None:
        """Verify RuntimeError when version constraint not satisfied."""
        with (
            patch("hazenlib.orchestration.__version__", "1.0.0"),
            self.assertRaises(RuntimeError) as context,
        ):
            BatchConfig._validate_hazen_version(">=2.0.0")

        self.assertIn("Version constraint mismatch", str(context.exception))
        self.assertIn("pip install", str(context.exception))

    def test_validate_hazen_version_invalid_specifier(self) -> None:
        """Verify ValueError for invalid version specifier."""
        with self.assertRaises(ValueError) as context:
            BatchConfig._validate_hazen_version("not_a_valid_specifier")

        self.assertIn(
            "Invalid hazen_version_constraint",
            str(context.exception),
        )

    def test_validate_schema_version_current(self) -> None:
        """Verify no migration needed for current schema."""
        data = {"key": "value"}
        result = BatchConfig._validate_schema_version("1.0", data)
        self.assertEqual(result, data)

    def test_validate_schema_version_newer_raises_error(self) -> None:
        """Verify ValueError when config schema is newer than supported."""
        with self.assertRaises(ValueError) as context:
            BatchConfig._validate_schema_version("99.0", {})

        self.assertIn("newer than supported", str(context.exception))

    def test_migrate_config_not_implemented(self) -> None:
        """Verify NotImplementedError for schema migration."""
        with self.assertRaises(NotImplementedError):
            BatchConfig._migrate_config({}, "0.9")

    @patch("pathlib.Path.open", mock_open(read_data="{}"))
    @patch("yaml.safe_load")
    def test_from_config_validates_hazen_version(
        self,
        mock_yaml: Mock,
    ) -> None:
        """Verify hazen version constraint is validated during load."""
        mock_yaml.return_value = {
            **self.config_data,
            "hazen_version_constraint": ">=2.0.0",
        }

        with (
            patch("pathlib.Path.exists") as mock_exists,
            patch.object(
                BatchConfig,
                "_validate_hazen_version",
            ) as mock_validate,
            patch.dict(TASK_REGISTRY, {"snr": Mock()}, clear=False),
        ):
            mock_exists.return_value = True
            BatchConfig.from_config(self.config_path)

            mock_validate.assert_called_once_with(">=2.0.0")

    @patch("pathlib.Path.open", mock_open(read_data="{}"))
    @patch("yaml.safe_load")
    def test_from_config_optional_report_paths(self, mock_yaml: Mock) -> None:
        """Verify optional report_docx and report_template paths handled."""
        mock_yaml.return_value = {
            **self.config_data,
            "report_docx": "report.docx",
            "report_template": "template.docx",
        }

        with (
            patch("pathlib.Path.exists") as mock_exists,
            patch.dict(TASK_REGISTRY, {"snr": Mock()}, clear=False),
        ):
            mock_exists.return_value = True
            config = BatchConfig.from_config("/tmp/config.yaml")

        self.assertEqual(config.report_docx, "/tmp/report.docx")
        self.assertEqual(
            config.report_template,
            "/tmp/template.docx",
        )

    @patch("pathlib.Path.open", mock_open(read_data="{}"))
    @patch("yaml.safe_load")
    def test_from_config_default_values(self, mock_yaml: Mock) -> None:
        """Verify default values applied when not in config."""
        minimal_data = {
            "output": "/tmp/output",
            "jobs": [],
        }
        mock_yaml.return_value = minimal_data

        with patch("pathlib.Path.exists", return_value=True):
            config = BatchConfig.from_config(self.config_path)

        self.assertEqual(config.version, "1.0")  # default
        self.assertEqual(config.description, "")  # default
        self.assertEqual(config.levels, "final")  # default
        self.assertIsNone(config.report_docx)  # default
        self.assertEqual(config.defaults, {})  # default

    @patch("pathlib.Path.open", mock_open(read_data="{}"))
    @patch("yaml.safe_load")
    def test_run_dry_run_mode(self, mock_yaml: Mock) -> None:
        """Verify dry run prints job info without executing."""
        mock_yaml.return_value = self.config_data

        with (
            patch("pathlib.Path.exists") as mock_exists,
            patch.dict(TASK_REGISTRY, {"snr": Mock()}, clear=False),
            patch("builtins.print") as mock_print,
            patch("hazenlib.orchestration.get_dicom_files") as mock_get_files,
        ):
            mock_exists.return_value = True
            mock_get_files.return_value = ["file1.dcm", "file2.dcm"]
            config = BatchConfig.from_config(self.config_path)
            result = config.run(dry_run=True)

        # Verify result returned without executing
        self.assertEqual(result.task, "Batch Configuration Job")

        # Verify dry run output printed
        print_calls = [str(call) for call in mock_print.call_args_list]
        self.assertTrue(
            any("Dry run complete" in c for c in print_calls),
        )

    @patch("pathlib.Path.open", mock_open(read_data="{}"))
    @patch("yaml.safe_load")
    def test_run_executes_protocols(self, mock_yaml: Mock) -> None:
        """Verify protocols are executed during run."""
        mock_yaml.return_value = {
            **self.config_data,
            "jobs": [{"task": "acr_all", "folders": ["/t1", "/t2", "/sl"]}],
        }

        mock_protocol_class = Mock()
        mock_protocol_instance = Mock()
        mock_protocol_instance.run.return_value = Mock(
            task="acr_all",
            results=[],
        )
        mock_protocol_class.return_value = mock_protocol_instance

        with (
            patch("pathlib.Path.exists") as mock_exists,
            patch.dict(
                PROTOCOL_REGISTRY,
                {"acr_all": mock_protocol_class},
                clear=False,
            ),
            patch("hazenlib.orchestration.get_dicom_files") as mock_get_files,
        ):
            mock_exists.return_value = True
            mock_get_files.return_value = ["file.dcm"]
            config = BatchConfig.from_config(self.config_path)
            config.run(dry_run=False)

        mock_protocol_class.assert_called_once()
        mock_protocol_instance.run.assert_called_once()

    @patch("pathlib.Path.open", mock_open(read_data="{}"))
    @patch("yaml.safe_load")
    def test_run_executes_tasks(self, mock_yaml: Mock) -> None:
        """Verify individual tasks are executed during run."""
        mock_yaml.return_value = self.config_data

        with (
            patch("pathlib.Path.exists") as mock_exists,
            patch.dict(TASK_REGISTRY, {"snr": Mock()}, clear=False),
            patch("hazenlib.orchestration.get_dicom_files") as mock_get_files,
            patch(
                "hazenlib.orchestration.wait_on_parallel_results",
            ) as mock_wait,
        ):
            mock_exists.return_value = True
            mock_get_files.return_value = ["file1.dcm", "file2.dcm"]
            mock_wait.return_value = []

            config = BatchConfig.from_config(self.config_path)
            config.run(dry_run=False)

            # Verify task jobs were prepared for parallel execution
            mock_wait.assert_called_once()
            call_args = mock_wait.call_args
            self.assertEqual(call_args[1]["debug"], False)


if __name__ == "__main__":
    unittest.main()
