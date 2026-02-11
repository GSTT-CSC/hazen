"""Orchestration Module for performing multiple tasks."""

from __future__ import annotations

# Type Checking
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Python imports
    from collections.abc import Sequence
    from pathlib import Path

    # Local imports
    from hazenlib.HazenTask import HazenTask

# Python imports
import importlib
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TypeVar

# Module imports
from pydicom import dcmread

# Local imports
from hazenlib.ACRObject import ACRObject
from hazenlib.exceptions import (
    UnknownAcquisitionTypeError,
    UnknownTaskNameError,
)
from hazenlib.types import Measurement, PhantomType, Result, TaskMetadata
from hazenlib.utils import get_dicom_files, wait_on_parallel_results

logger = logging.getLogger(__name__)

# Note that if changing the TASK_REGISTRY keys you should
# update Protocol to match up with the task registry.
TASK_REGISTRY = {
    # MagNET #
    "snr": TaskMetadata(
        module_name="snr",
        class_name="SNR",
        single_image=True,
        phantom=PhantomType.MAGNET,
    ),
    "ghosting": TaskMetadata(
        module_name="ghosting",
        class_name="Ghosting",
        single_image=True,
        phantom=PhantomType.MAGNET,
    ),
    "uniformity": TaskMetadata(
        module_name="uniformity",
        class_name="Uniformity",
        single_image=True,
        phantom=PhantomType.MAGNET,
    ),
    "spatial_resolution": TaskMetadata(
        module_name="spatial_resolution",
        class_name="SpatialResolution",
        single_image=True,
        phantom=PhantomType.MAGNET,
    ),
    "slice_width": TaskMetadata(
        module_name="slice_width",
        class_name="SliceWidth",
        single_image=True,
        phantom=PhantomType.MAGNET,
    ),
    "slice_position": TaskMetadata(
        module_name="slice_position",
        class_name="SlicePosition",
        single_image=True,
        phantom=PhantomType.MAGNET,
    ),
    "snr_map": TaskMetadata(
        module_name="snr_map",
        class_name="SNRMap",
        single_image=True,
        phantom=PhantomType.MAGNET,
    ),
    # ACR #
    "acr_geometric_accuracy": TaskMetadata(
        module_name="acr_geometric_accuracy",
        class_name="ACRGeometricAccuracy",
        single_image=False,
        phantom=PhantomType.ACR,
    ),
    "acr_ghosting": TaskMetadata(
        module_name="acr_ghosting",
        class_name="ACRGhosting",
        single_image=False,
        phantom=PhantomType.ACR,
    ),
    "acr_low_contrast_object_detectability": TaskMetadata(
        module_name="acr_low_contrast_object_detectability",
        class_name="ACRLowContrastObjectDetectability",
        single_image=False,
        phantom=PhantomType.ACR,
    ),
    "acr_object_detectability": TaskMetadata(
        module_name="acr_object_detectability",
        class_name="ACRObjectDetectability",
        single_image=False,
        phantom=PhantomType.ACR,
    ),
    "acr_slice_position": TaskMetadata(
        module_name="acr_slice_position",
        class_name="ACRSlicePosition",
        single_image=False,
        phantom=PhantomType.ACR,
    ),
    "acr_slice_thickness": TaskMetadata(
        module_name="acr_slice_thickness",
        class_name="ACRSliceThickness",
        single_image=False,
        phantom=PhantomType.ACR,
    ),
    "acr_snr": TaskMetadata(
        module_name="acr_snr",
        class_name="ACRSNR",
        single_image=False,
        phantom=PhantomType.ACR,
    ),
    "acr_spatial_resolution": TaskMetadata(
        module_name="acr_spatial_resolution",
        class_name="ACRSpatialResolution",
        single_image=False,
        phantom=PhantomType.ACR,
    ),
    "acr_sagittal_geometric_accuracy": TaskMetadata(
        module_name="acr_sagittal_geometric_accuracy",
        class_name="ACRSagittalGeometricAccuracy",
        single_image=False,
        phantom=PhantomType.ACR,
    ),
    "acr_uniformity": TaskMetadata(
        module_name="acr_uniformity",
        class_name="ACRUniformity",
        single_image=False,
        phantom=PhantomType.ACR,
    ),
    # Caliber
    "relaxometry": TaskMetadata(
        module_name="relaxometry",
        class_name="Relaxometry",
        single_image=False,
        phantom=PhantomType.CALIBER,
    ),
}


def init_task(
    selected_task: str,
    files: list[str],
    report: bool,
    report_dir: str,
    **kwargs,
) -> HazenTask:
    """Initialise object of the correct HazenTask class.

    Args:
        selected_task (string): name of task script/module to load
        files (list): list of filepaths to DICOM images
        report (bool): whether to generate report images
        report_dir (string): path to folder to save report images to
        kwargs: any other key word arguments

    Returns:
        an object of the specified HazenTask class

    """
    try:
        meta = TASK_REGISTRY[selected_task]
    except KeyError as err:
        msg = f"Unknown task: {selected_task}"
        logger.exception(
            "%s. Supported tasks are:\n%s",
            msg,
            "\n\t".join(TASK_REGISTRY),
        )
        raise ValueError(msg) from err

    # Import module
    task_module = importlib.import_module(f"hazenlib.tasks.{meta.module_name}")

    # Get explicit class
    try:
        task_class = getattr(task_module, meta.class_name)
    except AttributeError as err:
        msg = f"Module {meta.module_name} has no class '{meta.class_name}'"
        raise ImportError(msg) from err

    return task_class(
        input_data=files,
        report=report,
        report_dir=report_dir,
        **kwargs,
    )


class AcquisitionType(Enum):
    """Supported Acquisition Types."""

    ACR_T1 = "ACR T1"
    ACR_T2 = "ACR T2"
    ACR_SL = "ACR Sagittal Localizer"

    @classmethod
    def from_string(cls, value: str) -> AcquisitionType:
        """Create AcquisitionType from string with fuzzy matching.

        Handles British/American spelling variations (Localiser vs Localizer)
        and case insensitivity.

        Args:
            value: String representation of acquisition type

        Returns:
            AcquisitionType enum value

        Raises:
            UnknownAcquisitionTypeError: If string cannot be matched to
                a known type

        Example:
            >>> AcquisitionType.from_string("t1")
            AcquisitionType.ACR_T1
            >>> AcquisitionType.from_string("sagittal localiser")
            AcquisitionType.ACR_SL

        """
        normalized = value.lower().replace("localiser", "localizer")
        mapping = {
            "t1": cls.ACR_T1,
            "t2": cls.ACR_T2,
            "sagittal localizer": cls.ACR_SL,
            "sagittal": cls.ACR_SL,
            "localizer": cls.ACR_SL,
        }
        if normalized in mapping:
            return mapping[normalized]
        raise UnknownAcquisitionTypeError(value)


@dataclass(frozen=True)
class ProtocolStep:
    """Single protocol step representing the execution of a single task.

    Attributes:
        task_name: Name of the task as registered in TASK_REGISTRY
        acquisition_type: Type of acquisition this step applies to

    """

    task_name: str
    acquisition_type: AcquisitionType

    def __post_init__(self) -> None:
        """Validate that task name exists in the global registry."""
        if self.task_name not in TASK_REGISTRY:
            available = ", ".join(sorted(TASK_REGISTRY.keys()))
            raise UnknownTaskNameError(self.task_name, available)


@dataclass
class Protocol:
    """Orchestrator for collections of tasks.

    Attributes:
        name: Human-readable identifier for this protocol
        steps: Ordered collection of protocol steps to execute

    """

    name: str
    steps: tuple[ProtocolStep, ...] = field(default_factory=tuple)

    @classmethod
    def from_config(cls, config_path: Path) -> Protocol:
        """Load protocol from configuration file."""
        msg = "The class method 'from_config' has not been implemented yet."
        raise NotImplementedError(msg)


class ProtocolResult(Result):
    """Class for the protocol result."""

    def __post_init__(self) -> None:
        """Initialise the results list."""
        super().__post_init__()

        # Set initial result to contain
        # protocol information.
        self._results: list[Result] = [
            Result(
                self.task,
                self.desc,
                self.files,
            ),
        ]

    @property
    def measurements(self) -> tuple[Measurement, ...]:
        """Tuple of initial result measurements."""
        return self.results[0].measurements

    def add_measurement(self, measurement: Measurement) -> None:
        """Add a measurement to the initial result."""
        self.results[0].add_measurement(measurement)

    @property
    def report_images(self) -> tuple[str, ...]:
        """Tuple of initial report image locations."""
        return self.results[0].report_images

    def add_report_image(self, image_path: str | Sequence[str]) -> None:
        """Add a report image to the initial result."""
        return self.results[0].add_report_image(image_path)

    @property
    def results(self) -> tuple[Result, ...]:
        """Return an immutable results list."""
        return tuple(self._results)

    def add_result(self, result: Result) -> None:
        """Add a result to the list."""
        self._results.append(result)


T = TypeVar("T")


class ACRLargePhantomProtocol(Protocol):
    """Protocol for ACR Large Phantom."""

    def __init__(self, dirs: list[str | Path], **kwargs: T.kwargs) -> None:
        """Set up the protocol for the ACR Large Phantom."""
        self.name = "ACR Large Phantom"
        self.steps = (
            # Geometric Accuracy.
            ProtocolStep("acr_geometric_accuracy", AcquisitionType.ACR_T1),
            ProtocolStep(
                "acr_sagittal_geometric_accuracy",
                AcquisitionType.ACR_SL,
            ),
            # High Contrast Object Detection.
            ProtocolStep("acr_spatial_resolution", AcquisitionType.ACR_T1),
            ProtocolStep("acr_spatial_resolution", AcquisitionType.ACR_T2),
            # Slice Thickness Accuracy.
            ProtocolStep("acr_slice_thickness", AcquisitionType.ACR_T1),
            ProtocolStep("acr_slice_thickness", AcquisitionType.ACR_T2),
            # Slice Position Accuracy.
            ProtocolStep("acr_slice_position", AcquisitionType.ACR_T1),
            ProtocolStep("acr_slice_position", AcquisitionType.ACR_T2),
            # Image Intensity Uniformity.
            ProtocolStep("acr_uniformity", AcquisitionType.ACR_T1),
            ProtocolStep("acr_uniformity", AcquisitionType.ACR_T2),
            # Percent Signal Ghosting.
            ProtocolStep("acr_ghosting", AcquisitionType.ACR_T1),
            # Low Contrast Object Detectability.
            ProtocolStep(
                "acr_low_contrast_object_detectability",
                AcquisitionType.ACR_T1,
            ),
            ProtocolStep(
                "acr_low_contrast_object_detectability",
                AcquisitionType.ACR_T2,
            ),
            # SNR
            ProtocolStep("acr_snr", AcquisitionType.ACR_T1),
            ProtocolStep("acr_snr", AcquisitionType.ACR_T2),
        )

        kwargs.setdefault("report", False)
        kwargs.setdefault("report_dir", None)
        self.kwargs = kwargs

        required_acquisition_types = {s.acquisition_type for s in self.steps}
        if len(dirs) != (
            num_aq := len(required_acquisition_types)
        ):
            msg = f"Incorrect number of directories - should be {num_aq}"
            logger.exception("%s but got %i", msg, len(dirs))
            raise ValueError(msg)

        files_list = (get_dicom_files(d) for d in dirs)
        self.file_groups = {}
        for files in files_list:
            acr_obj = ACRObject([dcmread(files[0], stop_before_pixels=False)])
            acquisition_type = AcquisitionType.from_string(
                acr_obj.acquisition_type(strict=True),
            )
            self.file_groups[acquisition_type] = files

        if len(self.file_groups.keys()) != len(required_acquisition_types):
            acquisition_types = set(self.file_groups.keys())
            msg = (
                f"Missing sequences - only {acquisition_types}"
                " were found meaning"
                f" {required_acquisition_types - acquisition_types}"
                " were not present."
            )
            logger.exception(
                "%s The following directories were passed: %s",
                msg,
                dirs,
            )
            raise ValueError(msg)

    def run(self, *, debug: bool = False) -> ProtocolResult:
        """Run the Protocol for each of the steps."""
        results = ProtocolResult(
            self.name,
            ", ".join(
                f"{s.task_name} {s.acquisition_type}" for s in self.steps
            ),
            list(self.file_groups.values()),
        )

        arg_list = [
            (step, self.file_groups, self.kwargs) for step in self.steps
        ]
        parallel_results = wait_on_parallel_results(
            _execute_step,
            arg_list,
            debug=debug,
        )
        for r in parallel_results:
            results.add_result(r)
        return results


def _execute_step(
    step: ProtocolStep,
    file_groups: dict,
    kwargs: T.kwargs,
) -> Result:
    """Encapsulate the work for a single step."""
    task = init_task(
        step.task_name,
        file_groups[step.acquisition_type],
        **kwargs,
    )
    return task.run()
