"""Module for discovering QA tasks from a directory."""

from __future__ import annotations

# Python imports
import datetime
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# Module imports
import numpy as np
import pydicom

# Local imports
from hazenlib._version import __version__
from hazenlib.ACRObject import ACRObject
from hazenlib.orchestration import BatchConfig, JobTaskConfig
from hazenlib.utils import is_enhanced_dicom

logger = logging.getLogger(__name__)


class Orientation(Enum):
    """Orientation enumerator."""

    SAG = "Sagittal"
    COR = "Coronal"
    AX = "Axial"


@dataclass
class AcquisitionMetadata:
    """Acquisition metadata."""

    receiver_coil: str
    te: int | float
    tr: int | float
    sequence_name: str
    acquisition_matrix: tuple[int, int, ...]
    orientation: Orientation


@dataclass(frozen=True)
class DiscoveredAcquisition:
    """Discovered acquisition from folder."""

    path: str | Path
    dicom: pydicom.Dataset
    metadata: AcquisitionMetadata
    acquisition_time: datetime.datetime

    def is_likely_the_same_as(self, other: DiscoveredAcquisition) -> bool:
        """Acquisition equality, i.e. is like the same as other."""
        return self.metadata == other.metadata

    @property
    def type(self) -> str:
        """Return the acquisition task type."""
        acr_obj = ACRObject([self.dicom])
        if acr_obj.acquisition_type(strict=True) != "Unknown":
            return "acr"
        if "snr" in self.path.name.lower():
            return "snr"
        return "Unknown"


    @classmethod
    def from_path(cls, path: str | Path) -> DiscoveredAcquisition:
        """Return a DiscoveredAcquisition from a path."""
        dcm = cls._get_dicoms_from_path(path)[0]

        return cls(
            path=Path(path),
            dicom=dcm,
            metadata=AcquisitionMetadata(
                receiver_coil=cls._get_receiver_coil(dcm),
                te=cls._get_te(dcm),
                tr=cls._get_tr(dcm),
                sequence_name=cls._get_sequence_name(dcm),
                acquisition_matrix=cls._get_acquisition_matrix(dcm),
                orientation=cls._get_orientation(dcm),
            ),
            acquisition_time=cls._get_acquisition_time(dcm),
        )

    @staticmethod
    def _get_dicoms_from_path(path: Path) -> list[Path]:
        return [pydicom.dcmread(p) for p in path.glob("*.dcm")]

    @staticmethod
    def _get_receiver_coil(dcm: pydicom.Dataset) -> str:
        """DICOM tag is a standard DICOM tag.

        That is, it is the same for both enhanced and standard DICOMS.
        """
        if is_enhanced_dicom(dcm):
            return dcm[
                (0x5200, 0x9229)
            ][0][
                (0x0018, 0x9042)
            ][0][
                (0x0018, 0x1250)
            ].value
        return dcm[(0x0018, 0x1250)].value

    @staticmethod
    def _get_te(dcm: pydicom.Dataset) -> int | float:
        if is_enhanced_dicom(dcm):
            return dcm[
                (0x5200, 0x9230)        # Per-Frame Functional Groups Sequence
            ][0][
                (0x0018, 0x9114)        # MR Echo Sequence
            ][0][
                (0x0018, 0x9082)        # Effective Echo Time
            ].value
        return dcm[(0x0018, 0x0081)].value

    @staticmethod
    def _get_tr(dcm: pydicom.Dataset) -> int | float:
        if is_enhanced_dicom(dcm):
            return dcm[             # Shared Functional Groups Sequence
                (0x5200, 0x9229)
            ][0][
                (0x0018, 0x9112)    # MR Timing and Related Parameters Sequence
            ][0][
                (0x0018, 0x0080)    # Repetition Time
            ].value
        return dcm[(0x0018, 0x0080)].value

    @staticmethod
    def _get_sequence_name(dcm: pydicom.Dataset) -> str:
        if is_enhanced_dicom(dcm):
            return dcm[(0x0018, 0x9005)].value
        return dcm[(0x0019, 0x109C)].value

    @staticmethod
    def _get_acquisition_matrix(dcm: pydicom.Dataset) -> tuple[int, ...]:
        return (
            dcm[(0x0028, 0x0010)].value,        # Rows
            dcm[(0x0028, 0x0011)].value,        # Cols
        )

    @staticmethod
    def _get_orientation(dcm: pydicom.Dataset) -> Orientation:
        if is_enhanced_dicom(dcm):
            orientation_patient = dcm[
                (0x5200, 0x9230)
            ][0][
                (0x0020, 0x9116)
            ][0][
                (0x0020, 0x0037)
            ].value
        else:
            orientation_patient = dcm[(0x0020, 0x0037)].value
        row_vec = orientation_patient[:3]
        col_vec = orientation_patient[3:]
        normal = np.cross(row_vec, col_vec)

        loc = np.where(np.abs(normal) == 1)[0][0]

        match loc:
            case 0:
                return Orientation.SAG
            case 1:
                return Orientation.COR
            case 2:
                return Orientation.AX
            case _:
                logger.warning(
                    "Could not determine orientation from orientation patient"
                    " DICOM field.\n"
                    "Row vector:\t %s\n"
                    "Col vector:\t %s\n"
                    "Normal vector:\t%s",
                    row_vec,
                    col_vec,
                    normal,
                )
                return None

    @staticmethod
    def _get_acquisition_time(dcm: pydicom.Dataset) -> datetime.datetime:
        if is_enhanced_dicom(dcm):
            tzstring = "%Y%m%d%H%M%S.%f"
            try:
                return datetime.datetime.strptime(
                    dcm[(0x0008, 0x002A)].value,
                    tzstring,
                )
            except ValueError:
                return datetime.datetime.strptime(
                    dcm[(0x0008, 0x002A)].value,
                    tzstring+"%z",
                )
        return datetime.datetime.strptime(
            f"{dcm[(0x0008, 0x0022)].value}{dcm[(0x0008, 0x0032)].value}",
            "%Y%m%d%H%M%S",
        )


@dataclass
class AcquisitionCollector:
    """Collect acquisitions together."""

    acquisitions: list[DiscoveredAcquisition]

    @property
    def jobs(self) -> list[JobTaskConfig]:
        """A list of jobs gathered from the acquisitions."""
        acq_sets = {
            "acr": ACRSet(),
            "snr": SNRSet(),
        }

        for acq in self.acquisitions:
            acq_set = acq_sets.get(acq.type, None)
            if acq_set is not None:
                acq_set.ingest(acq)
            else:
                logger.warning(
                    "Acquisition %s could not be recognised as: %s",
                    str(acq),
                    " or ".join(acq_sets),
                )

        return [job for acq_set in acq_sets.values() for job in acq_set.jobs]


##################
# Task Type Sets #
##################


class Ingestible(ABC):
    """Base class for task type sets."""

    @property
    @abstractmethod
    def jobs(self) -> list[JobTaskConfig]:
        """List of JobTaskConfig jobs."""
        ...

    @abstractmethod
    def ingest(self, acq: DiscoveredAcquisition) -> None:
        """Ingest an acquisition into the jobs list."""
        ...


@dataclass
class ACRSet(Ingestible):
    """ACR task type sets for the ACRLargePhantomProtocol."""

    t1: DiscoveredAcquisition | None = None
    t2: DiscoveredAcquisition | None = None
    sl: DiscoveredAcquisition | None = None

    @property
    def jobs(self) -> list[JobTaskConfig]:
        """The active jobs that have been ingested."""
        if any(p is None for p in (self.t1, self.t2, self.sl)):
            logger.warning(
                "ACRSet is currently incomplete due to missing acquisitions. "
                "Should have a T1, T2 and Sagittal Localizer acquisition "
                "but %s are missing.",
                [
                    seq
                    for seq, p in zip(
                        ("T1", "T2", "Sagittal Localizer"),
                        (self.t1, self.t2, self.sl),
                        strict=True,
                    )
                    if p is None
                ],
            )
            return []

        return [
            JobTaskConfig(
                task="acr_all",
                folders=[self.t1.path, self.t2.path, self.sl.path],
            ),
        ]

    def ingest(self, acq: DiscoveredAcquisition) -> None:
        """Ingest the acquisition data."""
        acq_type = ACRObject([acq.dicom]).acquisition_type(strict=True)

        match acq_type:
            case "T1":
                self.t1 = self.get_latest_acquisition(self.t1, acq)
            case "T2":
                self.t2 = self.get_latest_acquisition(self.t2, acq)
            case "Sagittal Localiser":
                self.sl = self.get_latest_acquisition(self.sl, acq)

    @staticmethod
    def get_latest_acquisition(
        acq1: DiscoveredAcquisition | None,
        acq2: DiscoveredAcquisition,
    ) -> DiscoveredAcquisition:
        """Get the latest acquisition."""
        if acq1 is None or acq2.acquisition_time > acq1.acquisition_time:
            return acq2
        return acq1


@dataclass
class SNRSet(Ingestible):
    """SNR task type sets."""

    _acqs: list[JobTaskConfig] = field(default_factory=list)

    @property
    def jobs(self) -> list[JobTaskConfig]:
        """List of current jobs."""
        _jobs = []

        # Tries to gather SNR pairs
        matched_acqs_idxs = set()
        for i, acq in enumerate(self._acqs):
            for j, possible_pair in enumerate(self._acqs):
                if i == j or j in matched_acqs_idxs:
                    continue
                if acq.is_likely_the_same_as(possible_pair):
                    _jobs.append(
                        JobTaskConfig(
                            task="snr",
                            folders=[acq.path],
                            overrides={
                                "subtract": possible_pair.path,
                            },
                        ),
                    )
                    break
                matched_acqs_idxs &= {i, j}

        # The remaining SNR tasks will be treated as smoothing
        for i, acq in enumerate(self._acqs):
            if i not in matched_acqs_idxs:
                _jobs.append(
                    JobTaskConfig(
                        task="snr",
                        folders=[acq.path],
                    ),
                )
        return _jobs

    def ingest(self, acq: DiscoveredAcquisition) -> None:
        """Ingest an acquisition into the SNR job list."""
        self._acqs.append(acq)


def generate_batch_config(path: str | Path) -> BatchConfig:
    """Generate a batch configuration object."""
    path = Path(path)

    dirs = [d for d in path.glob("*") if d.is_dir()]
    acqs = [DiscoveredAcquisition.from_path(d) for d in dirs]
    jobs = AcquisitionCollector(acqs).jobs

    return BatchConfig(
        version=BatchConfig._CURRENT_BATCHCONFIG_VERSION,       # noqa: SLF001
        hazen_version_constraint=f">={__version__}",
        description=(
            "Batch configuration file automatically generated"
            f" from {path.as_posix()}"
        ),
        jobs=jobs,
        output=path.parent / "hazen_output.json",
        report_docx=path.parent / "hazen_report.docx",
        report_template=None,
        levels=("final", "all"),
        defaults={},
        _file=path.parent / "hazen_batch_config.yml",
    )
