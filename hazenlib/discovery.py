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
import pydicom

# Local imports
from hazenlib.ACRObject import ACRObject
from hazenlib.orchestration import JobTaskConfig

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

    @property
    def type(self) -> str:
        """Return the acquisition task type."""

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
    def _get_receiver_coil(dcm: pydicom.Dataset) -> str:
        pass

    @staticmethod
    def _get_te(dcm: pydicom.Dataset) -> int | float:
        pass

    @staticmethod
    def _get_tr(dcm: pydicom.Dataset) -> int | float:
        pass

    @staticmethod
    def _get_sequence_name(dcm: pydicom.Dataset) -> str:
        pass

    @staticmethod
    def _get_acquisition_matrix(dcm: pydicom.Dataset) -> tuple[int, ...]:
        pass

    @staticmethod
    def _get_orientation(dcm: pydicom.Dataset) -> Orientation:
        pass

    @staticmethod
    def _get_acquisition_time(dcm: pydicom.Dataset) -> tuple[int, ...]:
        pass


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
                folders=[self.t1, self.t2, self.sl],
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
        if acq1 is None or acq2 > acq1:
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
                            override={
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
