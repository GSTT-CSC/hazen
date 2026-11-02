"""Shared engine for protocols run over one session folder.

A protocol subclasses :class:`SessionProtocol` and supplies its role
table, the vendors it covers and two hooks: :meth:`~SessionProtocol.discover`
splits a session folder into roles and listed-only series, and
:meth:`~SessionProtocol.build_steps` returns the tasks to run on those roles.
The engine runs the steps, normalises subtraction SNR by the measured slice
width and flattens the results into labelled rows.

SNR mode is step data: a step with ``subtract`` and ``slice_width_from`` runs
after the slice thickness steps as subtraction SNR, a step with neither runs
in the first stage as single-image SNR.
"""

from __future__ import annotations

# Type Checking
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hazenlib.types import Result

# Python imports
import datetime
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal

# Module imports
import pydicom
from pydicom.multival import MultiValue

# Local imports
from hazenlib._version import __version__
from hazenlib.orchestration import init_task
from hazenlib.utils import get_slice_thickness, wait_on_parallel_results

logger = logging.getLogger(__name__)

Status = Literal["ok", "skipped", "failed"]
ListedKind = Literal["ignored", "uncorrected", "visual_check", "unknown"]

# Default report image folder inside a session, not an acquisition
REPORT_DIR_NAME = "hazen_report"

PLANE_ALIASES = {
    "transverse": "tra",
    "axial": "tra",
    "tra": "tra",
    "ax": "tra",
    "sagittal": "sag",
    "sag": "sag",
    "coronal": "cor",
    "cor": "cor",
}
# Regex alternation of the aliases, longest first
PLANE_PATTERN = "|".join(PLANE_ALIASES)


############
# Sessions #
############


@dataclass(frozen=True)
class Role:
    """The part a series plays in a protocol, e.g. ``Body_Sag_4``."""

    name: str
    coil: str
    plane: str


@dataclass(frozen=True)
class RoleSeries:
    """The images found for a role."""

    role: Role
    path: Path
    files: tuple[str, ...]
    series_description: str
    series_number: str
    manufacturer: str
    vendor: str | None


@dataclass(frozen=True)
class ListedSeries:
    """Something in a session that is reported but not analysed."""

    kind: ListedKind
    path: Path
    series_description: str = ""
    series_number: str = ""


@dataclass(frozen=True)
class Session:
    """A session split into roles and listed-only series."""

    path: Path
    roles: dict[str, RoleSeries]
    listed: tuple[ListedSeries, ...]


def detect_vendor(
    session: Session,
    supported: tuple[str, ...],
    override: str | None = None,
) -> str:
    """Work out which vendor rules apply to a session.

    Args:
        session: The session to inspect.
        supported: Vendors the protocol covers.
        override: Vendor to use regardless of the DICOM headers.

    Raises:
        ValueError: If the roles disagree, the manufacturer cannot be
            read, or the vendor is not covered by the protocol.

    """
    vendors = {rs.role.name: rs.vendor for rs in session.roles.values()}
    found = set(vendors.values())

    if override is not None:
        override = override.lower()
        if override not in supported:
            msg = f"Unsupported vendor {override!r}"
            raise ValueError(msg)
        if found - {override}:
            logger.warning(
                "Using %s rules as requested although the DICOM headers"
                " report %s",
                override,
                vendors,
            )
        return override

    if len(found) != 1 or None in found:
        msg = (
            "Could not determine a single scanner manufacturer from the"
            f" session folders: {vendors}. Use --vendor to set it."
        )
        raise ValueError(msg)

    vendor = found.pop()
    if vendor not in supported:
        msg = (
            f"Manufacturer {vendor!r} is not covered by the protocol"
            f" (supported: {', '.join(supported)})"
        )
        raise ValueError(msg)
    return vendor


#########
# Steps #
#########


@dataclass(frozen=True)
class Step:
    """A single task run on a role.

    Attributes:
        test: Short name of the QA test, e.g. ``"uniformity"``.
        task: Key in ``TASK_REGISTRY``.
        role: Role whose images the task is run on.
        subtract: Role subtracted for SNR by subtraction.
        slice_width_from: Role whose measured slice thickness is used
            to normalise SNR.

    """

    test: str
    task: str
    role: str
    subtract: str | None = None
    slice_width_from: str | None = None


############
# Outcomes #
############


@dataclass
class StepOutcome:
    """What happened when a step was run."""

    step: Step
    status: Status
    detail: str = ""
    result: Result | None = None
    slice_width: float | None = None
    slice_width_source: str | None = None


@dataclass(frozen=True)
class ResultRow:
    """One labelled row of the results table."""

    coil: str
    plane: str
    role: str
    test: str
    measurement: str
    subtype: str
    type: str
    description: str
    value: Any
    unit: str
    visibility: str
    status: Status
    detail: str
    series_description: str
    series_number: str
    folder: str
    report_images: str


def _row(
    outcome: StepOutcome,
    role: Role,
    folder: RoleSeries | None,
    *,
    measurement: str = "",
    subtype: str = "",
    type_: str = "",
    description: str = "",
    value: Any = None,
    unit: str = "",
    visibility: str = "",
    report_images: str = "",
    detail: str | None = None,
) -> ResultRow:
    """Build a result row labelled with the outcome's role and status."""
    return ResultRow(
        coil=role.coil,
        plane=role.plane,
        role=role.name,
        test=outcome.step.test,
        measurement=measurement,
        subtype=subtype,
        type=type_,
        description=description,
        value=value,
        unit=unit,
        visibility=visibility,
        status=outcome.status,
        detail=outcome.detail if detail is None else detail,
        series_description=folder.series_description if folder else "",
        series_number=folder.series_number if folder else "",
        folder=folder.path.as_posix() if folder else "",
        report_images=report_images,
    )


def _execute(
    step: Step,
    files: list[str],
    kwargs: dict[str, Any],
) -> tuple[Status, str, Result | None]:
    """Run one step, turning any error into a failed status."""
    try:
        task = init_task(step.task, files, **kwargs)
        return "ok", "", task.run()
    # ACRObject calls sys.exit() on unexpected slice orientations, so
    # SystemExit is caught too or it would take down the whole run.
    except (Exception, SystemExit) as err:  # noqa: BLE001
        logger.exception("%s failed on %s", step.task, step.role)
        return "failed", f"{type(err).__name__}: {err}", None


def rows_as_records(rows: list[ResultRow]) -> list[dict[str, Any]]:
    """Convert result rows to plain dictionaries for the writers."""
    return [asdict(r) for r in rows]


############
# Protocol #
############


class SessionProtocol:
    """Run a protocol's steps over one session folder.

    Subclasses supply ``name``, ``roles``, ``supported_vendors``,
    :meth:`discover` and :meth:`build_steps`.
    """

    name: ClassVar[str]
    roles: ClassVar[tuple[Role, ...]]
    supported_vendors: ClassVar[tuple[str, ...]]

    @classmethod
    def discover(cls, path: str | Path) -> Session:
        """Split a session folder into roles and listed series.

        Raises:
            NotADirectoryError: If *path* is not a folder.
            ValueError: If two candidates match one role.

        """
        raise NotImplementedError

    @classmethod
    def build_steps(cls, vendor: str) -> tuple[Step, ...]:
        """Return the steps the protocol requires for a vendor."""
        raise NotImplementedError

    def __init__(
        self,
        session_dir: str | Path,
        *,
        vendor: str | None = None,
        report: bool = True,
        report_dir: str | Path | None = None,
    ) -> None:
        """Identify the session roles and the steps to run.

        Args:
            session_dir: Folder holding the session.
            vendor: Override for the vendor rules.
            report: Whether tasks save report images.
            report_dir: Where report images go; defaults to
                ``<session_dir>/hazen_report``. Each role gets a subfolder.

        Raises:
            ValueError: If no roles are found or the vendor cannot be
                determined.

        """
        self.session = self.discover(session_dir)
        if not self.session.roles:
            examples = ", ".join(r.name for r in self.roles[:2])
            msg = (
                f"No role folders or series ({examples}, ...) found in"
                f" {self.session.path}"
            )
            raise ValueError(msg)
        self.vendor = detect_vendor(
            self.session,
            self.supported_vendors,
            vendor,
        )
        self.steps = self.build_steps(self.vendor)
        self.report = report
        self.report_dir = (
            Path(report_dir)
            if report_dir is not None
            else self.session.path / REPORT_DIR_NAME
        )

    def run(self, *, debug: bool = False) -> list[StepOutcome]:
        """Run every step, SNR after the slice thickness as it depends on it.

        Args:
            debug: Run sequentially instead of in parallel.

        Returns:
            One outcome per step, in step order.

        """
        first_stage = [s for s in self.steps if s.slice_width_from is None]
        outcomes = dict(
            zip(
                first_stage,
                self._run_stage([(s, {}) for s in first_stage], debug=debug),
                strict=True,
            ),
        )

        thickness = {
            o.step.role: o
            for o in outcomes.values()
            if o.step.task == "acr_slice_thickness"
        }
        snr_steps = [s for s in self.steps if s.slice_width_from is not None]
        snr_jobs = []
        widths = {}
        for step in snr_steps:
            width, measured, source = self._slice_width(step, thickness)
            widths[step] = (width, source)
            kwargs: dict[str, Any] = {
                "measured_slice_width": width if measured else None,
            }
            if step.subtract in self.session.roles:
                kwargs["subtract"] = str(
                    self.session.roles[step.subtract].path,
                )
            snr_jobs.append((step, kwargs))

        for step, outcome in zip(
            snr_steps,
            self._run_stage(snr_jobs, debug=debug),
            strict=True,
        ):
            if outcome.status == "ok":
                outcome.slice_width, outcome.slice_width_source = widths[step]
                outcome.detail = (
                    f"slice width {outcome.slice_width} mm from"
                    f" {outcome.slice_width_source}"
                )
            outcomes[step] = outcome

        return [outcomes[s] for s in self.steps]

    def _run_stage(
        self,
        jobs: list[tuple[Step, dict[str, Any]]],
        *,
        debug: bool,
    ) -> list[StepOutcome]:
        """Run steps in parallel, skipping those with missing roles."""
        outcomes: dict[Step, StepOutcome] = {}
        runnable = []
        for step, kwargs in jobs:
            missing = [
                r
                for r in (step.role, step.subtract)
                if r is not None and r not in self.session.roles
            ]
            if missing:
                outcomes[step] = StepOutcome(
                    step,
                    "skipped",
                    f"missing {', '.join(missing)}",
                )
                continue
            runnable.append((step, kwargs))

        args = [
            (
                step,
                list(self.session.roles[step.role].files),
                {
                    "report": self.report,
                    "report_dir": str(self.report_dir / step.role),
                    **kwargs,
                },
            )
            for step, kwargs in runnable
        ]
        results = wait_on_parallel_results(_execute, args, debug=debug)
        for (step, _), (status, detail, result) in zip(
            runnable,
            results,
            strict=True,
        ):
            outcomes[step] = StepOutcome(step, status, detail, result)
        return [outcomes[step] for step, _ in jobs]

    def _slice_width(
        self,
        step: Step,
        thickness: dict[str, StepOutcome],
    ) -> tuple[float | None, bool, str]:
        """Slice width for SNR normalisation.

        Returns:
            The width, whether it was measured, and its source.

        """
        from_role = step.slice_width_from or ""
        source = thickness.get(from_role)
        if source is not None and source.result is not None:
            widths = source.result.get_measurement(
                name="SliceWidth",
                subtype="slice width",
            )
            if widths:
                return float(widths[0].value), True, from_role

        reason = (
            f"{from_role} {source.status}"
            if source is not None
            else f"no slice thickness step for {from_role}"
        )
        nominal = None
        if step.role in self.session.roles:
            dcm = pydicom.dcmread(
                self.session.roles[step.role].files[0],
                stop_before_pixels=True,
            )
            nominal = float(get_slice_thickness(dcm))
            logger.warning(
                "SNR on %s uses the nominal slice thickness (%s mm)"
                " because %s",
                step.role,
                nominal,
                reason,
            )
        return nominal, False, f"nominal (DICOM) - {reason}"

    ###########
    # Outputs #
    ###########

    def rows(self, outcomes: list[StepOutcome]) -> list[ResultRow]:
        """Flatten outcomes into labelled rows, one per measurement."""
        roles = {r.name: r for r in self.roles}
        rows = []
        for o in outcomes:
            role = roles[o.step.role]
            folder = self.session.roles.get(o.step.role)
            if o.result is None:
                rows.append(_row(o, role, folder))
                continue

            images = "; ".join(o.result.report_images)
            rows.extend(
                _row(
                    o,
                    role,
                    folder,
                    measurement=m.name,
                    subtype=m.subtype,
                    type_=m.type,
                    description=m.description,
                    value=m.value,
                    unit=m.unit,
                    visibility=m.visibility,
                    report_images=images,
                )
                for m in o.result.measurements
            )
            if o.slice_width is not None:
                source = o.slice_width_source or ""
                rows.append(
                    _row(
                        o,
                        role,
                        folder,
                        measurement="SliceWidth",
                        subtype="slice width used for SNR normalisation",
                        type_=(
                            "nominal"
                            if source.startswith("nominal")
                            else "measured"
                        ),
                        value=o.slice_width,
                        unit="mm",
                        visibility="intermediate",
                        detail=f"source: {source}",
                    ),
                )
        return rows

    def session_info(self) -> dict[str, Any]:
        """Scanner and run details, read from the first role."""
        first = next(iter(self.session.roles.values()))
        dcm = pydicom.dcmread(first.files[0], stop_before_pixels=True)
        software = dcm.get("SoftwareVersions", "")
        if isinstance(software, MultiValue):
            software = " / ".join(str(s) for s in software)
        field_strength = dcm.get("MagneticFieldStrength")
        return {
            "Session folder": self.session.path.as_posix(),
            "Institution": str(dcm.get("InstitutionName", "")),
            "Station name": str(dcm.get("StationName", "")),
            "Manufacturer": str(dcm.get("Manufacturer", "")),
            "Model": str(dcm.get("ManufacturerModelName", "")),
            "Field strength (T)": (
                float(field_strength) if field_strength is not None else None
            ),
            "Software version": str(software),
            "Study date": str(dcm.get("StudyDate", "")),
            "Vendor rules": self.vendor,
            "Hazen version": __version__,
            "Run at": datetime.datetime.now(tz=datetime.UTC).isoformat(
                timespec="seconds",
            ),
        }

    def summary(self, outcomes: list[StepOutcome]) -> str:
        """Short human-readable summary of a run."""
        info = self.session_info()
        counts = {
            s: sum(o.status == s for o in outcomes)
            for s in ("ok", "skipped", "failed")
        }
        listed = []
        for kind in ("uncorrected", "visual_check", "unknown", "ignored"):
            of_kind = [ls for ls in self.session.listed if ls.kind == kind]
            if not of_kind:
                continue
            text = f"{len(of_kind)} {kind.replace('_', ' ')}"
            # Name the series that may need attention
            if kind in ("unknown", "ignored"):
                names = (
                    ls.series_description or ls.path.name for ls in of_kind
                )
                text += f" ({', '.join(names)})"
            listed.append(text)
        lines = [
            f"{info['Manufacturer']} {info['Model']}"
            f" {info['Field strength (T)']}T - {self.vendor} rules -"
            f" {len(self.session.roles)} roles found"
            + (f" · listed: {', '.join(listed)}" if listed else ""),
            f"ok {counts['ok']} · skipped {counts['skipped']}"
            f" · failed {counts['failed']}",
        ]
        lines += [
            f"  {o.status.upper():7} {o.step.role:11} {o.step.test}:"
            f" {o.detail}"
            for o in outcomes
            if o.status != "ok"
        ]
        return "\n".join(lines)
