"""Shared engine for protocols run over one session folder.

A protocol subclasses :class:`SessionProtocol` and supplies its role
table, the vendors it covers and two hooks: :meth:`~SessionProtocol.discover`
splits a session folder into roles and listed-only series, and
:meth:`~SessionProtocol.build_steps` returns the tasks to run on those roles.
The engine detects each role's phantom orientation once, runs the steps,
normalises subtraction SNR by the measured slice width and flattens the
results into labelled rows.

SNR mode is step data: a step with ``subtract`` and ``slice_width_from`` runs
after the slice thickness steps as subtraction SNR, a step with neither runs
in the first stage as single-image SNR.
"""

from __future__ import annotations

# Type Checking
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hazenlib.types import Measurement, Result

# Python imports
import datetime
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal

# Module imports
import pydicom
from pydicom.multival import MultiValue

# Local imports
from hazenlib._version import __version__
from hazenlib.ACRObject import ACRObject
from hazenlib.orchestration import init_task
from hazenlib.orientation import describe_orientation
from hazenlib.protocols.outputs import ImageGroup, Tile
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
class Cell:
    """One value of the results table; *missing* marks a skipped or
    failed step that should have filled it.
    """

    value: Any
    unit: str = ""
    missing: bool = False


@dataclass(frozen=True)
class TableRow:
    """One measurement of a test, by column; absent columns are blank."""

    test: str
    measurement: str
    cells: dict[str, Cell]


@dataclass(frozen=True)
class Table:
    """Final measurements with one row per test and measurement."""

    columns: tuple[str, ...]
    rows: tuple[TableRow, ...]


def _measurement_label(m: Measurement) -> str:
    """Name a measurement the same way for every role.

    The subtype, with the type when it is not a plain measurement and
    the slice from the description when the subtype does not give it,
    e.g. "Normalised SNR (smoothing)", "Length difference (slice 1)".
    """
    if m.type != "measured":
        label = f"{m.type.capitalize()} {m.name}"
        label += f" ({m.subtype})" if m.subtype else ""
    else:
        label = m.subtype or m.name
        label = label[:1].upper() + label[1:]
    found = re.match(r"Slice \d+", m.description)
    if found and not label.lower().startswith("slice"):
        label += f" ({found[0].lower()})"
    return label


# Slices of the figures of the tasks that save more than one, and so
# of their montage tiles
_FIGURE_SLICES = {
    "acr_slice_position": (1, 11),
    "acr_geometric_accuracy": (1, 5),
}
# Short names for the values of a tile with several
_SHORT_LABELS = {
    "Horizontal": "H",
    "Vertical": "V",
    "Diagonal SW": "SW",
    "Diagonal SE": "SE",
}


def _tile_value(result: Result, slice_number: int | None) -> str:
    """Final measurement(s) of one figure, e.g. ``4.98mm``.

    Several values are named, e.g. ``H 190.10, V 189.80 mm``.
    """
    finals = [m for m in result.measurements if m.final]
    if slice_number is not None:
        of_slice = re.compile(rf"slice {slice_number}\b", re.IGNORECASE)
        finals = [m for m in finals if of_slice.search(_measurement_label(m))]
    # SNR by subtraction is the protocol's value when both methods ran
    finals = [m for m in finals if m.subtype == "subtraction"] or finals
    if not finals:
        return "failed"

    def number(m: Measurement) -> str:
        return f"{round(float(m.value), 2) + 0.0:.2f}"  # + 0.0 so never "-0.00"

    if len(finals) == 1:
        return f"{number(finals[0])}{finals[0].unit or ''}"
    values = []
    for m in finals:
        label = re.sub(
            r"\(?slice \d+\)?| distance",
            "",
            _measurement_label(m),
            flags=re.IGNORECASE,
        ).strip()
        values.append(f"{_SHORT_LABELS.get(label, label)} {number(m)}")
    units = {m.unit for m in finals if m.unit}
    unit = f" {units.pop()}" if len(units) == 1 else ""
    return ", ".join(values) + unit


def detect_orientation(files: list[str]) -> tuple[int, bool] | None:
    """(k, flip) correction putting a series in the standard view.

    Returns None if it cannot be determined, so each task tries again
    and reports the error as its own failure.
    """
    try:
        return ACRObject([pydicom.dcmread(f) for f in files]).orientation
    # ACRObject calls sys.exit() on unexpected slice orientations
    except (Exception, SystemExit):  # noqa: BLE001
        logger.exception("Could not determine the phantom orientation")
        return None


def _execute(
    step: Step,
    files: list[str],
    kwargs: dict[str, Any],
) -> tuple[Status, str, Result | None]:
    """Run one step, turning any error into a failed status.

    Tasks log and swallow errors in their own analysis, so a result with
    no final measurements is a failure too.
    """
    try:
        task = init_task(step.task, files, **kwargs)
        result = task.run()
        if not any(m.final for m in result.measurements):
            return "failed", "no final measurements (see the log)", result
        return "ok", "", result
    # ACRObject calls sys.exit() on unexpected slice orientations, so
    # SystemExit is caught too or it would take down the whole run.
    except (Exception, SystemExit) as err:  # noqa: BLE001
        logger.exception("%s failed on %s", step.task, step.role)
        return "failed", f"{type(err).__name__}: {err}", None


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
        orientations = self._orientations(debug=debug)
        first_stage = [s for s in self.steps if s.slice_width_from is None]
        outcomes = dict(
            zip(
                first_stage,
                self._run_stage(
                    [(s, {}) for s in first_stage],
                    orientations,
                    debug=debug,
                ),
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
            self._run_stage(snr_jobs, orientations, debug=debug),
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

    def _orientations(
        self,
        *,
        debug: bool,
    ) -> dict[str, tuple[int, bool] | None]:
        """Detect the orientation of every role a step uses, in parallel.

        Done once per role rather than once per step, as every task on a
        role would otherwise detect it from the same slice 1.
        """
        names = [
            name
            for name in self.session.roles
            if any(name in (s.role, s.subtract) for s in self.steps)
        ]
        found = wait_on_parallel_results(
            detect_orientation,
            [(list(self.session.roles[n].files),) for n in names],
            debug=debug,
        )
        for name, orientation in zip(names, found, strict=True):
            if orientation is not None:
                logger.info(
                    "%s: %s",
                    name,
                    describe_orientation(*orientation),
                )
        return dict(zip(names, found, strict=True))

    def _run_stage(
        self,
        jobs: list[tuple[Step, dict[str, Any]]],
        orientations: dict[str, tuple[int, bool] | None],
        *,
        debug: bool,
    ) -> list[StepOutcome]:
        """Run steps in parallel, skipping those with missing roles.

        Each task gets its role's orientation, and SNR by subtraction
        the subtracted role's too; None lets the task detect it.
        """
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
            if step.subtract is not None:
                kwargs = {
                    **kwargs,
                    "orientation2": orientations[step.subtract],
                }
            runnable.append(
                (step, {**kwargs, "orientation": orientations[step.role]}),
            )

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

    def column(self, role: Role) -> str:
        """Results table column for a role: its coil and plane."""
        return f"{role.coil.capitalize()} {role.plane.upper()}"

    def table(self, outcomes: list[StepOutcome]) -> Table:
        """Final measurements, one row per test and measurement.

        Each column is a coil and plane. A step that was skipped or
        failed marks its column missing on every row of its test, and
        a test with no measurements at all gets one unnamed row.
        """
        roles = {r.name: r for r in self.roles}
        columns = tuple(
            dict.fromkeys(self.column(roles[s.role]) for s in self.steps),
        )
        cells: dict[tuple[str, str], dict[str, Cell]] = {}
        missing: dict[str, list[str]] = {}
        for o in outcomes:
            column = self.column(roles[o.step.role])
            missing.setdefault(o.step.test, [])
            if o.status != "ok" or o.result is None:
                missing[o.step.test].append(column)
                continue
            for m in o.result.measurements:
                if m.final:
                    key = (o.step.test, _measurement_label(m))
                    cells.setdefault(key, {})[column] = Cell(m.value, m.unit)
            if o.slice_width is not None:
                # What subtraction SNR was normalised by; a nominal
                # width gets its own row so it stands out
                label = "Slice width used"
                if (o.slice_width_source or "").startswith("nominal"):
                    label += " (nominal)"
                cells.setdefault((o.step.test, label), {})[column] = Cell(
                    o.slice_width,
                    "mm",
                )

        rows = []
        for test, missing_columns in missing.items():
            labels = [label for t, label in cells if t == test] or [""]
            for label in labels:
                row = cells.get((test, label), {})
                row.update(
                    (c, Cell(None, missing=True)) for c in missing_columns
                )
                rows.append(TableRow(test, label, row))
        return Table(columns, tuple(rows))

    def images(
        self, outcomes: list[StepOutcome]
    ) -> dict[str, list[ImageGroup]]:
        """Report images by test, each group titled e.g. ``COR_SlicePosition``.

        Empty unless the run saved report images.
        """
        groups: dict[str, list[ImageGroup]] = {}
        if not self.report:
            return groups
        for o in outcomes:
            if o.result is None or not o.result.report_images:
                continue
            title = f"{o.step.role}_{o.result.task.removeprefix('ACR')}"
            groups.setdefault(o.step.test, []).append(
                ImageGroup(title, tuple(o.result.report_images)),
            )
        return groups

    def tiles(self, outcomes: list[StepOutcome]) -> list[Tile]:
        """Montage tiles: one per figure of every step, by table column.

        Each is captioned e.g. ``HEAD_TRA: Slice Thickness = 4.98mm``.
        A skipped or failed step keeps its tiles, captioned with its
        status and with no image unless the task still saved one.
        """
        roles = {r.name: r for r in self.roles}
        tiles = []
        for o in outcomes:
            row = self.column(roles[o.step.role]).upper().replace(" ", "_")
            test = o.step.test.replace("_", " ").title().replace("Snr", "SNR")
            for n in _FIGURE_SLICES.get(o.step.task, (None,)):
                title = test if n is None else f"{test} (slice {n})"
                value: str = o.status
                if o.status == "ok" and o.result is not None:
                    value = _tile_value(o.result, n)
                path = o.result.overlays.get(n) if o.result else None
                tiles.append(
                    Tile(row, title, f"{row}: {title} = {value}", path),
                )
        return tiles

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
