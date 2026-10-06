"""GSTT ACR Large Phantom QA protocol.

Runs every hazen measurement required by the GSTT ACR work instruction
on one session folder. Each acquisition sits in a subfolder named after
its role, see :func:`match_role` for the names accepted.

======================  ==================================================
Role                    Acquisition
======================  ==================================================
``Head_1``, ``Head_2``  No filters on: repeat pair for subtraction SNR
``Head_3``              Intensity correction / normalisation (uniformity)
``Head_4``              Distortion correction (slice and geometry tests)
``Head_5``              Philips CLASSIC reconstruction (for ghosting test)
``Body_<Plane>_1``-4    As for Head 1-4, ``<Plane>`` is Tra, Sag or Cor
======================  ==================================================

Vendor rules (from the DICOM ``Manufacturer`` unless overridden):

- Philips measures ghosting on ``Head_5``, other vendors on ``Head_1``.
- GE has no distortion correction instructions, so slice thickness, slice position
  and geometric accuracy use series 1. Body uniformity also uses series 1.

SNR is measured by subtraction (series 1 - series 2) once the
slice thickness of the matching distortion-corrected series is known,
and that measured slice width is used to normalise the SNR.

Sagittal and coronal series must already be rotated so the phantom
appears as in a transverse acquisition. This protocol does not rotate anything.
"""

from __future__ import annotations

# Type Checking
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from hazenlib.types import Measurement, Result

# Python imports
import datetime
import logging
import re
from dataclasses import asdict, dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal

# Module imports
import pydicom
from pydicom.multival import MultiValue

# Local imports
from hazenlib._version import __version__
from hazenlib.orchestration import init_task
from hazenlib.protocols.outputs import ImageGroup, TemplateCell
from hazenlib.utils import (
    get_dicom_files,
    get_manufacturer,
    get_slice_thickness,
    wait_on_parallel_results,
)

logger = logging.getLogger(__name__)

SUPPORTED_VENDORS = ("siemens", "philips", "ge")

HEAD = "head"
BODY = "body"
PLANES = ("tra", "sag", "cor")

Status = Literal["ok", "skipped", "failed"]

# Default report image folder inside a session, not an acquisition
REPORT_DIR_NAME = "hazen_report"


#########
# Roles #
#########


@dataclass(frozen=True)
class Role:
    """An acquisition role within a session, e.g. ``Body_Sag_4``."""

    coil: str
    plane: str
    number: int

    @property
    def name(self) -> str:
        """Canonical folder name for the role."""
        if self.coil == HEAD:
            return f"Head_{self.number}"
        return f"Body_{self.plane.capitalize()}_{self.number}"


ROLES: tuple[Role, ...] = (
    *(Role(HEAD, "tra", n) for n in range(1, 6)),
    *(Role(BODY, plane, n) for plane in PLANES for n in range(1, 5)),
)
ROLES_BY_NAME = {r.name: r for r in ROLES}


_PLANE_ALIASES = {
    "transverse": "tra",
    "axial": "tra",
    "tra": "tra",
    "ax": "tra",
    "sagittal": "sag",
    "sag": "sag",
    "coronal": "cor",
    "cor": "cor",
}
_PLANE = "|".join(_PLANE_ALIASES)  # longest aliases first
_SEP = r"[\W_]*"
# Coil, then the plane before or after the number, then an optional
# suffix after a separator, e.g. Body_Sag_4, BODY4_sag_rotMAT, HEAD2_rep
_ROLE_PATTERN = re.compile(
    rf"(?P<coil>head|body){_SEP}"
    rf"(?:(?P<plane_before>{_PLANE}){_SEP})?"
    rf"(?P<number>\d)"
    rf"(?:{_SEP}(?P<plane_after>{_PLANE}))?"
    r"(?:[\W_]+.*)?",
    re.IGNORECASE,
)


def match_role(folder_name: str) -> str | None:
    """Return the role a folder name refers to, or None.

    Case and separators are ignored, the plane may come before or after
    the number (``ax``/``axial`` for ``tra``) and anything after a
    trailing separator is ignored.
    """
    found = _ROLE_PATTERN.fullmatch(folder_name)
    if found is None:
        return None
    before, after = found["plane_before"], found["plane_after"]
    if before and after:
        return None
    coil = found["coil"].lower()
    plane = _PLANE_ALIASES[(before or after or "tra").lower()]
    if coil == BODY and not (before or after):
        return None
    role = Role(coil, plane, int(found["number"]))
    if coil == HEAD and plane != "tra":
        return None
    return role.name if role.name in ROLES_BY_NAME else None


@dataclass(frozen=True)
class RoleFolder:
    """A session subfolder identified as a role."""

    role: str
    path: Path
    files: tuple[str, ...]
    series_description: str
    series_number: str
    manufacturer: str
    vendor: str | None


@dataclass(frozen=True)
class Session:
    """A session folder split into role folders and ignored folders."""

    path: Path
    folders: dict[str, RoleFolder]
    ignored: tuple[Path, ...]

    @classmethod
    def from_dir(cls, path: str | Path) -> Session:
        """Identify the role subfolders of a session folder.

        Raises:
            NotADirectoryError: If *path* is not a folder.
            ValueError: If two subfolders map to the same role.

        """
        path = Path(path).resolve()
        if not path.is_dir():
            msg = f"Session folder not found: {path}"
            raise NotADirectoryError(msg)

        folders: dict[str, RoleFolder] = {}
        ignored: list[Path] = []
        for sub in sorted(
            p
            for p in path.iterdir()
            if p.is_dir() and p.name != REPORT_DIR_NAME
        ):
            role = match_role(sub.name)
            if role is None:
                ignored.append(sub)
                continue
            if role in folders:
                msg = (
                    f"Folders {folders[role].path.name!r} and {sub.name!r}"
                    f" both match role {role}. Rename or move the one"
                    " that should not be used."
                )
                raise ValueError(msg)
            files = tuple(sorted(get_dicom_files(str(sub))))
            if not files:
                logger.warning("No DICOM images in %s - ignoring", sub)
                ignored.append(sub)
                continue
            folders[role] = _read_role_folder(role, sub, files)

        if ignored:
            logger.warning(
                "Ignoring folders that do not match a role: %s",
                ", ".join(p.name for p in ignored),
            )
        ordered = {r.name: folders[r.name] for r in ROLES if r.name in folders}
        return cls(path, ordered, tuple(ignored))


def _read_role_folder(
    role: str,
    path: Path,
    files: tuple[str, ...],
) -> RoleFolder:
    dcm = pydicom.dcmread(files[0], stop_before_pixels=True)
    try:
        vendor = get_manufacturer(dcm)
    except Exception:  # noqa: BLE001 - unknown or missing manufacturer
        vendor = None
    return RoleFolder(
        role=role,
        path=path,
        files=files,
        series_description=str(dcm.get("SeriesDescription", "")),
        series_number=str(dcm.get("SeriesNumber", "")),
        manufacturer=str(dcm.get("Manufacturer", "")),
        vendor=vendor,
    )


def detect_vendor(session: Session, override: str | None = None) -> str:
    """Work out which vendor rules apply to a session.

    Args:
        session: The session to inspect.
        override: Vendor to use regardless of the DICOM headers.

    Raises:
        ValueError: If the folders disagree, the manufacturer cannot be
            read, or the vendor is not covered by the protocol.

    """
    vendors = {rf.role: rf.vendor for rf in session.folders.values()}
    found = set(vendors.values())

    if override is not None:
        override = override.lower()
        if override not in SUPPORTED_VENDORS:
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
    if vendor not in SUPPORTED_VENDORS:
        msg = (
            f"Manufacturer {vendor!r} is not covered by the GSTT ACR"
            f" protocol (supported: {', '.join(SUPPORTED_VENDORS)})"
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


def build_steps(vendor: str) -> tuple[Step, ...]:
    """Return the steps the GSTT work instruction requires for a vendor."""
    steps: list[Step] = []
    for coil, plane in (
        (HEAD, "tra"),
        (BODY, "tra"),
        (BODY, "sag"),
        (BODY, "cor"),
    ):

        def role(n: int, coil: str = coil, plane: str = plane) -> str:
            return Role(coil, plane, n).name

        geometry = role(1 if vendor == "ge" else 4)
        # For GE, the uniformity test is run on a different role
        uniformity = role(1 if vendor == "ge" and coil == BODY else 3)
        steps += [
            Step("slice_thickness", "acr_slice_thickness", geometry),
            Step("slice_position", "acr_slice_position", geometry),
            Step("geometric_accuracy", "acr_geometric_accuracy", geometry),
            Step("uniformity", "acr_uniformity", uniformity),
        ]
        if coil == HEAD:
            # Philips uses the CLASSIC filter for ghosting
            ghosting = role(5 if vendor == "philips" else 1)
            steps.append(Step("ghosting", "acr_ghosting", ghosting))
        steps.append(
            Step(
                "snr",
                "acr_snr",
                role(1),
                subtract=role(2),
                slice_width_from=geometry,
            ),
        )
    return tuple(steps)


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
    folder: RoleFolder | None,
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
    role = ROLES_BY_NAME[outcome.step.role]
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


############
# Workbook #
############

WORKBOOK_TEMPLATE = files("hazenlib") / "data" / "acr_gstt_template.xlsx"
COVER = "Hazen Output"
GEOMETRY = "Geometric accuracy"

# Hazen Output result column for each coil and plane
_COVER_COLUMNS = {
    (HEAD, "tra"): "C",
    (BODY, "tra"): "D",
    (BODY, "sag"): "E",
    (BODY, "cor"): "F",
}
# Hazen Output row for each test, with the measurement that fills it
_COVER_ROWS: tuple[tuple[str, int, Callable[[Measurement], bool]], ...] = (
    (
        "snr",
        14,
        lambda m: m.subtype == "subtraction" and m.type == "measured",
    ),
    (
        "snr",
        15,
        lambda m: m.subtype == "subtraction" and m.type == "normalised",
    ),
    ("uniformity", 16, lambda m: m.name == "Uniformity"),
    ("ghosting", 17, lambda m: m.name == "Ghosting"),
    ("slice_thickness", 23, lambda m: m.subtype == "slice width"),
    (
        "slice_position",
        24,
        lambda m: m.description.startswith("Slice 1 "),
    ),
    (
        "slice_position",
        25,
        lambda m: m.description.startswith("Slice 11 "),
    ),
)
# First "Hazen" row of each block on the geometric accuracy sheet
_GEOMETRY_ROWS = {
    (HEAD, "tra"): 12,
    (BODY, "tra"): 26,
    (BODY, "sag"): 40,
    (BODY, "cor"): 54,
}
# Offset from the first row of a block, with the measurement that fills it.
# Offset 6 is the template's own "Max error" formula.
_GEOMETRY_CELLS: tuple[tuple[int, Callable[[Measurement], bool]], ...] = (
    *(
        (i, lambda m, s=f"Slice {n} {line} distance": m.subtype == s)
        for i, (n, line) in enumerate(
            (
                (1, "Horizontal"),
                (1, "Vertical"),
                (5, "Horizontal"),
                (5, "Vertical"),
                (5, "Diagonal SW"),
                (5, "Diagonal SE"),
            ),
        )
    ),
    (7, lambda m: m.description == "Coefficient of variation"),
)
# Report image sheet for each coil and plane
_IMAGE_SHEETS = {
    (HEAD, "tra"): "Head Images",
    (BODY, "tra"): "Body Tra Images",
    (BODY, "sag"): "Body Sag Images",
    (BODY, "cor"): "Body Cor Images",
}


def _targets(step: Step) -> list[tuple[str, str, Callable[..., bool]]]:
    """Workbook cells filled by a step, with the measurement for each."""
    role = ROLES_BY_NAME[step.role]
    key = (role.coil, role.plane)
    if step.test == "geometric_accuracy":
        first = _GEOMETRY_ROWS[key]
        return [
            (GEOMETRY, f"B{first + offset}", match)
            for offset, match in _GEOMETRY_CELLS
        ]
    column = _COVER_COLUMNS[key]
    return [
        (COVER, f"{column}{row}", match)
        for test, row, match in _COVER_ROWS
        if test == step.test
    ]


############
# Protocol #
############


class ACRGSTTProtocol:
    """GSTT ACR Large Phantom QA protocol over a single session folder."""

    name = "ACR Large Phantom (GSTT)"

    def __init__(
        self,
        session_dir: str | Path,
        *,
        vendor: str | None = None,
        report: bool = True,
        report_dir: str | Path | None = None,
    ) -> None:
        """Identify the session folders and the steps to run.

        Args:
            session_dir: Folder containing the role subfolders.
            vendor: Override for the vendor rules (siemens, philips, ge).
            report: Whether tasks save report images.
            report_dir: Where report images go; defaults to
                ``<session_dir>/hazen_report``. Each role gets a subfolder.

        Raises:
            ValueError: If no role folders are found or the vendor
                cannot be determined.

        """
        self.session = Session.from_dir(session_dir)
        if not self.session.folders:
            msg = (
                f"No role folders (Head_1, Body_Sag_4, ...) found in"
                f" {self.session.path}"
            )
            raise ValueError(msg)
        self.vendor = detect_vendor(self.session, vendor)
        self.steps = build_steps(self.vendor)
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
            if step.subtract in self.session.folders:
                kwargs["subtract"] = str(
                    self.session.folders[step.subtract].path,
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
                if r is not None and r not in self.session.folders
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
                list(self.session.folders[step.role].files),
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
        if step.role in self.session.folders:
            dcm = pydicom.dcmread(
                self.session.folders[step.role].files[0],
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
        rows = []
        for o in outcomes:
            folder = self.session.folders.get(o.step.role)
            if o.result is None:
                rows.append(_row(o, folder))
                continue

            images = "; ".join(o.result.report_images)
            rows.extend(
                _row(
                    o,
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
        """Scanner and run details, read from the first role folder."""
        first = next(iter(self.session.folders.values()))
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

    def workbook_cells(
        self,
        outcomes: list[StepOutcome],
    ) -> list[TemplateCell]:
        """Values for the Hazen Output and Geometric accuracy tables.

        Cells without a value get a note saying why (skipped, failed or
        not measured), so gaps in the workbook are explained.
        """
        cells = []
        for o in outcomes:
            for sheet, ref, match in _targets(o.step):
                found = (
                    [m for m in o.result.measurements if match(m)]
                    if o.result is not None
                    else []
                )
                if found:
                    cells.append(TemplateCell(sheet, ref, found[0].value))
                    continue
                note = (
                    f"{o.step.role} {o.step.test} {o.status}: {o.detail}"
                    if o.status != "ok"
                    else f"{o.step.role} {o.step.test}: not measured"
                )
                cells.append(TemplateCell(sheet, ref, note=note))

        return cells

    def image_groups(
        self,
        outcomes: list[StepOutcome],
    ) -> dict[str, list[ImageGroup]]:
        """Report images per coil and plane sheet, one group per step.

        Every sheet is empty when report images were not generated.
        """
        sheets: dict[str, list[ImageGroup]] = {
            name: [] for name in _IMAGE_SHEETS.values()
        }
        if not self.report:
            return sheets
        for o in outcomes:
            role = ROLES_BY_NAME[o.step.role]
            images = tuple(o.result.report_images) if o.result else ()
            note = (
                f"{o.status}: {o.detail}"
                if o.status != "ok"
                else ("" if images else "no report images")
            )
            sheets[_IMAGE_SHEETS[role.coil, role.plane]].append(
                ImageGroup(f"{role.name} {o.step.test}", images, note),
            )
        return sheets

    def summary(self, outcomes: list[StepOutcome]) -> str:
        """Short human-readable summary of a run."""
        info = self.session_info()
        counts = {
            s: sum(o.status == s for o in outcomes)
            for s in ("ok", "skipped", "failed")
        }
        lines = [
            f"{info['Manufacturer']} {info['Model']}"
            f" {info['Field strength (T)']}T - {self.vendor} rules -"
            f" {len(self.session.folders)} role folders found,"
            f" {len(self.session.ignored)} ignored"
            + (
                f" ({', '.join(p.name for p in self.session.ignored)})"
                if self.session.ignored
                else ""
            ),
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


def rows_as_records(rows: list[ResultRow]) -> list[dict[str, Any]]:
    """Convert result rows to plain dictionaries for the writers."""
    return [asdict(r) for r in rows]
