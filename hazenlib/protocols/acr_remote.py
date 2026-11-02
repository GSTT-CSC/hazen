"""Remote ACR Large Phantom QA protocol.

Runs the ACR measurements of the NHSBSP breast-screening MRI QA on a
session exported by a site: one T1 acquisition per plane, transverse on
the head/neck coil and sagittal and coronal on the body and spine coils.

Series are matched on their DICOM headers, not folder names. A plane is
an ``ACR`` token followed by a plane alias (``ACR_TRA``, ``ACR Axial``,
``acr-coronal``) and only the distortion-corrected series of each plane
is analysed. See :func:`classify_series` for the rules.
"""

from __future__ import annotations

# Python imports
import logging
import re
from pathlib import Path

# Module imports
import pydicom
from pydicom.errors import InvalidDicomError

# Local imports
from hazenlib.protocols.engine import (
    PLANE_ALIASES,
    PLANE_PATTERN,
    REPORT_DIR_NAME,
    ListedKind,
    ListedSeries,
    Role,
    RoleSeries,
    Session,
    SessionProtocol,
    Step,
)
from hazenlib.utils import get_manufacturer

logger = logging.getLogger(__name__)

REMOTE_ROLES: tuple[Role, ...] = (
    Role("TRA", "head", "tra"),
    Role("SAG", "body", "sag"),
    Role("COR", "body", "cor"),
)
ROLES_BY_PLANE = {r.plane: r for r in REMOTE_ROLES}

# QA test and TASK_REGISTRY key, run on every role
REMOTE_TESTS = (
    ("snr", "acr_snr"),
    ("uniformity", "acr_uniformity"),
    ("ghosting", "acr_ghosting"),
    ("slice_thickness", "acr_slice_thickness"),
    ("slice_position", "acr_slice_position"),
    ("geometric_accuracy", "acr_geometric_accuracy"),
)

# Case-insensitive SeriesDescription substrings of visual-check series
VISUAL_CHECK = (
    "noise",
    "field_map",
    "fieldmap",
    "b0_map",
    "b0map",
    "nonsup",
    "nosup",
    "non_sup",
    "fatsup",
    "fat_sup",
    "fat sup",
)

_ACR_PLANE = re.compile(
    rf"(?<![a-z0-9])acr[\W_]*(?P<plane>{PLANE_PATTERN})(?![a-z])",
    re.IGNORECASE,
)
_ND_SUFFIX = re.compile(r"[\W_]nd(?![a-z0-9])", re.IGNORECASE)


def _plane(text: str) -> str | None:
    found = _ACR_PLANE.search(text)
    return PLANE_ALIASES[found["plane"].lower()] if found else None


def classify_series(
    series_description: str,
    protocol_name: str,
    image_type: tuple[str, ...],
    vendor: str | None,
    *,
    label: str = "",
) -> tuple[Role | None, ListedKind | None]:
    """Work out what one series is from its headers.

    In order: a visual-check marker in the description, then the plane
    from the description (ProtocolName as fallback), then for Siemens
    ``ImageType`` DIS2D (the plane's role) or ND (uncorrected). Other
    vendors have no such marker, so every ACR series of a plane is a
    candidate for its role.

    Args:
        series_description: DICOM SeriesDescription.
        protocol_name: DICOM ProtocolName.
        image_type: DICOM ImageType values.
        vendor: Manufacturer as returned by ``get_manufacturer``.
        label: Names the series in warnings.

    Returns:
        The role and None, or None and the kind of listed series.

    """
    label = label or repr(series_description)
    lowered = series_description.lower()
    if any(token in lowered for token in VISUAL_CHECK):
        return None, "visual_check"

    from_description = _plane(series_description)
    from_protocol = _plane(protocol_name)
    if (
        from_description
        and from_protocol
        and from_description != from_protocol
    ):
        logger.warning(
            "%s: SeriesDescription names plane %s but ProtocolName %r"
            " names %s - using %s",
            label,
            from_description,
            protocol_name,
            from_protocol,
            from_description,
        )
    plane = from_description or from_protocol
    if plane is None:
        logger.warning(
            "%s: no ACR plane (e.g. ACR_TRA) in SeriesDescription or"
            " ProtocolName - listed as unknown",
            label,
        )
        return None, "unknown"
    role = ROLES_BY_PLANE[plane]

    if vendor != "siemens":
        logger.warning(
            "%s: distortion correction status cannot be read from %s"
            " headers - using it for %s",
            label,
            vendor or "unrecognised manufacturer",
            role.name,
        )
        return role, None

    markers = set(image_type)
    has_suffix = _ND_SUFFIX.search(series_description) is not None
    if "DIS2D" in markers:
        if has_suffix:
            logger.warning(
                "%s: ImageType says DIS2D but the description has an _ND"
                " suffix - using it for %s",
                label,
                role.name,
            )
        return role, None
    if "ND" in markers:
        if not has_suffix:
            logger.warning(
                "%s: ImageType says ND but the description has no _ND"
                " suffix - listed as uncorrected",
                label,
            )
        return None, "uncorrected"
    logger.warning(
        "%s: ACR series with neither DIS2D nor ND in ImageType - listed"
        " as unknown",
        label,
    )
    return None, "unknown"


def _vendor(dcm: pydicom.Dataset) -> str | None:
    try:
        return get_manufacturer(dcm)
    except Exception:  # noqa: BLE001 - unknown or missing manufacturer
        return None


def _read_series(path: Path) -> list[tuple[pydicom.Dataset, list[str]]]:
    """Group the DICOM files under *path* by SeriesInstanceUID.

    Report folders, non-DICOM files and DICOM files without a series
    (e.g. DICOMDIR) are skipped.

    Returns:
        The first file's header and the files of each series.

    """
    series: dict[str, tuple[pydicom.Dataset, list[str]]] = {}
    for file in sorted(path.rglob("*")):
        if (
            not file.is_file()
            or REPORT_DIR_NAME in file.relative_to(path).parts
        ):
            continue
        try:
            dcm = pydicom.dcmread(file, stop_before_pixels=True)
        except InvalidDicomError:
            continue
        uid = dcm.get("SeriesInstanceUID")
        if uid is None:
            continue
        series.setdefault(str(uid), (dcm, []))[1].append(str(file))
    return list(series.values())


def _check_one_scanner(headers: list[pydicom.Dataset]) -> None:
    """Refuse sessions from two scanners, warn about several dates.

    Raises:
        ValueError: If the series come from more than one scanner.

    """
    scanners = {
        str(dcm.get("DeviceSerialNumber") or dcm.get("StationName") or "")
        for dcm in headers
    } - {""}
    if len(scanners) > 1:
        msg = (
            "The session holds series from more than one scanner"
            f" ({', '.join(sorted(scanners))}). Give hazen one scanner's"
            " export at a time."
        )
        raise ValueError(msg)
    dates = {str(dcm.get("StudyDate", "")) for dcm in headers} - {""}
    if len(dates) > 1:
        logger.warning(
            "The session holds more than one study date (%s)",
            ", ".join(sorted(dates)),
        )


class ACRRemoteProtocol(SessionProtocol):
    """Remote ACR QA protocol over one exported session folder."""

    name = "ACR Large Phantom (Remote)"
    roles = REMOTE_ROLES
    supported_vendors = ("siemens", "philips", "ge")

    @classmethod
    def build_steps(cls, vendor: str) -> tuple[Step, ...]:
        """Return every Remote test on every role, for any vendor.

        SNR steps have no ``subtract`` or ``slice_width_from``, so the
        engine runs them as single-image SNR in the first stage.
        """
        return tuple(
            Step(test, task, role.name)
            for role in cls.roles
            for test, task in REMOTE_TESTS
        )

    @classmethod
    def discover(cls, path: str | Path) -> Session:
        """Group an export into series and match each on its headers.

        Raises:
            NotADirectoryError: If *path* is not a folder.
            ValueError: If two series match one role, or the series
                come from more than one scanner.

        """
        path = Path(path).resolve()
        if not path.is_dir():
            msg = f"Session folder not found: {path}"
            raise NotADirectoryError(msg)

        series = _read_series(path)
        _check_one_scanner([dcm for dcm, _ in series])

        found: dict[str, tuple[RoleSeries, str]] = {}
        listed: list[ListedSeries] = []
        uncorrected_planes: set[str] = set()
        for dcm, files in series:
            folder = Path(files[0]).parent
            description = str(dcm.get("SeriesDescription", ""))
            number = str(dcm.get("SeriesNumber", ""))
            protocol_name = str(dcm.get("ProtocolName", ""))
            label = (
                f"series {number} {description!r} in"
                f" {folder.relative_to(path).as_posix()}"
            )
            vendor = _vendor(dcm)
            role, kind = classify_series(
                description,
                protocol_name,
                tuple(str(v) for v in dcm.get("ImageType", ())),
                vendor,
                label=label,
            )
            if role is None:
                listed.append(
                    ListedSeries(
                        kind or "unknown", folder, description, number
                    ),
                )
                if kind == "uncorrected":
                    uncorrected_planes.add(
                        _plane(description) or _plane(protocol_name) or "",
                    )
                continue
            if role.name in found:
                msg = (
                    f"{found[role.name][1]} and {label} both match role"
                    f" {role.name}. Move the one that should not be used"
                    " out of the session folder."
                )
                raise ValueError(msg)
            found[role.name] = (
                RoleSeries(
                    role=role,
                    path=folder,
                    files=tuple(files),
                    series_description=description,
                    series_number=number,
                    manufacturer=str(dcm.get("Manufacturer", "")),
                    vendor=vendor,
                ),
                label,
            )

        for role in cls.roles:
            if role.plane in uncorrected_planes and role.name not in found:
                logger.warning(
                    "Only an uncorrected series for %s, so its tests are"
                    " skipped: no distortion-corrected series for %s."
                    " Reconstruct it again with distortion correction.",
                    role.name,
                    role.name,
                )
        roles = {
            r.name: found[r.name][0] for r in cls.roles if r.name in found
        }
        return Session(path, roles, tuple(listed))
