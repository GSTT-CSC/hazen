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

Series rotated by a multiple of 90 degrees or mirrored, as sagittal and
coronal series often are, need no rotating first: the orientation of each
role is detected once from slice 1 and every task on it is corrected.
"""

from __future__ import annotations

# Python imports
import logging
import re
from pathlib import Path

# Module imports
import pydicom

# Local imports
from hazenlib.protocols.engine import (
    PLANE_ALIASES,
    PLANE_PATTERN,
    REPORT_DIR_NAME,
    ListedSeries,
    Role,
    RoleSeries,
    Session,
    SessionProtocol,
    Step,
)
from hazenlib.utils import get_dicom_files, get_manufacturer

logger = logging.getLogger(__name__)

HEAD = "head"
BODY = "body"
PLANES = ("tra", "sag", "cor")


#########
# Roles #
#########


def _role_name(coil: str, plane: str, number: int) -> str:
    """Canonical folder name for a GSTT role."""
    if coil == HEAD:
        return f"Head_{number}"
    return f"Body_{plane.capitalize()}_{number}"


ROLES: tuple[Role, ...] = (
    *(Role(_role_name(HEAD, "tra", n), HEAD, "tra") for n in range(1, 6)),
    *(
        Role(_role_name(BODY, plane, n), BODY, plane)
        for plane in PLANES
        for n in range(1, 5)
    ),
)
ROLES_BY_NAME = {r.name: r for r in ROLES}

_SEP = r"[\W_]*"
# Coil, then the plane before or after the number, then an optional
# suffix after a separator, e.g. Body_Sag_4, BODY4_sag_rotMAT, HEAD2_rep
_ROLE_PATTERN = re.compile(
    rf"(?P<coil>head|body){_SEP}"
    rf"(?:(?P<plane_before>{PLANE_PATTERN}){_SEP})?"
    rf"(?P<number>\d)"
    rf"(?:{_SEP}(?P<plane_after>{PLANE_PATTERN}))?"
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
    plane = PLANE_ALIASES[(before or after or "tra").lower()]
    if coil == BODY and not (before or after):
        return None
    if coil == HEAD and plane != "tra":
        return None
    name = _role_name(coil, plane, int(found["number"]))
    return name if name in ROLES_BY_NAME else None


def _read_role_series(
    role: Role,
    path: Path,
    files: tuple[str, ...],
) -> RoleSeries:
    dcm = pydicom.dcmread(files[0], stop_before_pixels=True)
    try:
        vendor = get_manufacturer(dcm)
    except Exception:  # noqa: BLE001 - unknown or missing manufacturer
        vendor = None
    return RoleSeries(
        role=role,
        path=path,
        files=files,
        series_description=str(dcm.get("SeriesDescription", "")),
        series_number=str(dcm.get("SeriesNumber", "")),
        manufacturer=str(dcm.get("Manufacturer", "")),
        vendor=vendor,
    )


############
# Protocol #
############


class ACRGSTTProtocol(SessionProtocol):
    """GSTT ACR Large Phantom QA protocol over a single session folder."""

    name = "ACR Large Phantom (GSTT)"
    roles = ROLES
    supported_vendors = ("siemens", "philips", "ge")

    @classmethod
    def discover(cls, path: str | Path) -> Session:
        """Identify the role subfolders of a session folder.

        Raises:
            NotADirectoryError: If *path* is not a folder.
            ValueError: If two subfolders map to the same role.

        """
        path = Path(path).resolve()
        if not path.is_dir():
            msg = f"Session folder not found: {path}"
            raise NotADirectoryError(msg)

        folders: dict[str, RoleSeries] = {}
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
            folders[role] = _read_role_series(
                ROLES_BY_NAME[role],
                sub,
                files,
            )

        if ignored:
            logger.warning(
                "Ignoring folders that do not match a role: %s",
                ", ".join(p.name for p in ignored),
            )
        ordered = {
            r.name: folders[r.name] for r in cls.roles if r.name in folders
        }
        listed = tuple(ListedSeries("ignored", p) for p in ignored)
        return Session(path, ordered, listed)

    @classmethod
    def build_steps(cls, vendor: str) -> tuple[Step, ...]:
        """Return the steps the GSTT work instruction requires for a vendor."""
        steps: list[Step] = []
        for coil, plane in (
            (HEAD, "tra"),
            (BODY, "tra"),
            (BODY, "sag"),
            (BODY, "cor"),
        ):

            def role(n: int, coil: str = coil, plane: str = plane) -> str:
                return _role_name(coil, plane, n)

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
