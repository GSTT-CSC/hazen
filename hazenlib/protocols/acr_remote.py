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

# Local imports
from hazenlib.protocols.engine import (
    PLANE_ALIASES,
    PLANE_PATTERN,
    ListedKind,
    Role,
)

logger = logging.getLogger(__name__)

REMOTE_ROLES: tuple[Role, ...] = (
    Role("TRA", "head", "tra"),
    Role("SAG", "body", "sag"),
    Role("COR", "body", "cor"),
)
ROLES_BY_PLANE = {r.plane: r for r in REMOTE_ROLES}

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
            vendor,
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
