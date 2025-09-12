"""Types used for hazenlib."""

from __future__ import annotations

# Typing imports
from typing import TYPE_CHECKING

from hazenlib.logger import logger

if TYPE_CHECKING:
    import pydicom
    from numpy.typing import NDArray

# Python imports
import functools
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, ParamSpec, get_args

# Module imports
import numpy as np
import scipy as sp

# Local imports
from hazenlib.constants import MEASUREMENT_NAMES, MEASUREMENT_TYPES
from hazenlib.exceptions import (InvalidMeasurementNameError,
                                 InvalidMeasurementTypeError)
from hazenlib.utils import get_pixel_size

#########################################
# ParamSpec for public methods/function #
#########################################

P_HazenTask = ParamSpec("P_HazenTask")

################
# Base Classes #
################

class JsonSerializableMixin:
    """Mix-in that supplies a shallow dict and json representation."""

    def to_dict(self) -> dict[str, Any]:
        """Return a shallow dictionary representation of the instance."""
        try:
            data = asdict(self)  # type: ignore[call-overload]

        except TypeError as err:
            logger.debug(
                "JsonSerializableMixin should only really be inhereted from by"
                " dataclasses but got  %s which raised error %s. "
                "Falling back to using __dict__",
                type(self),
                err,
            )
            data = dict(self.__dict__)
        return {k: v for k, v in data.items() if not k.startswith("_")}

    def to_json(
        self,
        *,
        indent: int | None = 2,
        sort_keys: bool = False,
        ensure_ascii: bool = True,
        **extra_kwargs: Any,    # noqa: ANN401
    ) -> str:
        """Serialize JSON with common formatting options.

        Args:
            indent : Number of spaces for pretty printing. None for compact output.
                     Defaults to 2.
            sort_keys : Alphabetically sort object keys. Default is False.
            ensure_ascii : Escape non-ASCII characters. Default is True.
            **extra_kwargs : Addition parameters passed directly to json.dumps().
                    See Python's json module documentation for available options.

        Notes:
            This method intentionally exposes only the most commonly used
            parameters with specific defaults. Advanced use cases can leverage
            extra_kwargs for parameters like `cls` (custom encoder) or `default`
            (fallback serializer).

        """
        payload = self.to_dict()
        return json.dumps(
            payload,
            indent=indent,
            sort_keys=sort_keys,
            ensure_ascii=ensure_ascii,
            **extra_kwargs,
        )

####################################################
# The canonical result that every task must return #
####################################################

@dataclass(frozen=True, slots=True)
class Measurement(JsonSerializableMixin):
    """Canonical result each measurment must have."""

    name: MEASUREMENT_NAMES
    value: float | NDArray[np.float64]
    type: MEASUREMENT_TYPES = "measured"
    subtype: str = ""
    description: str = ""
    unit: str = ""

    def __post_init__(self) -> None:
        """Validate the measurement inputs."""
        if self.name not in get_args(MEASUREMENT_NAMES):
            raise InvalidMeasurementNameError(self.name)

        if self.type not in get_args(MEASUREMENT_TYPES):
            raise InvalidMeasurementTypeError(self.type)

@dataclass(slots=True)
class Metadata(JsonSerializableMixin):
    """Canonical dictionary for result metadata."""

    files: Sequence[str] | None = None
    slice_position: Sequence[float] | None = None
    plate: int | None = None
    relaxation_type: str | None = None
    institution_name: str | None = None
    manufacturer: str | None = None
    model: str | None = None
    date: str | None = None
    manufacturers_times: Sequence[str] | None = None
    calc_times: Sequence[str] | None = None
    frac_time_difference: Sequence[str] | None = None


@dataclass
class Result(JsonSerializableMixin):
    """Canonical result any Task.run() must return."""

    task: str
    desc: str = ""
    files: str | Sequence[str] | None = None
    desc: str = ""

    def __post_init__(self) -> None:
        """Initialize the measurements, report_images and metadata."""
        self._measurements: list[Measurement] = []
        self._report_images: list[str] = []
        self.metadata = Metadata()

    @property
    def measurements(self) -> tuple[Measurement, ...]:
        """Tuple of result measurements."""
        return tuple(self._measurements)

    @property
    def report_images(self) -> tuple[str, ...]:
        """Tuple of report image locations."""
        return tuple(str(p) for p in self._report_images)


    def add_measurement(self, measurement: Measurement) -> None:
        """Add a measurement to the results."""
        self._measurements.append(measurement)


    def add_report_image(self, image_path: str | Sequence[str]) -> None:
        """Add a report image location to the report_images."""
        if isinstance(image_path, Sequence) and not isinstance(image_path, str):
            paths = image_path
        else:
            paths = [image_path]
        self._report_images += paths


    def get_measurement(
            self,
            name: str | None = None,
            measurement_type: str | None = None,
            subtype: str | None = None,
            description: str | None = None,
            unit: str | None = None,
    ) -> list[Measurement]:
        """Get the measurement(s) that match fields."""
        return [
            m
            for m in self.measurements
            if (
                    m.name == name
                    and (measurement_type is None or m.type == measurement_type)
                    and (subtype is None or m.subtype == subtype)
                    and (description is None or m.description in description)
                    and (unit is None or m.unit == unit)
            )
        ]


    def to_dict(self) -> dict[str, Any]:
        """Return dict."""
        # JsonSerializableMixin  doesn't include properties
        base = super().to_dict()

        # Add properties to dict.
        base["measurements"] = [m.to_dict() for m in self.measurements]
        base["report_images"] = list(self.report_images)
        base["metadata"]  = self.metadata.to_dict()

        return base


#############################
### Task Specific Classes ###
#############################


########
# LCOD #
########



@dataclass
class LowContrastObject:
    """Class for the Low Contrast Present within each spoke."""

    x: int
    y: int
    diameter: float
    value: float | None = None
    detected: bool = False


@dataclass
class Spoke:
    """Spoke radiating from the center to the edge of the LCOD disk.

    Also contains the LowContrastObjects along the spoke.
    """

    cx: float
    cy: float
    theta: float        # Degrees

    diameter: float

    # Distance from the center (mm)
    dist: tuple[float] = (12.5, 25, 38.0)

    # Spoke length
    length: float = 44.0

    passed: bool = False

    objects: Sequence[LowContrastObject] = field(init=False)

    def __post_init__(self) -> None:
        """Initialise the objects within the spoke."""
        # y increases as the you down the image
        # hence the minus sign is so that theta = 0
        # corresponds to the object being visually
        # above the center rather than below.
        logger.debug("Creating objects along spoke angle: %f", self.theta)
        self.objects = tuple(
            LowContrastObject(
                x=self.cx + d * np.sin(np.deg2rad(self.theta)),
                y=self.cy - d * np.cos(np.deg2rad(self.theta)),
                diameter=self.diameter,
            )
            for d in self.dist
        )


    def __len__(self) -> int:
        """Objects within the spoke."""
        return len(self.objects)

    def __iter__(self) -> LowContrastObject:
        """Iterate over the low contrast objects."""
        return iter(self.objects)

    def profile(
        self,
        dcm: pydicom.FileDataset,
        size: int = 128,
        offset: tuple[float] = (0.0, 0.0),
        *,
        return_object_mask: bool = False,
        return_coords: bool = False,
    ) -> tuple[np.ndarray]:
        """Return the 1D profile of the DICOM for the spoke."""
        px_x, px_y = get_pixel_size(dcm)
        if px_x != px_y:
            msg = "Only square pixels are supported"
            logger.critcal("%s but got (%f, %f)", msg, px_x, px_y)
            raise ValueError(msg)
        r_coords = np.linspace(
            0, self.length / px_x, size, endpoint=False,
        )
        theta_coords = np.zeros_like(r_coords) + self.theta

        x_coords = (
            (self.cx + offset[1]) / px_x
            + r_coords * np.sin(np.deg2rad(theta_coords))
        )
        y_coords = (
            (self.cy + offset[0]) / px_y
            - r_coords * np.cos(np.deg2rad(theta_coords))
        )

        profile = sp.ndimage.map_coordinates(
            dcm.pixel_array, [y_coords.ravel(), x_coords.ravel()], order=1,
        )

        rtn = [profile]

        if return_coords:
            rtn.append((x_coords, y_coords))
        if return_object_mask:
            object_mask = np.zeros((size, len(self)), dtype=bool)
            r = np.linspace(0, self.length, size)
            for i, obj in enumerate(self):
                obj_r_pos = np.sqrt(
                    (obj.x - self.cx) ** 2
                    + (obj.y - self.cy) ** 2,
                )
                object_mask[:, i] = np.abs(r - obj_r_pos) <= obj.diameter / 4

            rtn.append(object_mask)

        if len(rtn) == 1:
            return rtn[0]
        return tuple(rtn)


@dataclass
class LCODTemplate:
    """Template for the LCOD object."""

    # Position of the center of the LCOD on the image.
    cx: float
    cy: float

    # Angle of rotation for the 1st spoke (degrees)
    theta: float

    # Diameters of each low contrast object (mm)
    # Starting with the largest spoke and moving clockwise.
    diameters: tuple[float]  = (
        7.00, 6.39, 5.78, 5.17, 4.55, 3.94, 3.33, 2.72, 2.11, 1.50,
    )


    @functools.cached_property
    def spokes(self) -> Sequence[Spoke]:
        """Position of each of the low contrast object spokes."""
        return self._calc_spokes(
            self.cx, self.cy, self.theta, self.diameters,
        )

    @staticmethod
    @functools.lru_cache(maxsize=10)
    def _calc_spokes(
        cx: float,
        cy: float,
        theta: float,
        diameters: tuple[float],
    ) -> tuple[Spoke]:
        # Spokes start with the largest and decrease in size clockwise
        return tuple(
            Spoke(cx, cy, theta + i * (360 / len(diameters)), d)
            for i, d in enumerate(diameters)
        )

    def mask(
        self,
        dcm: pydicom.FileDataset,
        offset: tuple[float] = (0.0, 0.0),
        *,
        subset: str = "all",
        warn_if_object_out_of_bounds: bool = False,
    ) -> np.ndarray:
        """Mask the DICOM pixel array from the Spoke geometry.

        DICOM file is used for relating pixel geometry to
        the geometry of the acquisition.

        Similarly, offset (y, x) is used to account for cropped images.
        """
        mask = np.zeros_like(dcm.pixel_array, dtype=np.bool)

        # Convert from real coordinates to pixel coordinates
        dx, dy = get_pixel_size(dcm)
        y_grid, x_grid = np.meshgrid(
            *[
                np.linspace(v0, v0 + s * dv, num=s, endpoint=False)
                for v0, dv, s in zip(
                    offset, (dy, dx), mask.shape,
                )
            ],
            indexing="ij",
        )

        for sidx, spoke in enumerate(self.spokes):
            for oidx, obj in enumerate(spoke):
                is_object = (
                    (y_grid - obj.y) ** 2 + (x_grid - obj.x) ** 2
                    <= (obj.diameter / 2) ** 2
                )
                # Compare actual to measured area to check if object is on the
                # grid.
                mask_area = 2 * np.sum(is_object) * (dx * dy)
                obj_area = np.pi * (obj.diameter / 2) ** 2
                if warn_if_object_out_of_bounds and not np.isclose(
                    mask_area, obj_area, rtol=1e-1, atol=dx * dy,
                ):
                    logger.warning(
                        "Object %d in spoke %d is out of bounds.\n"
                        "File: %s\n"
                        "Object area:\t%f\nMask area:\t%f\n"
                        "Object:\n\tCenter: (%f, %f)\n\tDiameter: %f\n"
                        "Image:\n\tPixel size: (%f, %f)\n\tShape: (%d, %d)",
                        oidx,
                        sidx,
                        dcm.filename,
                        obj_area, mask_area,
                        obj.x, obj.y, obj.diameter,
                        dx, dy, *mask.shape,
                    )
                match subset:
                    case "all":
                        object_considered_for_mask = True
                    case "passed":
                        object_considered_for_mask = obj.detected
                    case "failed":
                        object_considered_for_mask = not obj.detected
                    case _:
                        logger.warning(
                            "Unrecognised mask subset %s."
                            " Object included in mask.",
                            subset,
                        )
                        object_considered_for_mask = True

                if object_considered_for_mask:
                    mask |= is_object
        return mask


@dataclass(frozen=True)
class FailedStatsModel:
    """Dataclass for the failed stats model."""

    @property
    def pvalues(self) -> np.ndarray:
        """Return the p-values - all ones."""
        return np.ones(3)

    @property
    def params(self) -> np.ndarray:
        """Return the p-values - all ones."""
        return np.ones(3)
