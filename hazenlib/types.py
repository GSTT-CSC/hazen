"""Types used for hazenlib."""

from __future__ import annotations

# Typing imports
from typing import TYPE_CHECKING

from hazenlib.logger import logger

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

# Python imports
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, get_args

# Local imports
from hazenlib.constants import MEASUREMENT_NAMES, MEASUREMENT_TYPES
from hazenlib.exceptions import (InvalidMeasurementNameError,
                                 InvalidMeasurementTypeError)

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
    files: str | Sequence[str] | None = None

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
        return tuple(self._report_images)


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
