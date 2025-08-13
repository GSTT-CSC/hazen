"""Types used for hazenlib."""

from __future__ import annotations

# Typing imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    from numpy.typing import NDArray

# Python imports
import json
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any

################
# Base Classes #
################

class JsonSerializableMixin:
    """Mixim that supplies a shallow dict and json representation."""

    def to_dict(self) -> dict[str, Any]:
        """Return a shallow dictionary representation of the instance."""
        data = asdict(self) if is_dataclass(self) else dict(self.__dict__)
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

    name: str
    value: float | NDArray[np.float64]
    type: str = "measured"
    subtype: str = ""
    unit: str = ""

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
    report_images: Sequence[str] | None = None
    measurements: list[Measurement] = field(default_factory=list)
    metadata: Metadata = field(default_factory=Metadata)

    def add_measurement(self, measurement: Measurement) -> Measurement:
        """Add a measurement to the results."""
        self.measurements.append(measurement)
        return measurement

    def get_measurement(
            self,
            name: str,
            measurement_type: str | None = None,
            subtype: str | None = None,
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
                    and (unit is None or m.unit == unit)
            )
        ]
