"""Types used for hazenlib."""

# ruff: noqa: ANN401

from __future__ import annotations

# Typing imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    from numpy.typing import NDArray

# Python Imports
from collections import defaultdict
from typing import Any, Mapping


class _FixedDict(dict):
    """Dictionary with keys fixed on initialisation."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialise the FixedDict."""
        super().__init__(*args, **kwargs)

    def __setitem__(self, key: str, value: Any) -> None:
        if key not in self:
            raise KeyError(key)
        super().__setitem__(key, value)


# ----------------------------------------------------------------------
# The canonical result that every task
# ----------------------------------------------------------------------

class Result(_FixedDict):
    """Canonical result any Task.run() must return."""

    def __init__(
        self,
        task: str,
        file: str | Sequence[str] | None = None,
        measurement: Measurement | Mapping[str, Measurement] | None = None,
        report_image: Sequence[str] | None = None,
    ) -> None:
        """Initialise the typed dictionary."""
        measurement = defaultdict() if measurement is None else measurement
        super().__init__(
            task=task,
            file=file,
            measurement=measurement,
            report_image=report_image,
        )


class Measurement(_FixedDict):
    """Canonical result each measurment must have."""

    def __init__(
        self,
        name: str,
        value: float | NDArray[np.float64],
        unit: str = "",
    ) -> None:
        """Initialise fixed dictionary."""
        super().__init__(name=name, value=value, unit=unit)
