""" Application-specific errors"""

# Python imports
from collections.abc import Sequence

# Local imports
from hazenlib.types import MEASUREMENT_NAMES, MEASUREMENT_TYPES


class ShapeError(Exception):
    """Base exception for shapes."""

    def __init__(self, shape, msg=None):
        if msg is None:
            # Default message
            msg = f"An error occured with {shape}"
        self.msg = msg
        self.shape = shape


class ShapeDetectionError(ShapeError):
    """Shape not found"""

    def __init__(self, shape, msg=None):
        if msg is None:
            # Default message
            msg = f"Could not find shape: {shape}"
        super(ShapeError, self).__init__(msg)
        self.shape = shape


class MultipleShapesError(ShapeDetectionError):
    """Shape not found"""

    def __init__(self, shape, msg=None):
        if msg is None:
            # Default message
            msg = f"Multiple {shape}s found. Multiple shape detection is currently unsupported."

        super(ShapeDetectionError, self).__init__(msg)
        self.shape = shape


class ArgumentCombinationError(Exception):
    """Argument combination not valid."""

    def __init__(self, msg="Invalid combination of arguments."):
        super().__init__(msg)


class InvalidMeasurementNameError(ValueError):
    """Invalid Measurement Name Error."""

    def __init__(self, name: str) -> None:
        """Initialise the error."""
        msg = (
            f"Invalid measurement name: {name}."
            " Must be one of {MEASUREMENT_NAMES}"
        )
        super().__init__(msg)


class InvalidMeasurementTypeError(ValueError):
    """Invalid Measurement Type Error."""

    def __init__(self, measurement_type: str) -> None:
        """Initialise the error."""
        msg = (
            f"Invalid measurement type: {measurement_type}."
            f" Must be one of {MEASUREMENT_TYPES}"
        )
        super().__init__(msg)
