""" Application-specific errors"""


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
    def __init__(self, msg='Invalid combination of arguments.'):
        super().__init__(msg)