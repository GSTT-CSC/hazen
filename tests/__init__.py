import os
import pathlib

import numpy as np

TEST_DATA_DIR = pathlib.Path(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")
)
TEST_REPORT_DIR = pathlib.Path(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "report")
)


def set_pixels(dcm, arr):
    """Replace the pixel data of a pydicom dataset in place.

    ``arr`` must already have the dataset's pixel dtype.
    """
    arr = np.ascontiguousarray(arr)
    dcm.Rows, dcm.Columns = arr.shape
    dcm.PixelData = arr.tobytes()
    # Drop pydicom's cached array so the new data is decoded.
    dcm._pixel_array = None
    dcm._pixel_id = {}


def transform_series(acr_obj, k, flip, tilt=0.0):
    """(Tilt, then) rotate every slice by k * 90 degrees and optionally
    mirror it, as if the series had been exported that way."""
    import scipy.ndimage

    for dcm in acr_obj.slice_stack:
        arr = dcm.pixel_array
        if tilt:
            tilted = scipy.ndimage.rotate(
                arr.astype(float), tilt, reshape=False, order=3
            )
            arr = np.clip(tilted, 0, None).round().astype(arr.dtype)
        arr = np.rot90(arr, k)
        set_pixels(dcm, np.fliplr(arr) if flip else arr)
    if k % 2:
        # rows and columns swap, so does the pixel spacing
        acr_obj.dx, acr_obj.dy = acr_obj.dy, acr_obj.dx
