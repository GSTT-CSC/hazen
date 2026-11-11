from unittest import mock

import pytest


@pytest.fixture(autouse=True)
def _no_image_viewer():
    """Stop CLI tests opening report images in the desktop image viewer."""
    with mock.patch("hazenlib.open_report_images"):
        yield
