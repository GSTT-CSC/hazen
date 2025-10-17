"""Constants used throughout hazenlib."""

from typing import Literal

#######################################
# Allowed measurement types and names #
#######################################

MEASUREMENT_NAMES = Literal[
    "GeometricAccuracy",
    "Ghosting",
    "LowContrastObjectDetectability",
    "Relaxometry",
    "SlicePosition",
    "SliceWidth",
    "SNR",
    "SNRMap",
    "SpatialResolution",
    "Uniformity",
]

MEASUREMENT_TYPES = Literal["measured", "normalised", "fitted", "raw"]
