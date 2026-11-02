"""Tests for the Remote ACR protocol (acr_remote)."""

# Python imports
import unittest

# Local imports
from hazenlib.protocols.acr_remote import classify_series

LOGGER = "hazenlib.protocols.acr_remote"

DIS2D = ("ORIGINAL", "PRIMARY", "M", "NORM", "DIS2D", "MFSPLIT")
ND = ("ORIGINAL", "PRIMARY", "M", "ND", "NORM", "MFSPLIT")
NO_MARKER = ("ORIGINAL", "PRIMARY", "M", "NORM")
SIEMENS = "siemens"
UNCORRECTED = "uncorrected"
UNKNOWN = "unknown"

VISUAL_CHECK_TOKENS = (
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


class TestClassifySeries(unittest.TestCase):
    def test_table(self) -> None:
        # description, protocol name, image type, vendor,
        # expected role, expected listed kind, one phrase per warning
        cases = [
            ("ACR_TRA", "ACR_TRA", DIS2D, SIEMENS, "TRA", None, ()),
            ("ACR_SAG_new90", "ACR_SAG", DIS2D, SIEMENS, "SAG", None, ()),
            ("ACR Axial", "", DIS2D, SIEMENS, "TRA", None, ()),
            ("ACR transverse", "", DIS2D, SIEMENS, "TRA", None, ()),
            ("ACR_AX", "", DIS2D, SIEMENS, "TRA", None, ()),
            ("ACR_Sagittal", "", DIS2D, SIEMENS, "SAG", None, ()),
            ("acr-coronal", "", DIS2D, SIEMENS, "COR", None, ()),
            ("localiser", "ACR_SAG", DIS2D, SIEMENS, "SAG", None, ()),
            ("ACR_TRA", "noise_check", DIS2D, SIEMENS, "TRA", None, ()),
            ("ACR_TRA", "", ("DIS2D", "ND"), SIEMENS, "TRA", None, ()),
            ("ACR_TRA_ND2", "", DIS2D, SIEMENS, "TRA", None, ()),
            ("ACR_TRA_2nd", "", DIS2D, SIEMENS, "TRA", None, ()),
            ("ACR_TRA_ND", "ACR_TRA", ND, SIEMENS, None, UNCORRECTED, ()),
            ("ACR_TRA", "ACR_COR", DIS2D, SIEMENS, "TRA", None, ("ACR_COR",)),
            ("ACR_COR_ND", "", DIS2D, SIEMENS, "COR", None, ("_ND",)),
            ("ACR_COR", "", ND, SIEMENS, None, UNCORRECTED, ("_ND",)),
            ("ACR_TRA", "", NO_MARKER, SIEMENS, None, UNKNOWN, ("DIS2D",)),
            ("t1_localiser", "", DIS2D, SIEMENS, None, UNKNOWN, ("ACR",)),
            ("ACR_corrected", "", DIS2D, SIEMENS, None, UNKNOWN, ("ACR",)),
            ("XACR_TRA", "", DIS2D, SIEMENS, None, UNKNOWN, ("ACR",)),
            ("ACR_SAG", "", ("ORIGINAL",), "ge", "SAG", None, ("corr",)),
            ("localiser", "", ("ORIGINAL",), "ge", None, UNKNOWN, ("ACR",)),
            (
                "ACR_TRA_ND",
                "ACR_COR",
                DIS2D,
                SIEMENS,
                "TRA",
                None,
                ("ACR_COR", "_ND"),
            ),
            *(
                (
                    f"RF_{t.upper()}",
                    "",
                    DIS2D,
                    SIEMENS,
                    None,
                    "visual_check",
                    (),
                )
                for t in VISUAL_CHECK_TOKENS
            ),
        ]
        for desc, protocol, image_type, vendor, role, kind, warnings in cases:
            with self.subTest(desc=desc, protocol=protocol, vendor=vendor):
                call = lambda: classify_series(  # noqa: E731
                    desc,
                    protocol,
                    image_type,
                    vendor,
                    label="series 7",
                )
                if not warnings:
                    with self.assertNoLogs(LOGGER, "WARNING"):
                        found_role, found_kind = call()
                else:
                    with self.assertLogs(LOGGER, "WARNING") as logs:
                        found_role, found_kind = call()
                    self.assertEqual(len(logs.output), len(warnings))
                    for phrase, line in zip(
                        warnings, logs.output, strict=True
                    ):
                        self.assertIn(phrase, line)
                        self.assertIn("series 7", line)
                self.assertEqual(
                    found_role.name if found_role else None,
                    role,
                )
                self.assertEqual(found_kind, kind)

    def test_warning_names_description_without_label(self) -> None:
        with self.assertLogs(LOGGER, "WARNING") as logs:
            classify_series("ACR_TRA", "", NO_MARKER, SIEMENS)
        self.assertIn("'ACR_TRA'", logs.output[0])
