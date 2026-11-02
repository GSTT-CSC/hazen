"""Tests for the Remote ACR protocol (acr_remote)."""

# Python imports
import shutil
import tempfile
import unittest
from pathlib import Path

# Module imports
import pydicom
from pydicom.dataset import FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

# Local imports
from hazenlib.protocols.acr_remote import ACRRemoteProtocol, classify_series
from hazenlib.protocols.engine import Step

LOGGER = "hazenlib.protocols.acr_remote"

DIS2D = ("ORIGINAL", "PRIMARY", "M", "NORM", "DIS2D", "MFSPLIT")
ND = ("ORIGINAL", "PRIMARY", "M", "ND", "NORM", "MFSPLIT")
NO_MARKER = ("ORIGINAL", "PRIMARY", "M", "NORM")
SIEMENS = "siemens"
UNCORRECTED = "uncorrected"
UNKNOWN = "unknown"


def make_series(folder: Path, n_files: int = 2, **tags: object) -> Path:
    """Write a header-only DICOM series, Siemens DIS2D ACR_TRA by default."""
    folder.mkdir(parents=True, exist_ok=True)
    tags = {
        "SeriesInstanceUID": generate_uid(),
        "SeriesDescription": "ACR_TRA",
        "SeriesNumber": 1,
        "ProtocolName": "",
        "ImageType": list(DIS2D),
        "Manufacturer": "SIEMENS",
        "DeviceSerialNumber": "1234",
        "StationName": "AWP1",
        "StudyDate": "20261101",
        **tags,
    }
    for i in range(n_files):
        ds = pydicom.Dataset()
        ds.file_meta = FileMetaDataset()
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds.file_meta.MediaStorageSOPClassUID = MRImageStorage
        ds.SOPClassUID = MRImageStorage
        ds.SOPInstanceUID = generate_uid()
        for keyword, value in tags.items():
            if value is not None:
                setattr(ds, keyword, value)
        ds.save_as(
            folder / f"{tags['SeriesNumber']}_{i}.dcm",
            enforce_file_format=True,
        )
    return folder


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


class TestRemoteDiscover(unittest.TestCase):
    """Series in an export are grouped and matched on their headers."""

    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp()).resolve()

    def tearDown(self) -> None:
        shutil.rmtree(self.root)

    def test_roles_and_listed_series(self) -> None:
        # Export folder names mean nothing; COR sorts before TRA on disk
        make_series(self.root / "a" / "1", SeriesDescription="ACR_COR")
        make_series(self.root / "b" / "1", n_files=3, SeriesNumber=4)
        make_series(
            self.root / "b" / "1",
            SeriesNumber=3,
            SeriesDescription="ACR_TRA_ND",
            ImageType=list(ND),
        )
        make_series(self.root / "c", SeriesDescription="RF_noise")
        make_series(
            self.root / "c" / "d",
            SeriesNumber=9,
            SeriesDescription="loc",
        )
        with self.assertLogs(LOGGER) as logs:
            session = ACRRemoteProtocol.discover(self.root)

        self.assertEqual(list(session.roles), ["TRA", "COR"])
        tra = session.roles["TRA"]
        self.assertEqual(len(tra.files), 3)
        self.assertEqual(
            (
                tra.path,
                tra.series_description,
                tra.series_number,
                tra.manufacturer,
                tra.vendor,
            ),
            (self.root / "b" / "1", "ACR_TRA", "4", "SIEMENS", "siemens"),
        )
        self.assertEqual(
            sorted(
                (ls.kind, ls.path, ls.series_description, ls.series_number)
                for ls in session.listed
            ),
            [
                ("uncorrected", self.root / "b" / "1", "ACR_TRA_ND", "3"),
                ("unknown", self.root / "c" / "d", "loc", "9"),
                ("visual_check", self.root / "c", "RF_noise", "1"),
            ],
        )
        self.assertEqual(len(logs.output), 1)  # only the unknown series
        self.assertIn("series 9 'loc' in c/d", logs.output[0])

    def test_uncorrected_only_plane_warns(self) -> None:
        # The plane may come from ProtocolName only
        for description, protocol in (
            ("ACR_SAG_ND", ""),
            ("t1_ND", "ACR_SAG"),
        ):
            with self.subTest(description=description, protocol=protocol):
                root = self.root / description
                make_series(
                    root,
                    SeriesDescription=description,
                    ProtocolName=protocol,
                    ImageType=list(ND),
                )
                with self.assertLogs(LOGGER) as logs:
                    session = ACRRemoteProtocol.discover(root)
                self.assertEqual(session.roles, {})
                self.assertEqual(
                    [ls.kind for ls in session.listed],
                    ["uncorrected"],
                )
                self.assertRegex(
                    logs.output[-1],
                    "no distortion-corrected .* SAG",
                )

    def test_duplicate_plane_raises(self) -> None:
        make_series(self.root / "1", SeriesNumber=4)
        make_series(self.root / "2", SeriesNumber=5)
        with self.assertRaisesRegex(ValueError, "series 4 .* series 5"):
            ACRRemoteProtocol.discover(self.root)

    def test_more_than_one_scanner_raises(self) -> None:
        # serials, stations, raises
        cases = [
            (("1", "2"), ("AWP1", "AWP1"), True),
            ((None, None), ("AWP1", "AWP2"), True),
            (("1", "1"), ("AWP1", "AWP2"), False),
            (("1", None), ("AWP1", None), False),
        ]
        for serials, stations, raises in cases:
            with self.subTest(serials=serials, stations=stations):
                root = self.root / str(serials) / str(stations)
                for n, (serial, station) in enumerate(
                    zip(serials, stations, strict=True),
                ):
                    make_series(
                        root / str(n),
                        SeriesNumber=n,
                        SeriesDescription=["ACR_TRA", "ACR_SAG"][n],
                        DeviceSerialNumber=serial,
                        StationName=station,
                    )
                if raises:
                    with self.assertRaisesRegex(ValueError, "scanner"):
                        ACRRemoteProtocol.discover(root)
                else:
                    self.assertEqual(
                        len(ACRRemoteProtocol.discover(root).roles),
                        2,
                    )

    def test_more_than_one_date_warns(self) -> None:
        make_series(self.root / "1", StudyDate="20261101")
        make_series(
            self.root / "2",
            SeriesDescription="ACR_SAG",
            StudyDate="20261108",
        )
        with self.assertLogs(LOGGER) as logs:
            session = ACRRemoteProtocol.discover(self.root)
        self.assertEqual(list(session.roles), ["TRA", "SAG"])
        self.assertIn("20261101, 20261108", logs.output[0])

    def test_missing_date_is_not_a_second_date(self) -> None:
        make_series(self.root / "1")
        make_series(
            self.root / "2", SeriesDescription="ACR_SAG", StudyDate=None
        )
        with self.assertNoLogs(LOGGER, "WARNING"):
            ACRRemoteProtocol.discover(self.root)

    def test_unknown_manufacturer_has_no_vendor(self) -> None:
        make_series(self.root / "1", Manufacturer="Acme")
        session = ACRRemoteProtocol.discover(self.root)
        self.assertIsNone(session.roles["TRA"].vendor)

    def test_skips_report_folder_and_non_series_files(self) -> None:
        # Only a report folder inside the session counts, not an ancestor
        root = self.root / "hazen_report" / "export"
        make_series(root / "1")
        make_series(root / "hazen_report", SeriesDescription="ACR_SAG")
        make_series(root / "DICOMDIR_dir", SeriesInstanceUID=None)
        (root / ".DS_Store").write_bytes(b"\x00\x01")
        (root / "WI.docx").write_bytes(b"PK\x03\x04")
        session = ACRRemoteProtocol.discover(root)
        self.assertEqual(list(session.roles), ["TRA"])
        self.assertEqual(session.listed, ())

    def test_missing_session_raises(self) -> None:
        with self.assertRaises(NotADirectoryError):
            ACRRemoteProtocol.discover(self.root / "nope")


class TestRemoteSteps(unittest.TestCase):
    """Six tests per plane; SNR is single-image on the corrected series."""

    def test_steps(self) -> None:
        expected = tuple(
            Step(test, f"acr_{test}", role)
            for role in ("TRA", "SAG", "COR")
            for test in (
                "snr",
                "uniformity",
                "ghosting",
                "slice_thickness",
                "slice_position",
                "geometric_accuracy",
            )
        )
        self.assertEqual(
            ACRRemoteProtocol.supported_vendors,
            ("siemens", "philips", "ge"),
        )
        for vendor in ACRRemoteProtocol.supported_vendors:
            with self.subTest(vendor=vendor):
                self.assertEqual(
                    ACRRemoteProtocol.build_steps(vendor), expected
                )
