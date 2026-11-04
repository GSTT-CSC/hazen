"""Tests for the Remote ACR protocol (acr_remote)."""

# Python imports
import contextlib
import io
import shutil
import sys
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path
from unittest.mock import Mock, patch

# Module imports
import openpyxl
import pydicom
import pytest
from PIL import Image as PILImage
from pydicom.dataset import FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

# Local imports
import hazenlib
from hazenlib.orchestration import TASK_REGISTRY, init_task
from hazenlib.protocols.acr_remote import ACRRemoteProtocol, classify_series
from hazenlib.protocols.engine import (
    Cell,
    ListedSeries,
    Step,
    StepOutcome,
    TableRow,
)
from hazenlib.types import Measurement, Result
from tests import TEST_DATA_DIR, mock_detect_orientation

LOGGER = "hazenlib.protocols.acr_remote"
REMOTE_SESSION = TEST_DATA_DIR / "acr" / "Siemens_Aera_1.5T_Remote"

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


def make_dicomdir(folder: Path, series: dict[str, list[str]]) -> Path:
    """Write a DICOMDIR listing *series* (description -> image file IDs)."""
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    for description, file_ids in series.items():
        record = pydicom.Dataset()
        record.DirectoryRecordType = "SERIES"
        record.SeriesDescription = description
        records.append(record)
        for file_id in file_ids:
            record = pydicom.Dataset()
            record.DirectoryRecordType = "IMAGE"
            record.ReferencedFileID = file_id.split("/")
            records.append(record)
    ds = pydicom.Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.1.3.10"
    ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
    ds.DirectoryRecordSequence = records
    ds.save_as(folder / "DICOMDIR", enforce_file_format=True)
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

    def test_duplicate_plane_keeps_newest(self) -> None:
        # (number, date, time) of the older then the newer series
        cases = [
            (("9", "20260721", "090000"), ("8", "20260722", "070000")),
            (("9", "20260721", "072110"), ("8", "20260721", "073531")),
            (("900", "20260721", "072110"), ("1015001", "20260721", "072110")),
            ((None, "20260721", "072110"), ("1", "20260721", "072110")),
        ]
        for older, newer in cases:
            # Both disk orders: kept first, and replacing the kept series
            for new_dir, old_dir in (("a", "b"), ("b", "a")):
                with self.subTest(older=older, newer=newer, first=new_dir):
                    root = self.root / str(older[0]) / new_dir
                    for folder, description, (number, date, time) in (
                        (new_dir, "ACR_SAG_new_90", newer),
                        (old_dir, "ACR_SAG", older),
                    ):
                        make_series(
                            root / folder,
                            SeriesNumber=number,
                            SeriesDescription=description,
                            SeriesDate=date,
                            SeriesTime=time,
                        )
                    with self.assertLogs(LOGGER) as logs:
                        session = ACRRemoteProtocol.discover(root)
                    old_no = older[0] or ""
                    self.assertEqual(session.roles["SAG"].path, root / new_dir)
                    self.assertEqual(
                        session.listed,
                        (
                            ListedSeries(
                                "ignored", root / old_dir, "ACR_SAG", old_no
                            ),
                        ),
                    )
                    self.assertEqual(len(logs.output), 1)
                    self.assertRegex(
                        logs.output[0],
                        rf"Using series {newer[0]} 'ACR_SAG_new_90' in {new_dir}"
                        rf" .* ignoring series {old_no} 'ACR_SAG' in {old_dir}$",
                    )

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

    def test_dicomdir_images_missing_from_disk_warn(self) -> None:
        # The site's export had a DICOMDIR per plane but no SAG/TRA images
        # File IDs of one component are read back as a str, not a list.
        # Only existence is checked, so the listed images can be empty.
        tra = make_series(self.root / "tra" / "DICOM" / "1")
        make_dicomdir(tra.parents[1], {"ACR_TRA": ["DICOM/1/IM1", "IM2"]})
        (tra / "IM1").write_bytes(b"")
        (tra.parents[1] / "IM2").write_bytes(b"")
        make_dicomdir(
            self.root / "site" / "sag",
            {
                "ACR_SAG": ["DICOM/2/IM1", "IM2"],
                "ACR_SAG_ND": ["DICOM/3/IM1"],
            },
        )
        with self.assertLogs(LOGGER, "WARNING") as logs:
            session = ACRRemoteProtocol.discover(self.root)
        self.assertEqual(list(session.roles), ["TRA"])
        self.assertEqual(len(logs.output), 1)
        self.assertIn(
            "DICOMDIR in site/sag lists 3 images that are not on disk"
            " ('ACR_SAG', 'ACR_SAG_ND')",
            logs.output[0],
        )

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


def fake_init_task(task_name: str, files: list, **kwargs: object) -> Mock:
    """Stand-in for init_task returning one measurement per task."""
    result = Result(task=task_name, _load_metadata=False)
    result.add_measurement(Measurement("SNR", 150.0, type="measured"))
    task = Mock()
    task.run.return_value = result
    return task


@patch("hazenlib.protocols.engine.init_task", side_effect=fake_init_task)
class TestRemoteRun(unittest.TestCase):
    """The engine runs Remote steps, with tasks mocked out."""

    def setUp(self) -> None:
        # The synthetic series have no pixel spacing to detect it from
        mock_detect_orientation(self)
        self.root = Path(tempfile.mkdtemp()).resolve()
        for n, plane in enumerate(("TRA", "SAG", "COR")):
            make_series(
                self.root / plane,
                SeriesNumber=n,
                SeriesDescription=f"ACR_{plane}",
            )

    def tearDown(self) -> None:
        shutil.rmtree(self.root)

    def test_single_image_snr(self, init_task: Mock) -> None:
        protocol = ACRRemoteProtocol(self.root, report=False)
        outcomes = protocol.run(debug=True)

        self.assertEqual([o.status for o in outcomes], ["ok"] * 18)
        # (task, plane folder) per call, in call order
        calls = [
            (c.args[0], Path(c.args[1][0]).parent.name)
            for c in init_task.call_args_list
        ]
        for c in init_task.call_args_list:
            if c.args[0] == "acr_snr":
                self.assertNotIn("subtract", c.kwargs)
                self.assertNotIn("measured_slice_width", c.kwargs)
        for plane in ("TRA", "SAG", "COR"):
            with self.subTest(plane=plane):
                # First stage: SNR does not wait for the slice thickness
                self.assertLess(
                    calls.index(("acr_snr", plane)),
                    calls.index(("acr_slice_thickness", plane)),
                )
        # Single-image SNR is not normalised by a slice width
        labels = [r.measurement for r in protocol.table(outcomes).rows]
        self.assertFalse([m for m in labels if m.startswith("Slice width")])

    def test_summary_listed_counts(self, init_task: Mock) -> None:
        def first_line() -> str:
            protocol = ACRRemoteProtocol(self.root, report=False)
            return protocol.summary([]).splitlines()[0]

        self.assertTrue(first_line().endswith(" - 3 roles found"))

        extras = self.root / "extras"
        make_series(extras, SeriesNumber=10, SeriesDescription="loc")
        make_series(extras, SeriesNumber=14, SeriesDescription="loc2")
        make_series(extras, SeriesNumber=11, SeriesDescription="RF_noise")
        make_series(extras, SeriesNumber=12, SeriesDescription="B0_map")
        make_series(
            extras,
            SeriesNumber=13,
            SeriesDescription="ACR_TRA_ND",
            ImageType=list(ND),
        )
        line = first_line()
        self.assertTrue(
            line.endswith(
                " - 3 roles found · listed: 1 uncorrected, 2 visual check,"
                " 2 unknown (loc, loc2)",
            ),
            line,
        )


def table_init_task(task_name: str, files: list, **kwargs: object) -> Mock:
    """Stand-in for init_task: SAG values are TRA's plus one."""
    plane = Path(files[0]).parent.name
    offset = {"TRA": 0, "SAG": 1}[plane]
    if task_name == "acr_slice_thickness" or (
        task_name == "acr_ghosting" and plane == "SAG"
    ):
        raise RuntimeError("no ramps")
    measurements = {
        "acr_snr": [
            Measurement("SNR", 150.0, type="normalised", subtype="smoothing"),
            Measurement(
                "SNR",
                9.0,
                subtype="smoothing",
                visibility="intermediate",
            ),
            Measurement("SNR", 1.0, type="normalised"),
        ],
        "acr_uniformity": [
            # Slice N only counts at the start of the description
            Measurement(
                "Uniformity",
                95.0,
                subtype="Integral uniformity",
                description="ACR_TRA Slice 7",
                unit="%",
            ),
            Measurement("Uniformity", 4.0),
        ],
        "acr_ghosting": [
            Measurement("Ghosting", 0.5, subtype="signal ghosting", unit="%"),
        ],
        "acr_slice_position": [
            Measurement(
                "SlicePosition",
                -3.0,
                subtype="length difference",
                description="Slice 11 ACR_TRA_3001_11",
            ),
            Measurement(
                "SlicePosition",
                2.0,
                subtype="length difference",
                description="Slice 1 ACR_TRA_3001_1",
            ),
        ],
        "acr_geometric_accuracy": [
            Measurement(
                "GeometricAccuracy",
                190.0,
                subtype="Slice 1 Horizontal distance",
                description="Slice 1 ACR_TRA_3001_1",
                unit="mm",
            ),
        ],
    }[task_name]
    result = Result(task=task_name, _load_metadata=False)
    for m in measurements:
        result.add_measurement(
            Measurement(
                m.name,
                m.value + offset,
                type=m.type,
                subtype=m.subtype,
                description=m.description,
                unit=m.unit,
                visibility=m.visibility,
            ),
        )
    task = Mock()
    task.run.return_value = result
    return task


class TestRemoteTable(unittest.TestCase):
    """Final measurements in one row per test and label, a column per role."""

    def setUp(self) -> None:
        mock_detect_orientation(self)
        self.root = Path(tempfile.mkdtemp()).resolve()
        for n, plane in enumerate(("TRA", "SAG")):  # no COR
            make_series(
                self.root / plane,
                SeriesNumber=n,
                SeriesDescription=f"ACR_{plane}",
            )

    def tearDown(self) -> None:
        shutil.rmtree(self.root)

    @patch("hazenlib.protocols.engine.init_task", side_effect=table_init_task)
    def test_table(self, _init_task: Mock) -> None:
        protocol = ACRRemoteProtocol(self.root, report=False)
        table = protocol.table(protocol.run(debug=True))

        red = Cell(None, missing=True)
        self.assertEqual(table.columns, ("TRA", "SAG", "COR"))
        self.assertEqual(
            list(table.rows),
            [
                TableRow(
                    "snr",
                    "Normalised SNR (smoothing)",
                    {"TRA": Cell(150.0), "SAG": Cell(151.0), "COR": red},
                ),
                TableRow(
                    "snr",
                    "Normalised SNR",
                    {"TRA": Cell(1.0), "SAG": Cell(2.0), "COR": red},
                ),
                TableRow(
                    "uniformity",
                    "Integral uniformity",
                    {
                        "TRA": Cell(95.0, "%"),
                        "SAG": Cell(96.0, "%"),
                        "COR": red,
                    },
                ),
                TableRow(
                    "uniformity",
                    "Uniformity",
                    {"TRA": Cell(4.0), "SAG": Cell(5.0), "COR": red},
                ),
                # Failed on SAG, skipped on COR
                TableRow(
                    "ghosting",
                    "Signal ghosting",
                    {"TRA": Cell(0.5, "%"), "SAG": red, "COR": red},
                ),
                # Failed everywhere, so no measurement names are known
                TableRow(
                    "slice_thickness",
                    "",
                    {"TRA": red, "SAG": red, "COR": red},
                ),
                TableRow(
                    "slice_position",
                    "Length difference (slice 11)",
                    {"TRA": Cell(-3.0), "SAG": Cell(-2.0), "COR": red},
                ),
                TableRow(
                    "slice_position",
                    "Length difference (slice 1)",
                    {"TRA": Cell(2.0), "SAG": Cell(3.0), "COR": red},
                ),
                TableRow(
                    "geometric_accuracy",
                    "Slice 1 Horizontal distance",
                    {
                        "TRA": Cell(190.0, "mm"),
                        "SAG": Cell(191.0, "mm"),
                        "COR": red,
                    },
                ),
            ],
        )


HEADER = ["test", "measurement", "TRA", "SAG", "COR"]


def results_header(session: Path) -> list[object]:
    """Header row of the Results sheet written into *session*."""
    ws = openpyxl.load_workbook(session / "hazen_results.xlsx")["Results"]
    return [c.value for c in ws[1]]


def image_init_task(
    task_name: str,
    files: list,
    report_dir: str,
    **kwargs: object,
) -> Mock:
    """Stand-in for init_task saving one report image, none for ghosting."""
    class_name = TASK_REGISTRY[task_name].class_name
    image = Path(report_dir) / class_name / "result.png"
    image.parent.mkdir(parents=True, exist_ok=True)
    PILImage.new("RGB", (40, 30), "white").save(image)
    result = Result(task=class_name, _load_metadata=False)
    result.add_measurement(Measurement("SNR", 150.0))
    if task_name != "acr_ghosting":
        result.add_report_image(str(image))
    task = Mock()
    task.run.return_value = result
    return task


def mock_tasks(test: Callable[..., None]) -> Callable[..., None]:
    """Mock the tasks and orientation detection out and run in-process."""
    return patch(
        "hazenlib.protocols.engine.init_task",
        side_effect=fake_init_task,
    )(
        patch(
            "hazenlib.protocols.engine.wait_on_parallel_results",
            side_effect=lambda f, args, debug=False: [f(*a) for a in args],
        )(
            patch(
                "hazenlib.protocols.engine.detect_orientation",
                new=Mock(return_value=None),  # no extra test argument
            )(test),
        ),
    )


class TestRemoteCli(unittest.TestCase):
    """``hazen acr_remote`` shares the protocol options but not --vendor."""

    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp()).resolve()
        # Not Siemens, so the vendor visibly comes from the headers
        for n, plane in enumerate(("TRA", "SAG", "COR")):
            make_series(
                self.root / plane,
                SeriesNumber=n,
                SeriesDescription=f"ACR_{plane}",
                Manufacturer="Philips",
            )

    def tearDown(self) -> None:
        shutil.rmtree(self.root)

    @staticmethod
    def main(session: Path, *extra: str) -> tuple[str, str]:
        """Run ``hazen acr_remote``, returning stdout and stderr."""
        argv = ["hazen", "acr_remote", str(session), *extra]
        out, err = io.StringIO(), io.StringIO()
        with (
            patch.object(sys, "argv", argv),
            contextlib.redirect_stdout(out),
            contextlib.redirect_stderr(err),
        ):
            hazenlib.main()
        return out.getvalue(), err.getvalue()

    def test_parser(self) -> None:
        parser = hazenlib.get_parser()
        self.assertEqual(
            parser.parse_args(["acr_gstt", "s", "--vendor", "ge"]).vendor,
            "ge",
        )
        for argv, message in [
            (["acr_remote", "s", "--vendor", "ge"], "unrecognized arguments"),
            (["acr_gstt", "s", "--vendor", "canon"], "invalid choice"),
        ]:
            with (
                self.subTest(argv=argv),
                self.assertRaises(SystemExit),
                contextlib.redirect_stderr(io.StringIO()) as err,
            ):
                parser.parse_args(argv)
            self.assertIn(message, err.getvalue())

    @mock_tasks
    def test_run(self, _wait: Mock, init_task: Mock) -> None:
        images = self.root / "imgs"
        out, _ = self.main(self.root, "--no-report", "--output", str(images))
        self.assertIn("philips rules - 3 roles found", out)
        self.assertNotIn("Images:", out)
        for c in init_task.call_args_list:
            self.assertFalse(c.kwargs["report"])
            self.assertEqual(Path(c.kwargs["report_dir"]).parent, images)
        self.assertEqual(results_header(self.root), HEADER)
        # Without --report no image sheets, even if a task returns images
        init_task.side_effect = image_init_task
        self.main(self.root, "--no-report", "--output", str(images))
        wb = openpyxl.load_workbook(self.root / "hazen_results.xlsx")
        self.assertEqual(wb.sheetnames, ["Results"])

    @mock_tasks
    def test_report_images(self, _wait: Mock, init_task: Mock) -> None:
        init_task.side_effect = image_init_task
        out, _ = self.main(self.root)
        # Paths line up under each other
        self.assertIn(f"Images:  {self.root / 'hazen_report'}", out)
        wb = openpyxl.load_workbook(self.root / "hazen_results.xlsx")
        self.assertEqual(
            wb.sheetnames,
            [
                "Results",
                "snr",
                "uniformity",
                "slice_thickness",
                "slice_position",
                "geometric_accuracy",
            ],
        )
        headings = [c.value for c in wb["slice_position"]["A"] if c.value]
        self.assertEqual(
            headings,
            ["TRA_SlicePosition", "SAG_SlicePosition", "COR_SlicePosition"],
        )
        self.assertEqual(wb["snr"]["A1"].value, "TRA_SNR")

    @mock_tasks
    def test_exit_codes(self, _wait: Mock, init_task: Mock) -> None:
        for session, failing, code, message in [
            (self.root, True, 1, ""),
            (self.root / "missing", False, 2, "Session folder not found"),
        ]:
            with self.subTest(session=session.name, failing=failing):
                init_task.side_effect = (
                    RuntimeError("boom") if failing else fake_init_task
                )
                argv = ["hazen", "acr_remote", str(session), "--no-report"]
                with (
                    patch.object(sys, "argv", argv),
                    contextlib.redirect_stdout(io.StringIO()),
                    contextlib.redirect_stderr(io.StringIO()) as err,
                    self.assertRaises(SystemExit) as raised,
                ):
                    hazenlib.main()
                self.assertEqual(raised.exception.code, code)
                self.assertIn(message, err.getvalue())

    @pytest.mark.slow
    def test_cli(self) -> None:
        session = self.root / "export"
        shutil.copytree(REMOTE_SESSION, session)
        out, _ = self.main(session, "--no-report")
        self.assertIn("3 roles found · listed: 3 uncorrected", out)
        self.assertIn("ok 18 · skipped 0 · failed 0", out)
        self.assertEqual(results_header(session), HEADER)


class TestRemoteSession(unittest.TestCase):
    """End to end on the scrubbed Siemens Aera Remote export."""

    protocol: ACRRemoteProtocol
    outcomes: list[StepOutcome]
    report_dir: Path

    @classmethod
    def setUpClass(cls) -> None:
        # Tasks create their report folders even without --report
        cls.report_dir = Path(tempfile.mkdtemp())
        cls.addClassCleanup(shutil.rmtree, cls.report_dir)
        cls.protocol = ACRRemoteProtocol(
            REMOTE_SESSION,
            report=False,
            report_dir=cls.report_dir,
        )
        cls.outcomes = cls.protocol.run()

    def test_corrected_series_are_the_roles(self) -> None:
        session = self.protocol.session
        self.assertEqual(
            {role: rs.series_number for role, rs in session.roles.items()},
            {"TRA": "4001", "SAG": "1020001", "COR": "10001"},
        )
        self.assertEqual(
            sorted((ls.kind, ls.series_number) for ls in session.listed),
            [
                ("uncorrected", "1013001"),
                ("uncorrected", "3001"),
                ("uncorrected", "9001"),
            ],
        )

    def test_matches_individual_tasks(self) -> None:
        """Protocol output must equal running each task on its own."""
        self.assertEqual([o.status for o in self.outcomes], ["ok"] * 18)
        for o in self.outcomes:
            with self.subTest(role=o.step.role, test=o.step.test):
                if o.result is None:
                    self.fail("no result")
                files = list(self.protocol.session.roles[o.step.role].files)
                expected = init_task(
                    o.step.task,
                    files,
                    report=False,
                    report_dir=str(self.report_dir / "individual"),
                ).run()
                self.assertEqual(o.result.measurements, expected.measurements)

    def test_snr_values(self) -> None:
        # Pinned hazen output. Reports made with hazen <= 2.0.0 gave
        # TRA 181.74, SAG 161.19 and COR 195.08; #531's ROI geometry
        # (20 mm squares at 40 mm) and float smoothing moved them.
        for role, snr in {"TRA": 184.23, "SAG": 173.99, "COR": 200.08}.items():
            with self.subTest(role=role):
                outcome = next(
                    o
                    for o in self.outcomes
                    if o.step.role == role and o.step.test == "snr"
                )
                if outcome.result is None:
                    self.fail("no result")
                measured = outcome.result.get_measurement(
                    name="SNR",
                    measurement_type="measured",
                )
                self.assertAlmostEqual(float(measured[0].value), snr, places=2)
