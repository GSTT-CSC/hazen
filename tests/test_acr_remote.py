"""Tests for the Remote ACR protocol (acr_remote)."""

# Python imports
import contextlib
import csv
import io
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

# Module imports
import pydicom
from pydicom.dataset import FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

# Local imports
import hazenlib
from hazenlib.protocols.acr_remote import ACRRemoteProtocol, classify_series
from hazenlib.protocols.engine import Step
from hazenlib.types import Measurement, Result

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
        rows = protocol.rows(outcomes)
        self.assertEqual(len(rows), 18)
        self.assertNotIn("SliceWidth", [r.measurement for r in rows])
        self.assertEqual({r.detail for r in rows}, {""})

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


@patch("hazenlib.protocols.engine.init_task", side_effect=fake_init_task)
@patch(
    "hazenlib.protocols.engine.wait_on_parallel_results",
    side_effect=lambda f, args, debug=False: [f(*a) for a in args],
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

    def test_parser(self, *_mocks: Mock) -> None:
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

    def test_run(self, _wait: Mock, init_task: Mock) -> None:
        images = self.root / "imgs"
        out, _ = self.main(self.root, "--no-report", "--output", str(images))
        self.assertIn("philips rules - 3 roles found", out)
        self.assertNotIn("Images:", out)
        for c in init_task.call_args_list:
            self.assertFalse(c.kwargs["report"])
            self.assertEqual(Path(c.kwargs["report_dir"]).parent, images)
        with (self.root / "hazen_results.csv").open(newline="") as fh:
            roles = {row["role"] for row in csv.DictReader(fh)}
        self.assertEqual(roles, {"TRA", "SAG", "COR"})

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
