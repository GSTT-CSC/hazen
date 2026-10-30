# Python imports
import logging
import sys
import tomllib
import unittest
from unittest import mock
from pathlib import Path

import hazenlib
from hazenlib.tasks.relaxometry import Relaxometry
from hazenlib.tasks.snr import SNR

from hazenlib.types import Measurement, Result
from hazenlib.utils import dcmread, get_dicom_files

# Local imports
from tests import TEST_DATA_DIR


logger = logging.getLogger(__name__)


class TestCliParser(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        self.file = str(
            TEST_DATA_DIR / "resolution" / "philips" / "IM-0004-0002.dcm"
        )
        self.dcm = dcmread(self.file)

    def test_version(self):
        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        with pyproject_path.open("rb") as fp:
            pyproject_data = tomllib.load(fp)

        expected_version = pyproject_data["project"]["version"]
        self.assertIn(
            hazenlib.__version__,
            {expected_version, expected_version + "+dev"},
            (
                f"Package version {hazenlib.__version__} does not match"
                f" pyproject.toml version {expected_version}"
            ),
        )

    def test1_logger(self):
        path = str(TEST_DATA_DIR / "resolution" / "RESOLUTION")
        sys.argv = ["hazen", "spatial_resolution", path, "--log", "warning"]

        hazenlib.main()

        logging = hazenlib.logging

        self.assertEqual(logging.root.level, logging.WARNING)

    def test2_logger(self):
        path = str(TEST_DATA_DIR / "resolution" / "RESOLUTION")
        sys.argv = ["hazen", "spatial_resolution", path]

        hazenlib.main()

        logging = hazenlib.logging

        self.assertEqual(logging.root.level, logging.INFO)

    def test_snr_measured_slice_width(self):
        path = str(TEST_DATA_DIR / "snr" / "GE")
        files = get_dicom_files(path)
        snr_task = SNR(input_data=files, report=False, measured_slice_width=5)
        result = snr_task.run()

        measurements = [
            Measurement(
                name="SNR",
                subtype="subtraction",
                value=183.97,
                type="measured",
                visibility="intermediate",
            ),
            Measurement(
                name="SNR",
                subtype="subtraction",
                value=1518.61,
                type="normalised",
            ),
            Measurement(
                name="SNR",
                subtype="smoothing",
                value=184.41,
                type="measured",
                description="SNR_SAG_MEAS1_23_1",
                visibility="intermediate",
            ),
            Measurement(
                name="SNR",
                subtype="smoothing",
                value=1522.17,
                type="normalised",
                description="SNR_SAG_MEAS1_23_1",
            ),
            Measurement(
                name="SNR",
                subtype="smoothing",
                value=189.38,
                type="measured",
                description="SNR_SAG_MEAS2_24_1",
                visibility="intermediate",
            ),
            Measurement(
                name="SNR",
                subtype="smoothing",
                value=1563.2,
                type="normalised",
                description="SNR_SAG_MEAS2_24_1",
            ),
        ]

        dict1 = Result(
            task="SNR",
            files=["SNR_SAG_MEAS1_23_1", "SNR_SAG_MEAS2_24_1"],
        )

        for m in measurements:
            dict1.add_measurement(m)

        self.assertEqual(vars(result).keys(), vars(dict1).keys())
        for k, v in vars(result).items():
            # Metadata is dynamically populated and is tested elsewhere.
            if k != "measurement" and k != "metadata":
                self.assertEqual(v, vars(dict1)[k])

        for m_d in dict1.measurements:
            m_r = result.get_measurement(
                name=m_d.name,
                measurement_type=m_d.type,
                subtype=m_d.subtype,
                description=m_d.description,
                unit=m_d.unit,
            )[0]
            self.assertAlmostEqual(m_r.value, m_d.value)

    def test_relaxometry(self):
        path = str(
            TEST_DATA_DIR / "relaxometry" / "T1" / "site3_ge" / "plate4"
        )
        files = get_dicom_files(path)
        relaxometry_task = Relaxometry(input_data=files, report=False)
        result = relaxometry_task.run(calc="T1", plate_number=4, verbose=False)

        dict1 = Result(
            task="Relaxometry",
            files="Spin_Echo_34_2_4_t1",
        )

        measurement = Measurement(
            name="Relaxometry",
            type="measured",
            subtype="rms_frac_time_difference",
            value=0.135,
        )
        dict1.add_measurement(measurement)
        self.assertEqual(vars(dict1).keys(), vars(result).keys())
        self.assertAlmostEqual(
            measurement.value,
            result.get_measurement(name="Relaxometry")[0].value,
            4,
        )


class TestShowOption(unittest.TestCase):
    """Check when main() opens the report images."""

    def setUp(self):
        self.result = Result(task="Task", _load_metadata=False)
        self.result.add_report_image(["a.png", "b.png"])
        task = mock.Mock()
        task.run.return_value = self.result

        protocol = mock.Mock()
        protocol.run.return_value.results = [self.result, self.result]

        patches = [
            mock.patch("hazenlib.get_dicom_files", return_value=["f1", "f2"]),
            mock.patch("hazenlib.init_task", return_value=task),
            mock.patch(
                "hazenlib.ACRLargePhantomProtocol", return_value=protocol
            ),
            mock.patch("hazenlib.write_result"),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        opener = mock.patch("hazenlib.open_report_images")
        self.open_report_images = opener.start()
        self.addCleanup(opener.stop)

    def run_main(self, *argv):
        with mock.patch.object(sys, "argv", ["hazen", *argv]):
            hazenlib.main()

    def test_show(self):
        image_pair = ["a.png", "b.png"]
        cases = {
            ("acr_uniformity", "f"): [image_pair],
            ("acr_uniformity", "f", "--no-show"): [],
            ("acr_uniformity", "f", "--no-report"): [],
            ("acr_all", "t1", "t2", "sag"): [],
            ("acr_all", "t1", "t2", "sag", "--show"): [image_pair * 2],
        }
        for argv, expected in cases.items():
            with self.subTest(argv=argv):
                self.open_report_images.reset_mock()
                self.run_main(*argv)
                opened = [
                    list(c.args[0])
                    for c in self.open_report_images.call_args_list
                ]
                self.assertEqual(opened, expected)

    def test_show_with_no_report_warns(self):
        with self.assertLogs(level="WARNING") as logs:
            self.run_main("acr_uniformity", "f", "--no-report", "--show")
        self.assertTrue(any("--no-report" in m for m in logs.output))
