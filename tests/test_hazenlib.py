import sys
import unittest

import hazenlib
import numpy as np
import pydicom
from hazenlib.tasks.relaxometry import Relaxometry
from hazenlib.tasks.snr import SNR
from hazenlib.types import Measurement, Result
from hazenlib.utils import get_dicom_files, is_dicom_file

from tests import TEST_DATA_DIR, TEST_REPORT_DIR


class TestCliParser(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        self.file = str(TEST_DATA_DIR / "resolution" / "philips" / "IM-0004-0002.dcm")
        self.dcm = pydicom.dcmread(self.file)

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
            if k != "measurement":
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
        path = str(TEST_DATA_DIR / "relaxometry" / "T1" / "site3_ge" / "plate4")
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
