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

        measurement = {
            "snr by subtraction": {
                "measured": 183.97, "normalised": 1518.61,
            },
            "snr by smoothing": {
                "SNR_SAG_MEAS1_23_1": {
                    "measured": 184.41, "normalised": 1522.17,
                },
                "SNR_SAG_MEAS2_24_1": {
                    "measured": 189.38, "normalised": 1563.2,
                },
            },
        }


        dict1 = Result(
            task="SNR",
            files=["SNR_SAG_MEAS1_23_1", "SNR_SAG_MEAS2_24_1"],
        )

        # Transformation from old-style measurement data to standardized  output
        for k, v in measurement.items():
            for ki, vi in v.items():
                try:
                    for t, val in vi.items():
                        dict1.add_measurement(
                            Measurement(
                                name=k, value=val, type=t, description=ki,
                            ),
                        )
                except AttributeError:
                    dict1.add_measurement(
                        Measurement(
                            name=k, value=np.float64(vi), type=ki,
                        ),
                    )

        self.assertEqual(vars(result).keys(), vars(dict1).keys())
        for k, v in vars(result).items():
            if k != "measurement":
                self.assertEqual(v, vars(dict1)[k])

        for m_d in dict1.measurements:
            m_r = result.get_measurement(
                name=m_d.name,
                measurement_type=m_d.type,
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

        dict1.add_measurement(
            Measurement(name="rms_frac_time_difference", value=0.135),
        )
        self.assertEqual(vars(dict1).keys(), vars(result).keys())
        self.assertAlmostEqual(
            dict1.get_measurement("rms_frac_time_difference")[0].value,
            result.get_measurement("rms_frac_time_difference")[0].value,
            4,
        )
