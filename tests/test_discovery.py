"""Tests for the discover.py module."""

# ruff: noqa: PT009

# Python imports
import unittest
from pathlib import Path

# Module imports
# Local imports
from hazenlib._version import __version__
from hazenlib.discovery import DiscoveredAcquisition, generate_batch_config
from hazenlib.orchestration import BatchConfig, JobTaskConfig
from tests import TEST_DATA_DIR


class TestGenerateBatchConfig(unittest.TestCase):
    """Test the generated_batch_config function."""

    PATH: Path = TEST_DATA_DIR / "batch_guess"

    @classmethod
    def setUpClass(cls) -> None:
        """Set up the class."""
        if not cls.PATH.exists():
            msg = f"Required path does not exist: {cls.PATH}"
            raise FileNotFoundError(msg)

        ########
        # Jobs #
        ########

        jobs = [
            # ARC ALL #
            JobTaskConfig(
                task="acr_all",
                folders=[
                    cls.PATH / "ACR_T1_2",
                    cls.PATH / "ACR_T2_1",
                    cls.PATH / "ACR_SAG_LOCALISER_2",
                ],
            ),
            # SNR BODY #
            JobTaskConfig(
                task="snr",
                folders=[cls.PATH / "SNR_BODY_AX_ACQ1"],
                overrides={"subtract": cls.PATH / "SNR_BODY_AX_ACQ2"},
            ),
            JobTaskConfig(
                task="snr",
                folders=[cls.PATH / "SNR_BODY_COR_ACQ1"],
                overrides={"subtract": cls.PATH / "SNR_BODY_COR_ACQ2"},
            ),
            JobTaskConfig(
                task="snr",
                folders=[cls.PATH / "SNR_BODY_SAG_ACQ1"],
                overrides={"subtract": cls.PATH / "SNR_BODY_SAG_ACQ2"},
            ),
            # SNR BREAST #
            JobTaskConfig(
                task="snr",
                folders=[cls.PATH / "SNR_BREAST_ACQ1"],
                overrides={"subtract": cls.PATH / "SNR_BREAST_ACQ2"},
            ),
            # SNR HEADNECK #
            JobTaskConfig(
                task="snr",
                folders=[cls.PATH / "SNR_HEADNECK_ACQ1"],
                overrides={"subtract": cls.PATH / "SNR_HEADNECK_ACQ2"},
            ),
            # SNR KNEE #
            JobTaskConfig(
                task="snr",
                folders=[cls.PATH / "SNR_KNEE_ACQ1"],
                overrides={"subtract": cls.PATH / "SNR_KNEE_ACQ2"},
            ),
        ]

        for job in jobs:
            for folder in job.folders:
                if not folder.exists():
                    msg = f"Required path does not exist: {folder}"
                    raise FileNotFoundError(msg)
            if (
                job.overrides
                and (sub_dir := job.overrides.get("subtract"))
                and not sub_dir.exists()
            ):
                msg = f"Required path does not exist: {sub_dir}"
                raise FileNotFoundError(msg)

        ###################################
        # True Batch Configuration Object #
        ###################################

        cls.batch_config = BatchConfig(
            version=BatchConfig._CURRENT_BATCHCONFIG_VERSION,  # noqa: SLF001
            hazen_version_constraint=f">={__version__}",
            description=(
                "Batch configuration file automatically generated"
                f" from {cls.PATH.as_posix()}"
            ),
            jobs=jobs,
            output=cls.PATH.parent / "hazen_output.json",
            report_docx=cls.PATH.parent / "hazen_report.docx",
            report_template=None,
            levels=("final", "all"),
            defaults={},
            _file=cls.PATH.parent / "hazen_batch_config.yml",
        )

        #####################################
        # Loaded Batch Configuration Object #
        #####################################

        cls.generated_batch_config = generate_batch_config(cls.PATH)

    def test_generate_batch_config(self) -> None:
        """Test the generate batch config."""
        self.assertTrue(generate_batch_config(self.PATH))

    def test_version(self) -> None:
        """Test Batch Config version."""
        self.assertEqual(
            self.batch_config.version,
            self.generated_batch_config.version,
        )

    def test_hazen_version_constraint(self) -> None:
        """Test Batch Config hazen version contraint."""
        self.assertEqual(
            self.batch_config.hazen_version_constraint,
            self.generated_batch_config.hazen_version_constraint,
        )

    def test_description(self) -> None:
        """Test Batch Config description."""
        self.assertEqual(
            self.batch_config.description,
            self.generated_batch_config.description,
        )

    def test_jobs(self) -> None:
        """Test Batch Config jobs."""

        def normalise_job(job: JobTaskConfig) -> tuple[tuple]:
            """Convert job into a hashable tuple."""
            return (
                job.task,
                sorted(job.folders),
                tuple(sorted(job.overrides.items())) if job.overrides else (),
            )

        for task in {j.task for j in self.batch_config.jobs}:
            expected = [
                normalise_job(j)
                for j in self.batch_config.jobs
                if j.task == task
            ]
            actual = [
                normalise_job(j)
                for j in self.generated_batch_config.jobs
                if j.task == task
            ]
            self.assertCountEqual(
                expected,
                actual,
                f"Jobs not equal for task: {task}",
            )

    def test_output(self) -> None:
        """Test Batch Config output."""
        self.assertEqual(
            self.batch_config.output,
            self.generated_batch_config.output,
        )

    def test_report_docx(self) -> None:
        """Test Batch Config report_docx."""
        self.assertEqual(
            self.batch_config.report_docx,
            self.generated_batch_config.report_docx,
        )

    def test_report_template(self) -> None:
        """Test Batch Config report_template."""
        self.assertEqual(
            self.batch_config.report_template,
            self.generated_batch_config.report_template,
        )

    def test_levels(self) -> None:
        """Test Batch Config levels."""
        self.assertEqual(
            self.batch_config.levels,
            self.generated_batch_config.levels,
        )

    def test_defaults(self) -> None:
        """Test Batch Config defaults."""
        self.assertEqual(
            self.batch_config.defaults,
            self.generated_batch_config.defaults,
        )


class TestIsLikelySNR(unittest.TestCase):
    """Tests for difflib-based fuzzy SNR matching."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up the class."""
        cls.is_snr = staticmethod(
            DiscoveredAcquisition._is_likely_snr,  # noqa: SLF001
        )

    def test_fuzzy_matches(self) -> None:
        """Test similarity threshold catches transpositions."""
        # These all have SequenceMatcher ratio >= 0.66 with "snr"
        fuzzy_typos = ["srn", "nsr", "smr"]  # smr is close on keyboard
        for typo in fuzzy_typos:
            with self.subTest(typo=typo):
                result = self.is_snr(typo)
                self.assertTrue(result, f"Should detect {typo} as SNR-like")

        not_snr = [
            "diffusion",
            "phase",
            "sag",
            "axial",
            "coronal",
            "soon",
        ]
        for seq in not_snr:
            with self.subTest(typo=seq):
                result = self.is_snr(seq)
                self.assertFalse(
                    result,
                    f"Should not detect {seq} as SNR-like",
                )


if __name__ == "__main__":
    unittest.main()
