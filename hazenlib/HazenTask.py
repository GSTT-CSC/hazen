"""Base class for Hazen tasks."""

# Typing imports
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pydicom

# Python imports
from pathlib import Path

# Module imports
from pydicom import dcmread

# Local imports
from hazenlib.logger import logger
from hazenlib.types import Result
from hazenlib.utils import scrub, REGEX_SCRUBNAME


class HazenTask:
    """Base class for performing tasks on image sets."""

    def __init__(
        self,
        input_data: Sequence,
        *,
        verbose: bool = False,
        report: bool = False,
        report_dir: str | Path | None = None,
    ) -> None:
        """Initialise a HazenTask instance.

        Args:
            input_data : list of filepaths to DICOM images
            report : Whether to create measurement visualisation diagrams.
                     Defaults to False.
            report_dir : Path to output report images. Defaults to None.

        """
        data_paths = sorted(input_data)
        self.dcm_list = [dcmread(dicom) for dicom in data_paths]

        # Log acquisition information for each DICOM file
        for dcm in self.dcm_list:
            acq_number = dcm.get("AcquisitionNumber", "N/A")
            series_desc = dcm.get("SeriesDescription", "N/A")
            series_number = dcm.get("SeriesNumber", "N/A")
            instance_number = dcm.get("InstanceNumber", "N/A")
            logger.info(
                "Loaded DICOM - AcquisitionNumber: %s, "
                "SeriesDescription: %s, SeriesNumber: %s, "
                "InstanceNumber: %s",
                acq_number,
                series_desc,
                series_number,
                instance_number,
            )

        self.report: bool = report
        self.report_path = (
            Path().cwd() / "report_image" / type(self).__name__
            if report_dir is None
            else Path(report_dir) / type(self).__name__
        )

        self.report_path.mkdir(parents=True, exist_ok=True)
        self.report_files: Sequence[str] = []

    def init_result_dict(self, desc: str = "", files: tuple = ()) -> Result:
        """Initialise measurement results holder and input description.

        Returns
            d : holds measurement results and task input description.

        """
        return Result(task=type(self).__name__, desc=desc, files=files)

    def img_desc(
        self,
        dcm: pydicom.Dataset,
        properties: Sequence | None = None,
    ) -> str:
        """Obtain values from the DICOM header to identify input series.

        Args:
            dcm : DICOM image object
            properties : list of DICOM header field names supported by pydicom
                that shuld be used to generate sereis identifier.
                Defaults to None.

        Returns:
            str: contatenation of the specified DICOM header property values

        """
        if properties is None:
            properties = [
                "SeriesDescription",
                "SeriesNumber",
                "InstanceNumber",
            ]
        try:
            metadata = [str(dcm.get(field)) for field in properties]
        except KeyError:
            logger.warning(
                f"Could not find one or more of the following "
                f"properties: {properties}",
            )
            metadata = [
                str(dcm.get(field))
                for field in ["SeriesDescription", "SeriesNumber"]
            ]

        join_char = "_"
        img_desc = join_char.join(metadata).replace(" ", join_char)
        # Let's make sure dirty names do not contaminate the file names
        # or any other string operations.
        return (
            scrub(
                img_desc,
                REGEX_SCRUBNAME,
                join_char,
            )
            .strip(join_char)
            .replace(join_char * 2, join_char)
        )
