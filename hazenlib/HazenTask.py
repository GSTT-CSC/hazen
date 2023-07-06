"""
HazenTask.py
"""

from pydicom import dcmread
from hazenlib.logger import logger
import pathlib
import os


class HazenTask:
    # parent class for all tasks available in Hazen

    def __init__(self,
                data_paths: list,
                report: bool = False,
                report_dir: str = os.path.join(os.getcwd(), 'reports')):
        self.data_paths = sorted(data_paths) # could be sorted upfront in main
        self.report: bool = report
        self.report_path = os.path.join(report_dir, type(self).__name__)
        # if report is requested, create output folder if does not exist yet
        if report:
            pathlib.Path(self.report_path).mkdir(parents=True, exist_ok=True)
        else:
            pass
        # placeholder for report output files
        self.report_files = []

    @property
    def data(self) -> list:
        return [dcmread(dicom)for dicom in self.data_paths]

    def key(self, dcm,
            properties=['SeriesDescription', 'SeriesNumber', 'InstanceNumber']
            ) -> str:
        """Creates a key from DICOM metadata

        Args:
            dcm (DICOM): a DICOM file
            properties (list, optional): list of DICOM metadata fields.
                Defaults to ['SeriesDescription', 'SeriesNumber', 'InstanceNumber'].

        Returns:
            str: an underscore concatenated string of the metadata fields
        """

        try:
            metadata = [str(dcm.get(field)) for field in properties]
        except KeyError:
            logger.warning(f"Could not find one or more of the following properties: {properties}")
            metadata = [str(dcm.get(field)) for field in ['SeriesDescription', 'SeriesNumber']]

        key = f"{type(self).__name__}_" + '_'.join(metadata).replace(' ', '_')
        return key
