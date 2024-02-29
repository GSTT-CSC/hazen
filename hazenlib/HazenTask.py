"""
HazenTask.py
"""
import os
import pathlib
from pydicom import dcmread

from hazenlib.logger import logger


class HazenTask:
    """Base class for performing tasks on image sets"""

    def __init__(
        self, input_data: list, report: bool = False, report_dir=None, **kwargs
    ):
        """Initialise a HazenTask instance

        Args:
            input_data (list): list of filepaths to DICOM images
            report (bool, optional): Whether to create measurement visualisation diagrams. Defaults to False.
            report_dir (string, optional): Path to output report images. Defaults to None.
        """
        data_paths = sorted(input_data)
        self.dcm_list = [dcmread(dicom) for dicom in data_paths]
        self.report: bool = report
        if report_dir is not None:
            self.report_path = os.path.join(str(report_dir), type(self).__name__)
        else:
            self.report_path = os.path.join(
                os.getcwd(), "report_image", type(self).__name__
            )
        if report:
            pathlib.Path(self.report_path).mkdir(parents=True, exist_ok=True)
        else:
            pass
        self.report_files = []

    def init_result_dict(self) -> dict:
        """Initialise the dictionary that holds measurement results and input description

        Returns:
            dict: holds measurement results and task input description
        """
        result_dict = {
            "task": f"{type(self).__name__}",
            "file": None,
            "measurement": {},
        }
        return result_dict

    def img_desc(self, dcm, properties=None) -> str:
        """Obtain values from the DICOM header to identify input series

        Args:
            dcm (pydicom.Dataset): DICOM image object
            properties (list, optional): list of DICOM header field names supported by pydicom
                that shuld be used to generate sereis identifier. Defaults to None.

        Returns:
            str: contatenation of the specified DICOM header property values
        """
        if properties is None:
            properties = ["SeriesDescription", "SeriesNumber", "InstanceNumber"]
        try:
            metadata = [str(dcm.get(field)) for field in properties]
        except KeyError:
            logger.warning(
                f"Could not find one or more of the following properties: {properties}"
            )
            metadata = [
                str(dcm.get(field)) for field in ["SeriesDescription", "SeriesNumber"]
            ]

        img_desc = "_".join(metadata).replace(" ", "_")
        return img_desc
