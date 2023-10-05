"""
HazenTask.py
"""

from pydicom import dcmread
from hazenlib.logger import logger
import pathlib
import os


class HazenTask:

    def __init__(self, input_data: list, report: bool = False, report_dir=None, **kwargs):
        data_paths = sorted(input_data)
        self.dcm_list = [dcmread(dicom)for dicom in data_paths]
        self.report: bool = report
        if report_dir is not None:
            self.report_path = os.path.join(str(report_dir), type(self).__name__)
        else:
            self.report_path = os.path.join(os.getcwd(), 'report_image',
                                            type(self).__name__)
        if report:
            pathlib.Path(self.report_path).mkdir(parents=True, exist_ok=True)
        else:
            pass
        self.report_files = []

    def init_result_dict(self) -> dict:
        result_dict = {
            "task": f"{type(self).__name__}",
            "file": None,
            "measurement": {}
        }
        return result_dict

    def img_desc(self, dcm, properties=None) -> str:
        if properties is None:
            properties = ['SeriesDescription', 'SeriesNumber', 'InstanceNumber']
        try:
            metadata = [str(dcm.get(field)) for field in properties]
        except KeyError:
            logger.warning(f"Could not find one or more of the following properties: {properties}")
            metadata = [str(dcm.get(field)) for field in ['SeriesDescription', 'SeriesNumber']]

        img_desc = '_'.join(metadata).replace(' ', '_')
        return img_desc
