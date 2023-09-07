"""
HazenTask.py
"""

from pydicom import dcmread
from hazenlib.logger import logger
import pathlib
import os


class HazenTask:

    def __init__(self, input_data, report: bool = False, report_dir: str = os.path.join(os.getcwd(), 'report')):
        # Check if input data is a single or list of files, load accordingly
        if isinstance(input_data, list):
            data_paths = sorted(input_data)
            self.dcm_list = [dcmread(dicom)for dicom in data_paths]
        else:
            self.single_dcm = dcmread(input_data)
        self.report: bool = report
        self.report_path = os.path.join(report_dir, type(self).__name__)
        if report:
            pathlib.Path(self.report_path).mkdir(parents=True, exist_ok=True)
        else:
            pass
        self.report_files = []

    def init_result_dict(self) -> dict:
        result_dict = {"task": f"{type(self).__name__}"}
        return result_dict

    def key(self, dcm, properties=None) -> str:
        if properties is None:
            properties = ['SeriesDescription', 'SeriesNumber', 'InstanceNumber']
        try:
            metadata = [str(dcm.get(field)) for field in properties]
        except KeyError:
            logger.warning(f"Could not find one or more of the following properties: {properties}")
            metadata = [str(dcm.get(field)) for field in ['SeriesDescription', 'SeriesNumber']]

        key =  '_'.join(metadata).replace(' ', '_')
        return key
