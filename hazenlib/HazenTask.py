"""
HazenTask.py
"""

from pydicom import dcmread
from hazenlib.logger import logger


class HazenTask:

    def __init__(self, data_paths: list, report: bool = False):
        self.data_paths = data_paths
        self.report: bool = report
        self.report_path = self.key(self.data[0])

    @property
    def data(self) -> list:
        return [dcmread(dicom)for dicom in self.data_paths]

    def key(self, dcm, properties=None) -> str:
        if properties is None:
            properties = ['SeriesDescription', 'SeriesNumber', 'InstanceNumber']
        try:
            metadata = [str(dcm.get(field)) for field in properties]
        except KeyError:
            logger.warning(f"Could not find one or more of the following properties: {properties}")
            metadata = [str(dcm.get(field)) for field in ['SeriesDescription', 'SeriesNumber']]

        key = f"{type(self).__name__}_" + '_'.join(metadata).replace(' ', '_')
        self.report_path = key
        return key
