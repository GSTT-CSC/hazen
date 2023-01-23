import numpy as np


class ACRData:
    def __init__(self, dcm):
        self.dcm = dcm
        self.img_sort()

        def img_sort(self):
            z = [dcm_file.ImagePositionPatient[2] for dcm_file in self.dcm]
            imgs = [np.array(dcm_file.pixel_array) for dcm_file in self.dcm]

            print(z)
            print(imgs)


ACRData()