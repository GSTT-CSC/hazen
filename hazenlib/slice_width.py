
import pydicom

import hazen.snr
import tests


def main(image):

    cenx, ceny, cradius = hazen.snr.find_circle(image)


if __name__ == "__main__":
    dcm = tests.TEST_DATA_DIR / 'snr' / 'uniform-circle.IMA'
    main(pydicom.read_file(dcm))
