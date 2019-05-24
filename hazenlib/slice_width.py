
import pydicom

import hazenlib as hazen
import tests


def main(image):

    cenx, ceny, cradius = hazen.find_circle(image)


if __name__ == "__main__":
    dcm = tests.TEST_DATA_DIR / 'snr' / 'uniform-circle.IMA'
    main(pydicom.read_file(dcm))
