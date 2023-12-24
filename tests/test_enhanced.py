import os

import pydicom
import unittest
import hazenlib.utils as hazen_tools

# class TestEnhanced(unittest.TestCase):
def main():
    TEST_DATA_DIR = "/Users/sophieratkai/Documents/projects/hazen_docs/ACR_se_enhanced"
    enhanced_folder = os.path.join(TEST_DATA_DIR, 'enhanced')
    folder = "/Users/sophieratkai/Documents/projects/hazen_docs/Philips_DicomTest/Res/Enhanced/DICOM"
    test_files = [os.path.join(TEST_DATA_DIR, file) for file in os.listdir(TEST_DATA_DIR) if not file.endswith(".txt")]
    
    for file in test_files:
        if not hazen_tools.has_pixel_array(file):
            break
        dcm = pydicom.dcmread(file)
        print("#################")
        print(file)
        try:
            print("SOPClassUID", dcm.SOPClassUID)
            print("Manufacturer", dcm.Manufacturer)
            print("rows, columns", dcm.Rows, dcm.Columns)
            pa = dcm.pixel_array # pixel_array (11, 512, 512)
            print("pixel_array", pa.shape)
            pa5 = pa[5]
            print(pa5.shape)
            print(type(pa))
            print("InstanceNumber", dcm.InstanceNumber)
        except:
            print("enhanced tags")
            print("NumberOfFrames", dcm.NumberOfFrames)
            print(dcm.PerFrameFunctionalGroupsSequence[0].keys())
            # print(dcm.PerFrameFunctionalGroupsSequence[0].Private_2005_140f[0].SliceThickness)

        # print("ImageOrientationPatient")
        # try:
        #     print(dcm.ImageOrientationPatient)
        # except:
        #     print("enhanced", dcm.PerFrameFunctionalGroupsSequence[0].PlaneOrientationSequence[0].ImageOrientationPatient)

        # print("ImagePositionPatient")
        # try:
        #     print(dcm.ImagePositionPatient)
        # except:
        #     print("enhanced", dcm.PerFrameFunctionalGroupsSequence[0].PlaneOrientationSequence[0].ImagePositionPatient)

        # print("NumberOfAverages")
        # try:
        #     print(dcm.NumberOfAverages)
        # except:
        #     try:
        #         print("enhanced, Siemens", dcm.PerFrameFunctionalGroupsSequence[0].MRAveragesSequence[0].NumberOfAverages)
        #     except:
        #         print("enhanced, Philips", dcm.SharedFunctionalGroupsSequence[0].MRAveragesSequence[0].NumberOfAverages)

        # print("PixelBandwidth")
        # try:
        #     print(dcm.PixelBandwidth)
        # except:
        #     print("enhanced", dcm.SharedFunctionalGroupsSequence[0].MRImagingModifierSequence[0].PixelBandwidth)

        # print("SliceThickness")
        # try:
        #     print("normal", dcm.SliceThickness)
        # except:
        #     print("enhanced", dcm.PerFrameFunctionalGroupsSequence[0].PixelMeasuresSequence[0].SliceThickness)

        # print("PixelSpacing")
        # try:
        #     print("normal", dcm.PixelSpacing)
        # except:
        #     print("enhanced", dcm.PerFrameFunctionalGroupsSequence[0].PixelMeasuresSequence[0].PixelSpacing)

        # try:
        #     print("RepetitionTime", dcm.RepetitionTime)
        # except:
        #     print("enhanced", dcm.SharedFunctionalGroupsSequence[0].MRTimingAndRelatedParametersSequence[0].RepetitionTime)


    # def setUp(self):
    


if __name__ == '__main__':
    main()
