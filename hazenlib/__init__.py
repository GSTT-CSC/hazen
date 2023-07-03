
import numpy as np
import pydicom
from hazenlib._version import __version__


EXCLUDED_FILES = ['.DS_Store']


def rescale_to_byte(array):
    image_histogram, bins = np.histogram(array.flatten(), 255)
    cdf = image_histogram.cumsum()  # cumulative distribution function
    cdf = 255 * cdf / cdf[-1]  # normalize
    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(array.flatten(), bins[:-1], cdf)
    return image_equalized.reshape(array.shape).astype('uint8')


def is_enhanced_dicom(dcm: pydicom.Dataset) -> bool:
    """

    Parameters
    ----------
    dcm

    Returns
    -------
    bool

    Raises
    ------
    Exception
     Unrecognised SOPClassUID

    """

    if dcm.SOPClassUID == '1.2.840.10008.5.1.4.1.1.4.1':
        return True
    elif dcm.SOPClassUID == '1.2.840.10008.5.1.4.1.1.4':
        return False
    else:
        raise Exception('Unrecognised SOPClassUID')


def get_manufacturer(dcm: pydicom.Dataset) -> str:
    supported = ['ge', 'siemens', 'philips', 'toshiba', 'canon']
    manufacturer = dcm.Manufacturer.lower()
    for item in supported:
        if item in manufacturer:
            return item

    raise Exception(f'{manufacturer} not recognised manufacturer')


def get_average(dcm: pydicom.Dataset) -> float:
    if is_enhanced_dicom(dcm):
        averages = dcm.SharedFunctionalGroupsSequence[0].MRAveragesSequence[0].NumberOfAverages
    else:
        averages = dcm.NumberOfAverages

    return averages


def get_bandwidth(dcm: pydicom.Dataset) -> float:
    """
    Returns PixelBandwidth

    Parameters
    ----------
    dcm: pydicom.Dataset

    Returns
    -------
    bandwidth: float
    """
    bandwidth = dcm.PixelBandwidth
    return bandwidth


def get_num_of_frames(dcm: pydicom.Dataset) -> int:
    """
    Returns number of frames of dicom object

    Parameters
    ----------
    dcm: pydicom.Dataset
        DICOM object

    Returns
    -------

    """
    if len(dcm.pixel_array.shape) > 2:
        return dcm.pixel_array.shape[0]
    elif len(dcm.pixel_array.shape) == 2:
        return 1


def get_slice_thickness(dcm: pydicom.Dataset) -> float:
    if is_enhanced_dicom(dcm):
        try:
            slice_thickness = dcm.PerFrameFunctionalGroupsSequence[0].PixelMeasuresSequence[0].SliceThickness
        except AttributeError:
            slice_thickness = dcm.PerFrameFunctionalGroupsSequence[0].Private_2005_140f[0].SliceThickness
        except Exception:
            raise Exception('Unrecognised metadata Field for Slice Thickness')
    else:
        slice_thickness = dcm.SliceThickness

    return slice_thickness


def get_pixel_size(dcm: pydicom.Dataset) -> (float, float):
    manufacturer = get_manufacturer(dcm)
    try:
        if is_enhanced_dicom(dcm):
            dx, dy = dcm.PerFrameFunctionalGroupsSequence[0].PixelMeasuresSequence[0].PixelSpacing
        else:
            dx, dy = dcm.PixelSpacing
    except:
        print('Warning: Could not find PixelSpacing..')
        if 'ge' in manufacturer:
            fov = get_field_of_view(dcm)
            dx = fov / dcm.Columns
            dy = fov / dcm.Rows
        else:
            raise Exception('Manufacturer not recognised')

    return dx, dy


def get_TR(dcm: pydicom.Dataset) -> (float):
    """
    Returns Repetition Time (TR)

    Parameters
    ----------
    dcm: pydicom.Dataset

    Returns
    -------
    TR: float
    """

    try:
        TR = dcm.RepetitionTime
    except:
        print('Warning: Could not find Repetition Time. Using default value of 1000 ms')
        TR = 1000
    return TR


def get_rows(dcm: pydicom.Dataset) -> (float):
    """
    Returns number of image rows (rows)

    Parameters
    ----------
    dcm: pydicom.Dataset

    Returns
    -------
    rows: float
    """
    try:
        rows = dcm.Rows
    except:
        print('Warning: Could not find Number of matrix rows. Using default value of 256')
        rows = 256

    return rows


def get_columns(dcm: pydicom.Dataset) -> (float):
    """
    Returns number of image columns (columns)

    Parameters
    ----------
    dcm: pydicom.Dataset

    Returns
    -------
    columns: float
    """
    try:
        columns = dcm.Columns
    except:
        print('Warning: Could not find matrix size (columns). Using default value of 256.')
        columns = 256
    return columns


def get_field_of_view(dcm: pydicom.Dataset):
    # assumes square pixels
    manufacturer = get_manufacturer(dcm)

    if 'ge' in manufacturer:
        fov = dcm[0x19, 0x101e].value
    elif 'siemens' in manufacturer:
        fov = dcm.Columns * dcm.PixelSpacing[0]
    elif 'philips' in manufacturer:
        if is_enhanced_dicom(dcm):
            fov = dcm.Columns * dcm.PerFrameFunctionalGroupsSequence[0].PixelMeasuresSequence[0].PixelSpacing[0]
        else:
            fov = dcm.Columns * dcm.PixelSpacing[0]
    elif 'toshiba' in manufacturer:
        fov = dcm.Columns * dcm.PixelSpacing[0]
    else:
        raise NotImplementedError('Manufacturer not ge,siemens, toshiba or philips so FOV cannot be calculated.')

    return fov


def parse_relaxometry_data(task, arguments, dicom_objects,
                           report):  # def parse_relaxometry_data(arguments, dicom_objects, report):   #

    # Relaxometry arguments
    relaxometry_cli_args = {'--calc_t1', '--calc_t2', '--plate_number',
                            '--show_template_fit', '--show_relax_fits',
                            '--show_rois', '--verbose'}

    # Pass arguments with dictionary, stripping initial double dash ('--')
    relaxometry_args = {}

    for key in relaxometry_cli_args:
        relaxometry_args[key[2:]] = arguments[key]

    return task.main(dicom_objects, report_path=report,
                     **relaxometry_args)


