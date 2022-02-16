"""
ACR Geometric Accuracy

Calculates end-to-end lengths for slices 1 and 5 of the ACR phantom. The results are also visualised.

Created by Yassine Azma
yassine.azma@rmh.nhs.uk

16/02/2022
"""

import sys
import traceback
import numpy as np
import skimage.morphology
import skimage.measure


def centroid_com(dcm):
    # Calculate centroid of object using a centre-of-mass calculation
    thresh_img = dcm > 0.25 * np.max(dcm)
    open_img = skimage.morphology.area_opening(thresh_img, area_threshold=500)
    bhull = skimage.morphology.convex_hull_image(open_img)
    coords = np.nonzero(bhull)  # row major - first array is columns

    sum_x = np.sum(coords[0])
    sum_y = np.sum(coords[1])
    cxy = sum_x / coords[0].shape, sum_y / coords[1].shape

    cxy = [cxy[0].astype(int), cxy[1].astype(int)]
    return cxy

def geo_accuracy_slice1(dcm,res):
    img = dcm.pixel_array
    cxy = centroid_com(img)
    thresh_img = img > 0.25 * np.max(img)
    open_img = skimage.morphology.area_opening(thresh_img, area_threshold=500)
    bhull = skimage.morphology.convex_hull_image(open_img)

    res = dcm.PixelSpacing  # In-plane resolution from metadata
    start_v = (cxy[1], 0)
    end_v = (cxy[1], 255)
    start_h = (0, cxy[0])
    end_h = (255, cxy[0])
    line_profile_v = skimage.measure.profile_line(bhull, start_v, end_v)
    line_profile_h = skimage.measure.profile_line(bhull, start_h, end_h)

    insert_extent_v = np.nonzero(line_profile_v)
    insert_extent_h = np.nonzero(line_profile_h)

    insert_dist_v = (insert_extent_v[-1]-insert_extent_v[0])*res[1]
    insert_dist_h = (insert_extent_h[-1]-insert_extent_h[0])*res[0]

    L = [insert_dist_v,insert_dist_h]
    print(insert_extent_h)

def main(data: list, report_path=False) -> dict:

    result = {}

    try:
        z = np.array([data[x].ImagePositionPatient[2] for x in range(len(data))])
    except AttributeError:
        print(data)
        dcm_list = data.PerFrameFunctionalGroupsSequence
        z = np.array([dcm_list[x].PlanePositionSequence[0].ImagePositionPatient[2] for x in range(len(dcm_list))])

    idx_sort = np.argsort(z)
    ind = [idx_sort[0], idx_sort[5]]
    dcm = data[ind[0]]

    try:
        key = f"{dcm.SeriesDescription}_{dcm.SeriesNumber}_{dcm.InstanceNumber}"
    except AttributeError as e:
        print(e)
        key = f"{dcm.SeriesDescription}_{dcm.SeriesNumber}"

    if report_path:
        report_path = key

    try:
        L = geo_accuracy_slice1(dcm, report_path)
        result[f"geo_accuracy_slice1_{key}"] = round(L, 2)
    except Exception as e:
        print(f"Could not calculate the geometric accuracy for {key} because of : {e}")
        traceback.print_exc(file=sys.stdout)

    return result


if __name__ == "__main__":
    main([sys.argv[1]])