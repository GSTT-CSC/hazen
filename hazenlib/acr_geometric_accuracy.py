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


def hori_length(mask,centroid,res):
    dims = mask.shape
    start_h = (centroid[1], 0)
    end_h = (centroid[1], dims[0]-1)
    line_profile_h = skimage.measure.profile_line(mask, start_h, end_h)
    extent_h = np.nonzero(line_profile_h)[0]
    dist_h = (extent_h[-1]-extent_h[0])*res[0]
    return dist_h


def vert_length(mask,centroid,res):
    dims = mask.shape
    start_v = (0, centroid[0])
    end_v = (dims[1]-1, centroid[0])
    line_profile_v = skimage.measure.profile_line(mask, start_v, end_v)
    extent_v = np.nonzero(line_profile_v)[0]
    dist_v = (extent_v[-1]-extent_v[0])*res[1]
    return dist_v


def rot_matrix(theta):
    theta = np.radians(theta)
    c,s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    return R


def geo_accuracy_slice1(dcm):
    img = dcm.pixel_array
    cxy = centroid_com(img)
    thresh_img = img > 0.25 * np.max(img)
    open_img = skimage.morphology.area_opening(thresh_img, area_threshold=500)
    bhull = skimage.morphology.convex_hull_image(open_img)

    res = dcm.PixelSpacing  # In-plane resolution from metadata

    L = [hori_length(bhull,cxy,res), vert_length(bhull,cxy,res)]
    return L


def geo_accuracy_slice5(dcm):
    img = dcm.pixel_array
    dims = img.shape
    cxy = centroid_com(img)
    thresh_img = img > 0.25 * np.max(img)
    open_img = skimage.morphology.area_opening(thresh_img, area_threshold=500)
    bhull = skimage.morphology.convex_hull_image(open_img)

    res = dcm.PixelSpacing  # In-plane resolution from metadata

    dist_h, dist_v = [hori_length(bhull,cxy,res), vert_length(bhull,cxy,res)]
    rot_matrix_sw = rot_matrix(45)
    rot_matrix_se = rot_matrix(90)

    coords = np.arange(1,dims[1],1)-cxy[0], [cxy[1]]*dims[0]-cxy[1]
    rot_coords_sw = np.matmul(coords, rot_matrix_sw, dtype=object)

    y_sw = rot_coords_sw[0][0]
    x_sw = rot_coords_sw[1][0]

    start_sw = (cxy[0] + x_sw[0], cxy[1] + y_sw[0])
    end_sw = (cxy[0] + x_sw[-1], cxy[1] + y_sw[-1])
    line_profile_sw = skimage.measure.profile_line(bhull, start_sw, end_sw)

    rot_coords_se = np.matmul(rot_coords_sw, rot_matrix_se, dtype=object)

    y_se = rot_coords_se[0][0]
    x_se = rot_coords_se[1][0]

    start_se = (cxy[0] + x_se[0], cxy[1] + y_se[0])
    end_se = (cxy[0] + x_se[-1], cxy[1] + y_se[-1])
    line_profile_se = skimage.measure.profile_line(bhull, start_se, end_se)

    extent_se = np.nonzero(line_profile_se)[0]
    extent_sw = np.nonzero(line_profile_sw)[0]

    eff_res = np.sqrt(np.mean(np.square(res)))

    dist_se = (extent_se[-1]-extent_se[0])*eff_res
    dist_sw = (extent_sw[-1]-extent_sw[0])*eff_res

    L = dist_h, dist_v, dist_se, dist_sw
    return L


def geo_accuracy(dcm, dcm2, report_path):
    L1 = geo_accuracy_slice1(dcm)
    L5 = geo_accuracy_slice5(dcm2)

    Ltot = L1, L5
    if report_path:
        pass
    return Ltot


def main(data: list, report_path=False) -> dict:

    result = {}

    try:
        z = np.array([data[x].ImagePositionPatient[2] for x in range(len(data))])
    except AttributeError:
        print(data)
        dcm_list = data.PerFrameFunctionalGroupsSequence
        z = np.array([dcm_list[x].PlanePositionSequence[0].ImagePositionPatient[2] for x in range(len(dcm_list))])

    idx_sort = np.argsort(z)
    ind = [idx_sort[0], idx_sort[4]]
    dcm = data[ind[0]]
    dcm2 = data[ind[1]]

    try:
        key = f"{dcm.SeriesDescription}_{dcm.SeriesNumber}_{dcm.InstanceNumber}"
    except AttributeError as e:
        print(e)
        key = f"{dcm.SeriesDescription}_{dcm.SeriesNumber}"

    if report_path:
        report_path = key

    try:
        L = geo_accuracy(dcm, dcm2, report_path)
        print(L)
        result[f"geometric_accuracy_slice1_horizontal_{key}"] = np.round(L[0][0], 2)
        result[f"geometric_accuracy_slice1_vertical_{key}"] = np.round(L[0][1], 2)
        result[f"geometric_accuracy_slice5_horizontal_{key}"] = np.round(L[1][0], 2)
        result[f"geometric_accuracy_slice5_vertical_{key}"] = np.round(L[1][1], 2)
        result[f"geometric_accuracy_slice5_diagonalse2nw_{key}"] = np.round(L[1][2], 2)
        result[f"geometric_accuracy_slice5_diagonalsw2ne_{key}"] = np.round(L[1][3], 2)
    except Exception as e:
        print(f"Could not calculate the geometric accuracy for {key} because of : {e}")
        traceback.print_exc(file=sys.stdout)

    return result


if __name__ == "__main__":
    main([sys.argv[1]])