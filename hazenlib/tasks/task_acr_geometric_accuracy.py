"""
ACR Geometric Accuracy
https://www.acraccreditation.org/-/media/acraccreditation/documents/mri/largephantomguidance.pdf

Calculates geometric accuracy for slices 1 and 5 of the ACR phantom.

This script calculates the horizontal and vertical lengths of the ACR phantom in Slice 1 in accordance with the ACR Guidance.
This script calculates the horizontal, vertical and diagonal lengths of the ACR phantom in Slice 5 in accordance with the ACR Guidance.
The average distance measurement error, maximum distance measurement error and coefficient of variation of all distance
measurements is reported as recommended by IPEM Report 112, "Quality Control and Artefacts in Magnetic Resonance Imaging".

This is done by first producing a binary mask for each respective slice. Line profiles are drawn with aid of rotation
matrices around the centre of the test object to determine each respective length. The results are also visualised.

Created by Yassine Azma
yassine.azma@rmh.nhs.uk

18/11/2022
"""

import sys
import traceback
import os
import numpy as np
import skimage.morphology
import skimage.measure

from hazenlib.HazenTask import HazenTask

class ACRGeometricAccuracy(HazenTask):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self) -> dict:
        results = {}
        z = []
        for dcm in self.data:
            z.append(dcm.ImagePositionPatient[2])

        idx_sort = np.argsort(z)

        for dcm in self.data:
            if dcm.ImagePositionPatient[2] == z[idx_sort[0]]:
                try:
                    result = self.get_geometric_accuracy_slice1(dcm)
                except Exception as e:
                    print(f"Could not calculate the geometric accuracy for {self.key(dcm)} because of : {e}")
                    traceback.print_exc(file=sys.stdout)
                    continue

                results[self.key(dcm)] = result
            elif dcm.ImagePositionPatient[2] == z[idx_sort[4]]:
                try:
                    result = self.get_geometric_accuracy_slice5(dcm)
                except Exception as e:
                    print(f"Could not calculate the percent-signal ghosting for {self.key(dcm)} because of : {e}")
                    traceback.print_exc(file=sys.stdout)
                    continue

                results[self.key(dcm)] = result

        results['reports'] = {'images': self.report_files}

        return results

    def centroid_com(self, dcm):
        # Calculate centroid of object using a centre-of-mass calculation
        thresh_img = dcm > 0.25 * np.max(dcm)
        open_img = skimage.morphology.area_opening(thresh_img, area_threshold=500)
        bhull = skimage.morphology.convex_hull_image(open_img)
        coords = np.nonzero(bhull)  # row major - first array is columns

        sum_x = np.sum(coords[1])
        sum_y = np.sum(coords[0])
        cxy = sum_x / coords[0].shape, sum_y / coords[1].shape

        cxy = [cxy[0].astype(int), cxy[1].astype(int)]
        return bhull, cxy

    def horizontal_length(self, res, mask, cxy):
        dims = mask.shape
        start_h = (cxy[1],0)
        end_h = (cxy[1], dims[0]-1)
        line_profile_h = skimage.measure.profile_line(mask, start_h, end_h, mode='reflect')
        extent_h = np.nonzero(line_profile_h)[0]
        dist_h = (extent_h[-1] - extent_h[0])*res[0]

        h_dict = {
            'Start': start_h,
            'End': end_h,
            'Extent': extent_h,
            'Distance': dist_h
        }
        return h_dict

    def vertical_length(self, res, mask, cxy):
        dims = mask.shape
        start_v = (0, cxy[0])
        end_v = (dims[1]-1, cxy[0])
        line_profile_v = skimage.measure.profile_line(mask, start_v, end_v, mode='reflect')
        extent_v = np.nonzero(line_profile_v)[0]
        dist_v = (extent_v[-1] - extent_v[0])*res[1]

        v_dict = {
            'Start': start_v,
            'End': end_v,
            'Extent': extent_v,
            'Distance': dist_v
        }
        return v_dict

    def rot_matrix(self, theta):
        theta = np.radians(theta)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)), dtype=object)
        return R

    def ne_sw_length(self, res, mask, cxy):
        dims = mask.shape

        rot_mat_sw = self.rot_matrix(45)
        coords = np.arange(1,dims[1],1) - cxy[0], [cxy[1]]*dims[0]-cxy[1]
        coords = np.array(coords ,dtype=object)

        rot_coords_sw = np.matmul(coords, rot_mat_sw, dtype=object)

        x_sw = rot_coords_sw[0][0]
        y_sw = rot_coords_sw[1][0]

        start_sw = (cxy[0] + x_sw[0], cxy[1] + y_sw[0])
        end_sw = (cxy[0] + x_sw[-1], cxy[1] + y_sw[-1])
        line_profile_sw = skimage.measure.profile_line(mask, start_sw, end_sw, mode='reflect')

        extent_sw = np.nonzero(line_profile_sw)[0]

        eff_res = np.sqrt(np.mean(np.square(res)))
        dist_sw = (extent_sw[-1] - extent_sw[0])*eff_res

        sw_dict = {
            'Start': start_sw,
            'End': end_sw,
            'Extent': extent_sw,
            'Distance': dist_sw
        }
        return sw_dict

    def nw_se_length(self, res, mask, cxy):
        dims = mask.shape

        rot_mat_se = self.rot_matrix(135)
        coords = np.arange(1,dims[1],1) - cxy[0], [cxy[1]]*dims[0]-cxy[1]
        coords = np.array(coords ,dtype=object)

        rot_coords_se = np.matmul(coords, rot_mat_se, dtype=object)

        x_se = rot_coords_se[0][0]
        y_se = rot_coords_se[1][0]

        start_se = (cxy[0] + x_se[0], cxy[0] + y_se[0])
        end_se = (cxy[1] + x_se[-1], cxy[1] + y_se[-1])
        line_profile_se = skimage.measure.profile_line(mask, start_se, end_se, mode='reflect')

        extent_se = np.nonzero(line_profile_se)[0]

        eff_res = np.sqrt(np.mean(np.square(res)))
        dist_se = (extent_se[-1] - extent_se[0])*eff_res

        se_dict = {
            'Start': start_se,
            'End': end_se,
            'Extent': extent_se,
            'Distance': dist_se
        }
        return se_dict

    def get_geometric_accuracy_slice1(self,dcm):
        img = dcm.pixel_array
        res = dcm.PixelSpacing
        mask, cxy = self.centroid_com(img)

        h_dict = self.horizontal_length(res, mask, cxy)
        v_dict = self.vertical_length(res, mask, cxy)

        if self.report:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            fig.set_size_inches(8, 8)
            plt.imshow(img)

            plt.arrow(h_dict['Extent'][0], float(cxy[1]), h_dict['Extent'][-1] - h_dict['Extent'][0], 1, color='blue',length_includes_head=True, head_width=5)
            plt.arrow(float(cxy[0]), v_dict['Extent'][0], 1, v_dict['Extent'][-1] - v_dict['Extent'][0], color='orange',length_includes_head=True, head_width=5)
            plt.legend([str(np.round(h_dict['Distance'], 2)) + 'mm',
                        str(np.round(v_dict['Distance'], 2)) + 'mm'])
            plt.axis('off')
            plt.title('Geometric Accuracy for Slice 1')

            img_path = os.path.realpath(os.path.join(self.report_path, f'{self.key(dcm)}.png'))
            fig.savefig(img_path)
            self.report_files.append(img_path)

        return h_dict['Distance'], v_dict['Distance']

    def get_geometric_accuracy_slice5(self,dcm):
        img = dcm.pixel_array
        res = dcm.PixelSpacing
        mask, cxy = self.centroid_com(img)

        h_dict = self.horizontal_length(res, mask, cxy)
        v_dict = self.vertical_length(res, mask, cxy)
        sw_dict = self.ne_sw_length(res, mask, cxy)
        se_dict = self.nw_se_length(res, mask, cxy)

        if self.report:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            fig.set_size_inches(8, 8)
            plt.imshow(img)

            plt.arrow(h_dict['Extent'][0], float(cxy[1]), h_dict['Extent'][-1] - h_dict['Extent'][0], 1, color='blue', length_includes_head=True, head_width=5)
            plt.arrow(float(cxy[0]), v_dict['Extent'][0], 1, v_dict['Extent'][-1] - v_dict['Extent'][0], color='orange', length_includes_head=True, head_width=5)

            se_arrow = [(se_dict['End'][0][0]-1)+se_dict['Extent'][0]/np.sqrt(2),
                        (se_dict['End'][1][0]-1)+se_dict['Extent'][0]/np.sqrt(2),
                        (se_dict['Extent'][-1] - se_dict['Extent'][0])/np.sqrt(2)]

            print(se_arrow)
            sw_arrow = [(sw_dict['End'][0][0]+1)-sw_dict['Extent'][0]/np.sqrt(2),
                        (sw_dict['End'][1][0]-1)+sw_dict['Extent'][0]/np.sqrt(2),
                        (sw_dict['Extent'][-1] - sw_dict['Extent'][0])/np.sqrt(2)]

            plt.arrow(se_arrow[0], se_arrow[1], se_arrow[2], se_arrow[2], color='purple', length_includes_head=True, head_width=5)
            plt.arrow(sw_arrow[0], sw_arrow[1], -sw_arrow[2], sw_arrow[2], color='yellow', length_includes_head=True, head_width=5)

            plt.legend([str(np.round(h_dict['Distance'], 2)) + 'mm',
                        str(np.round(v_dict['Distance'], 2)) + 'mm',
                        str(np.round(sw_dict['Distance'],2)) + 'mm',
                        str(np.round(se_dict['Distance'],2)) + 'mm'])
            plt.axis('off')
            plt.title('Geometric Accuracy for Slice 5')

            img_path = os.path.realpath(os.path.join(self.report_path, f'{self.key(dcm)}.png'))
            fig.savefig(img_path)
            self.report_files.append(img_path)

        return h_dict['Distance'], v_dict['Distance'], sw_dict['Distance'], se_dict['Distance']