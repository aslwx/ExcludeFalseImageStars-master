import numpy as np
from skimage import measure
import cv2
from scipy.ndimage import binary_erosion, binary_dilation, label, generate_binary_structure


# 删除骨骼图像中交叉点的函数。
def delete_cross_points_in_skel(skel_in, cross_points):
    # 如果有交叉点就消除交叉点
    if cross_points.any():
        skel_in[cross_points[:, 0], cross_points[:, 1]] = 0

    structure_element = generate_binary_structure(2, 2)
    # 消除交叉点后进行区域开操作，去除过小的轮廓
    labeled_img, num_objects = label(skel_in, structure_element)
    bw_img_out = skel_in.copy()
    for i in range(1, num_objects + 1):
        component = labeled_img == i
        sum = np.sum(component)
        if sum < 20:
            bw_img_out[component] = 0

    return bw_img_out
