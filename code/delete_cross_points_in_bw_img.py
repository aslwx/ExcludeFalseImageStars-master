import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation, label, generate_binary_structure


# 输入的二值图像中删除指定位置的交叉点及其周围的像素，并对处理后的图像进行区域开操作。
def delete_cross_points_in_bw_img(bw_img_in, cross_points):
    # 如果存在交叉点，将交叉点本身以及四邻域像素置为零
    if cross_points.shape[0] != 0:
        for j in range(cross_points.shape[0]):
            bw_img_in[cross_points[j, 0], cross_points[j, 1]] = 0
            bw_img_in[cross_points[j, 0] + 1, cross_points[j, 1]] = 0
            bw_img_in[cross_points[j, 0] - 1, cross_points[j, 1]] = 0
            bw_img_in[cross_points[j, 0], cross_points[j, 1] + 1] = 0
            bw_img_in[cross_points[j, 0], cross_points[j, 1] - 1] = 0

    # 置零后进行区域开操作,剔除连通区域像素小于20的
    structure_element = generate_binary_structure(2, 2)
    labeled_img, num_objects = label(bw_img_in, structure_element)
    bw_img_out = bw_img_in.copy()

    for i in range(1, num_objects + 1):
        component = labeled_img == i
        if np.sum(component) < 20:
            bw_img_out[component] = 0

    return bw_img_out


