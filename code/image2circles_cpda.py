from skimage.morphology import skeletonize
from pre_processing import *
from get_cross_points import *
from delete_cross_points_in_skel import *
from delete_cross_points_in_bw_img import *
from get_circles_from_bw_img import *
from cpda_simp import *


def image2circles_cpda(img):
    # 预处理
    gra_med_img, bw_img = pre_processing(img)
    edge_out = bw_img.copy()

    # 取骨架
    skel = skeletonize(bw_img)

    # 找Y型交叉点
    cross_points = get_cross_points(skel)

    # 删除骨架中的Y型交叉点并区域开操作
    skel = delete_cross_points_in_skel(skel, cross_points)

    # CPDA算法简化实现
    cout = cpda_simp(skel).astype(int)

    if len(cout) != 0:
        cout = cout.reshape((len(cout.shape), cout.shape[0]))
    # 删除预处理后图像中的交叉点
    bw_img = delete_cross_points_in_bw_img(bw_img, cross_points)

    # 切割拐点
    for j in range(cout.shape[0]):
        bw_img[cout[j, 0]-2:cout[j, 0]+3, cout[j, 1]-2:cout[j, 1]+3] = 0

    # 连通集 - 拟合 - 比较 - 合并
    circles = get_circles_from_bw_img(bw_img)

    return circles, edge_out
