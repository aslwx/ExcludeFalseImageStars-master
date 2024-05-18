import numpy as np
from scipy.ndimage import convolve


# FINDCROSSPOINTS 从骨骼图像找出交叉点
# 输入骨骼图像，输出交叉点位置。格式：2列，分别为行号r、列号c，行数为交叉点个数
def get_cross_points(skel):
    # 全1的相关模板
    h = np.ones((3, 3), dtype=int)

    # 相关操作，默认：结果大小与原图一致，边缘为0填充
    filted = convolve(skel.astype(np.uint8), h)

    # 交叉点大于3（包含自身、至少3个邻接点）
    c, r = np.where(filted.T > 3)

    cross_points = []

    # 遍历所有检测出的点，该点位于骨架中才为交叉点
    for j in range(len(r)):
        if skel[r[j], c[j]] == 1:
            cross_points.append([r[j], c[j]])

    # 转换为NumPy数组并返回
    return np.array(cross_points)
