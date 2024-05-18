import cv2
import numpy as np
from scipy.ndimage import label
from edge_edit import *
from scipy.ndimage import median_filter


def pre_processing(img):
    # PREPROCESSING
    # 图像预处理
    #  输入原图（double），输出经过中值滤波、求梯度、做阈值、处理四边、面积开操作的图像（logical）
    #
    #  尝试对原图对数压缩
    #  img = log(img + 1);
    #  figure;
    #  imshow(img, []);

    # 中值滤波
    med_img = medfilt2(img, (3, 3))
    # Roberts算子梯度
    ##########################
    # 存在问题
    gra_med_img = edge_edit(med_img, 'roberts')
    gra_med_img = np.sqrt(gra_med_img)

    gra_med_img /= np.max(gra_med_img)  # 使最大值为1

    # 阈值处理
    thresh = 0.3
    gra_list = np.sort(gra_med_img.ravel())
    gra_list = gra_list[gra_list != 0]
    gra_99_percent = gra_list[round(0.9999 * len(gra_list)) - 1]

    _, bw_gra_med_img = cv2.threshold(gra_med_img, thresh * gra_99_percent, 1, cv2.THRESH_BINARY)

    # 四周宽度10像素设为0，消除图像边缘缺陷影响
    bw_gra_med_img[:10, :] = 0
    bw_gra_med_img[-10:, :] = 0
    bw_gra_med_img[:, :10] = 0
    bw_gra_med_img[:, -10:] = 0

    # % 尝试调换提取骨架和面积开操作的顺序，以在最后只用细化后的边缘拟合圆。失败，原因是由于边缘有一定厚度并且中心梯度较低，导致中央有空心的情况，最终在细化后的图像上可能会出现
    # % 双层的边缘。另外有厚度的边缘只要厚度均匀拟合效果几乎不会有差别（但是图上一般中间厚两边薄，导致拟合后重心偏移），其次有厚度的边缘可能可以更好地定位边缘中心，因为细化后边
    # % 缘可能会产生偏移。后面可能可以在面积开操作以后做一次闭操作以封闭孔洞再细化，然后使用细化后的边缘，但是可能又会太繁琐了。
    # % skel = bwmorph(bwGraMedImg, 'skel', Inf);
    # % figure;
    # % imshow(skel)
    # % figure;
    # % imshow(bwGraMedImg)

    # 连通集 - 阈值 - 开操作
    labeled_img, num_objects = label(bw_gra_med_img, structure=np.ones((3, 3), dtype=bool))  # 连通集，即边缘
    cc_cell = [np.column_stack(np.where(labeled_img == i + 1)) for i in range(num_objects)]  # 边缘具体位置，列表中每个元素为每个天体边缘的位置
    # 存储每个连通集的像素数，加1为了后面确定阈值时处理边缘情况
    ccSizes = np.ones(num_objects + 1, dtype=int)

    # 记录每个连通集像素数
    for j in range(num_objects):
        ccSizes[j] = len(cc_cell[j])

    # 将连通集像素数逆序排序
    ccSizes = np.sort(ccSizes)[::-1]

    thres_set = 0

    for j in range(num_objects):
        if ccSizes[j] / ccSizes[j + 1] > 6:
            thres_set = ccSizes[j]
            break

    if thres_set != 0:
        labeled_img, num_objects = label(bw_gra_med_img, structure=np.ones((3, 3), dtype=bool))
        bw_img_out = bw_gra_med_img.copy()
        for i in range(1, num_objects + 1):
            component = labeled_img == i
            sum = np.sum(component)
            if np.sum(component) < thres_set:
                bw_img_out[component] = 0

    return gra_med_img, bw_img_out


# 模拟matlab中的medfilt2函数
def medfilt2(image, kernel_size):
    pad_width = [(s//2, s//2) for s in kernel_size]
    padded_image = np.pad(image, pad_width, mode='constant', constant_values=(0, 0))
    result = median_filter(padded_image, size=kernel_size, mode='constant', cval=0)
    result = result[1:-1, 1:-1]
    return result

