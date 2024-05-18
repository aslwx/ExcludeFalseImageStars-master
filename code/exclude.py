from image2circles_cpda import *
from astropy.io import fits
from astropy import io
import os
import matplotlib.pyplot as plt


def exclude(pic):
    path_str, file_name = os.path.split(pic)
    name, ext = os.path.splitext(file_name)
    if ext == '.fits':
        with fits.open(pic) as hdul:
            img = hdul[0].data
    else:
        img = io.loadmat(pic)['imgPixels']
    circles, edge_out = image2circles_cpda(img)

    # 初始化
    img_1 = np.zeros((1024, 1024), dtype=np.uint8)

    # 生成图像坐标网格
    x, y = np.meshgrid(np.arange(img_1.shape[1]), np.arange(img_1.shape[0]))

    # 对于每个圆，将对应的区域设置为1
    for circle in circles:
        center_x, center_y, radius = circle
        mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
        img_1[mask] = 1

    return img_1



