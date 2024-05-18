import os
import glob
import time
import cv2
import numpy as np
from astropy.io import fits
from astropy import io
import pylab
import matplotlib
from image2circles_cpda import *


# pic = '../data/N1487595493_1.fits'
# path_str, file_name = os.path.split(pic)
# name, ext = os.path.splitext(file_name)
# with fits.open(pic) as hdul:
#     img = hdul[0].data
#
# plt.figure()
# plt.imshow(img, cmap='gray')
# plt.show()
# circles, edge_out = image2circles_cpda(img)

fits_files = [f for f in os.listdir('../data') if f.endswith('.fits')]
mat_files = [f for f in os.listdir('../data') if f.endswith('.mat')]
mat_files = [f + ' ' for f in mat_files]  # 补齐 mat 字符串
pics = ['../data/' + f for f in fits_files + mat_files]

time_recorder = np.zeros(200)
for k, pic in enumerate(pics, start=1):
    path_str, file_name = os.path.split(pic)
    name, ext = os.path.splitext(file_name)
    if ext == '.fits':
        with fits.open(pic) as hdul:
            img = hdul[0].data
    else:
        img = io.loadmat(pic)['imgPixels']

    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.show()

    # Circular Hough Transform
    # edges = cv2.Canny(img, 20, 50)
    # circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
    #                            param1=50, param2=30, minRadius=20, maxRadius=400)
    #
    # if circles is not None:
    #     circles = np.uint16(np.around(circles))
    #     for i in circles[0, :]:
    #         cv2.circle(img, (i[0], i[1]), i[2], (255, 0, 0), 2)
    #
    # plt.figure()
    # plt.imshow(img, cmap='gray')

    # The CPDA method
    # 你需要替换下面这一行为你的 CPDA 方法的实现
    circles, edge_out = image2circles_cpda(img)

    if circles is not None:
        for circle in circles:
            cv2.circle(img, (int(circle[0]), int(circle[1])), int(circle[2]), (255, 0, 0), 2)

    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.show()

    time_recorder[k - 1] = time.time()

print("Total time:", np.sum(np.diff(time_recorder)))